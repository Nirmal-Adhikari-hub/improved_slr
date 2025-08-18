from __future__ import annotations
import torch, time
from torch import nn
from pathlib import Path
from box.box import Box
import wandb
import os, datetime

from torch.utils.tensorboard import SummaryWriter

from cslr.utils.config import load_config
from cslr.utils.logger import setup_logger
from cslr.utils.scheduler import create_scheduler
from cslr.utils.parse_args import parse_args
from cslr.models.build_model import build_model
from cslr.data_loader.build_dataloader import build_loaders
from cslr.utils.cli import apply_overrides

from cslr.models.legacy_slowfast_adapter import LegacySlowFastSLR


def main():
    args = parse_args()
    raw = apply_overrides(load_config(args.config), args.override)
    cfg = Box(raw)

    # W&B init
    run_name = f"{cfg.model.name}-"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb.init(
        project="cslr-slowfast",
        name=run_name,
        config=dict(cfg),
        tags=[cfg.data.dataset, "phoenix2014", "slowfast", "step1"],
    )
    
    save_dir = Path(cfg.trainer.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(cfg.trainer.save_dir, name="train_min")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.trainer.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # build model
    model = build_model(cfg)
    # (optional) lgo gradients/params; can be heavy on big models -> set log_freq higher or skip
    wandb.watch
    # with open(save_dir / "model.txt", "w") as f:
    #     f.write(str(model))
    log.info(f"Model built. kernel_spec from model: {model.kernel_spec}")
    wandb.watch(model, log="gradients", log_freq=200)

    # build laoders with that kernel spec
    train_loader, dev_loader = build_loaders(cfg, kernel_spec=model.kernel_spec)

    # optimizer, sched, scaler
    opt_cfg = cfg.get("optimiser", Box({}))
    if opt_cfg.get("name", "adamw").lower() == "adamw":
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.get("lr",1e-4),
            weight_decay=opt_cfg.get("weight_decay", 1e-2)
        )
    else:
        optim = torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg.get("lr", 1e-3),
            momentum=opt_cfg.get("momentum",0.9),
            weight_decay=opt_cfg.get("weight_decay", 1e-4)
        )
    sched = create_scheduler(optim, cfg.get("scheduler", {"name":"none"}))
    ctc = nn.CTCLoss(
        blank=cfg.model.get("blank_id", 0),
        zero_infinity=True
    )

    # training step (minimal sketch)
    epochs = int(cfg.trainer.get("epochs", 1))
    grad_accum = int(cfg.trainer.get("grad_accum_steps", 1))
    log_every = int(cfg.trainer.get("log_interval", 50))
    
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optim.zero_grad(set_to_none=True)
        epoch_loss, seen, t0 = 0.0, 0, time.time()

        accum = 0 # micro-batches accumulated since last optimizer.step()

        for it, batch in enumerate(train_loader, 1):
            if len(batch) == 5:
                vids, vid_lens, labels, lab_lens, _ = batch
            else:
                vids, vid_lens, labels, lab_lens = batch
            vids, vid_lens = vids.to(device, non_blocking=True), vid_lens.to(device)
            labels, lab_lens = labels.to(device), lab_lens.to(device)
            model.to(device)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(
                    vids,
                    vid_lens,
                    label=labels,
                    label_lgt=lab_lens
                )
                loss = model.compute_loss(out, labels, lab_lens)
                if loss.dim() > 0: loss = loss.mean()

                # backward / step
            (scaler.scale(loss) if amp_enabled else loss).div_(max(1, grad_accum)).backward()
            accum += 1

            # when we have accumulated k micro-batches
            if accum == grad_accum:
                if amp_enabled:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                accum = 0

            # running stats
            bs = vids.size(0)
            epoch_loss += float(loss.item()) * bs
            seen += bs
            global_step += 1

            if it % log_every == 0:
                ips = seen / (time.time() - t0 + 1e-6)
                lr = optim.param_groups[0]["lr"]
                log.info(f"[epoch {epoch} it {it}/{len(train_loader)}]"
                         f"loss {loss.item():.4f} | imgs/s {ips:.1f} | lr {lr:.6f}")

                # wandb log
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/imgs_per_s": float(ips),
                        "train/lr": float(lr),
                        "epoch": epoch,
                    },
                    step=global_step,
                )
            
        # FLUSH: if epoch ended mid_window; use remaining grads once
        if accum > 0:
            if amp_enabled:
                scaler.step(optim); scaler.update()
            else:
                optim.step()
            optim.zero_grad(set_to_none=True)
                
        avg = epoch_loss / max(1, seen)
        log.info(f"[EPOCH {epoch}] train_avg_loss={avg:.4f} seen {seen}/{len(train_loader.dataset)}")
        wandb.log({"train/epoch_avg_loss": float(avg), "epoch": epoch}, step=global_step)

        # save the model checkpoint each epoch
        ckpt = {
            "epoch": epoch,
            "model_state_dict": (
                model.module.state_dict() if isinstance(model, nn.DataParallel)
                else model.state_dict()
            ),
            "optimizer_state_dict": optim.state_dict(),
        }

        ckpt_path = save_dir / "last.pth"
        torch.save(ckpt, str(ckpt_path))
        log.info("saved checkpoint: last.pt")
        try:
            art = wandb.Artifact("slr_last", type="model")
            art.add_file(str(ckpt_path))
            wandb.log_artifact(art)
        except Exception as e:
            log.warning(F"W&B artifact upload failed: {e}")
    
    log.info("Training complete. Final model saved to 'last.pth'.")
    wandb.finish()


if __name__ == "__main__":
    main()