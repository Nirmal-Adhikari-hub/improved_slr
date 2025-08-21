from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from box.box import Box

from cslr.utils.config import load_config
from cslr.utils.parse_args import parse_args
from cslr.utils.cli import apply_overrides
from cslr.utils.scheduler import create_scheduler
from cslr.utils.logger import ExperimentLogger
from cslr.utils.ema import ModelEMA
from cslr.utils.metric_file_logger import ScalarFileLogger
from cslr.utils.checkpointing import _load_checkpoint_if_any, _save_checkpoint
from cslr.utils.checkpointing import CKPT_BEST, CKPT_BEST_EMA, CKPT_LAST

from cslr.models.build_model import build_model
from cslr.data_loader.build_dataloader import build_loaders
from cslr.engine.trainers import train_one_epoch
from cslr.engine.evaluate import evaluate_epoch
from cslr.engine.authors_eval import evaluate_split_authors


def main():
    args = parse_args()
    cfg = Box(apply_overrides(load_config(args.config), args.override))

    save_dir = Path(cfg.trainer.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    exp = ExperimentLogger(cfg, save_dir=str(save_dir), name="train_min")
    csv_logger = ScalarFileLogger(base_dir=str(save_dir / "metrics"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.trainer.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # model
    model = build_model(cfg).to(device)
    exp.watch_model(model)
    exp.log_info(f"Model built. kernel_spec from model: {model.kernel_spec}")

    # data
    train_loader, dev_loader, test_loader = build_loaders(cfg, kernel_spec=model.kernel_spec)

    # evaluation setups
    use_auth = bool(cfg.eval.get("use_authors_eval", True))
    work_dir = str(save_dir) if str(save_dir).endswith("/") else f"{save_dir}/"

    # optim/sched
    opt_cfg = cfg.get("optimiser", Box({}))
    if opt_cfg.get("name", "adamw").lower() == "adamw":
        optim = torch.optim.AdamW(model.parameters(),
                                  lr=opt_cfg.get("lr", 1e-4),
                                  weight_decay=opt_cfg.get("weight_decay", 1e-2))
    else:
        optim = torch.optim.SGD(model.parameters(),
                                lr=opt_cfg.get("lr", 1e-3),
                                momentum=opt_cfg.get("momentum", 0.9),
                                weight_decay=opt_cfg.get("weight_decay", 1e-4))
    sched = create_scheduler(optim, cfg.get("scheduler", {"name": "none"}))
    is_onecycle = isinstance(sched, OneCycleLR)

    # EMA
    ema = ModelEMA(model, decay=float(cfg.trainer.get("ema_decay", 0.999)), device=device) if bool(cfg.trainer.get("ema", False)) else None

    # resume
    start_epoch = 0
    best_dev_wer = float("inf")
    best_dev_wer_ema = float("inf")
    if bool(cfg.trainer.get("resume", True)):
        start_epoch, best_dev_wer, best_dev_wer_ema = _load_checkpoint_if_any(model, optim, sched, ema, save_dir, exp)

    # train loop
    epochs     = int(cfg.trainer.get("epochs", 1))
    grad_accum = int(cfg.trainer.get("grad_accum_steps", 1))
    log_every  = int(cfg.trainer.get("log_interval", 50))
    run_test_mid = bool(cfg.eval.get("run_test_mid_epoch", True))
    log_examples = int(cfg.eval.get("log_examples", 0))
    best_key = str(cfg.trainer.get("save_best_by", "dev_wer_ema"))

    global_step = 0
    for epoch in range(start_epoch + 1, epochs + 1):
        # ---- train ----
        tr = train_one_epoch(
            model=model, loader=train_loader, device=device,
            optim=optim, scaler=scaler, grad_accum=grad_accum, amp_enabled=amp_enabled,
            sched=sched, ema=ema, log_every=log_every, exp=exp,
            epoch=epoch, global_step=global_step
        )
        global_step = tr["global_step"]
        exp.log_info(f"[EPOCH {epoch}] train_avg_loss={tr['avg_loss']:.4f} seen {tr['seen']}/{len(train_loader.dataset)}")
        exp.log_scalars({"train/epoch_avg_loss": float(tr["avg_loss"])}, step=global_step)
        csv_logger.log("train", {"epoch": epoch, "train_avg_loss": float(tr["avg_loss"])})

        # epoch-wise schedulers
        if (sched is not None) and (not is_onecycle):
            sched.step()

        # ---- dev eval (WER + loss) ----
        exp.log_info(f"[epoch {epoch}] dev eval...")
        m_dev = evaluate_split_authors(cfg, 
                                       dev_loader, 
                                       model, 
                                       device, 
                                       mode="dev",  
                                       work_dir=work_dir, 
                                       python_evaluate=True)
                                       
        exp.log_scalars({"dev/wer": m_dev["wer"]}, step=global_step)

        m_dev_ema = None
        if ema is not None:
            exp.log_info(f"[epoch {epoch}] dev eval EMA ...")
            m_dev_ema = evaluate_split_authors(
                                        cfg, 
                                        dev_loader, 
                                        ema.ema, 
                                        device, 
                                        mode="dev", 
                                        work_dir=work_dir, 
                                        python_evaluate=True, 
                                        tag=f"ema-e{epoch}")
            
            exp.log_scalars({"dev/wer_ema": m_dev_ema["wer"]}, step=global_step)

        csv_logger.log("dev", {
            "epoch": epoch,
            "dev_wer": m_dev["wer"],
            "dev_loss": m_dev.get("loss", float("nan")),
            "dev_wer_ema": (m_dev_ema["wer"] if m_dev_ema else float("nan")),
            "dev_loss_ema": (m_dev_ema.get("loss", float("nan")) if m_dev_ema else float("nan")),
        })

        # ---- test eval (RAW test every epoch) ----
        exp.log_info(f"[epoch {epoch}] test eval...")
        m_test = evaluate_split_authors(cfg, 
                                        test_loader, 
                                        model, 
                                        device, 
                                        mode="test",
                                        work_dir=work_dir,
                                        python_evaluate=True)
        
        exp.log_scalars({"test/wer": m_test["wer"]},step=global_step)

        # EMA test only at the very end (after last epoch)
        is_last = (epoch == epochs)
        if is_last and ema is not None:
            exp.log_info(f"[epoch {epoch}] FINAL test eval (EMA)...")
            m_test_ema = evaluate_split_authors(cfg, 
                                                test_loader, 
                                                ema.ema, 
                                                device, 
                                                mode="test", 
                                                work_dir=work_dir, 
                                                python_evaluate=True, 
                                                tag="ema-final")
            exp.log_scalars({"test/wer_ema": m_test_ema["wer"]}, step=global_step)

        # --------- checkpointing (RAW+EMA) ------------
        _save_checkpoint(save_dir, 
                         epoch,
                         model,
                         optim,
                         sched,
                         ema,
                         best_dev_wer,
                         best_dev_wer_ema)
        exp.log_info("saved checkpoint: last.pth")
        if ema is not None:
            torch.save(ema.state_dict(), str(save_dir / "last_ema.pth"))






        csv_logger.log("test", {
            "epoch": epoch,
            "test_wer": m_test["wer"],
            "test_loss": m_test.get("loss", float("nan")),
            "test_wer_ema": (m_test_ema["wer"] if m_test_ema else float("nan")),
            "test_loss_ema": (m_test_ema.get("loss", float("nan")) if m_test_ema else float("nan")),
        })

        # ---- checkpoints ----
        _save_checkpoint(save_dir, epoch, model, optim, sched, ema, best_dev_wer, best_dev_wer_ema)
        exp.log_info("saved checkpoint: last.pth")

        if best_key == "dev_wer_ema" and m_dev_ema is not None:
            current = m_dev_ema["wer"]
            if current < best_dev_wer_ema:
                best_dev_wer_ema = current
                torch.save(ema.state_dict(), str(save_dir / CKPT_BEST_EMA))
                exp.log_info(f"New BEST EMA dev WER: {best_dev_wer_ema:.3f}. Saved {CKPT_BEST_EMA}.")
        else:
            current = m_dev["wer"]
            if current < best_dev_wer:
                best_dev_wer = current
                torch.save((model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()),
                           str(save_dir / CKPT_BEST))
                exp.log_info(f"New BEST dev WER: {best_dev_wer:.3f}. Saved {CKPT_BEST}.")

    exp.log_info("Training complete.")
    exp.close()

if __name__ == "__main__":
    main()
