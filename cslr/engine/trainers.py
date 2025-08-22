from __future__ import annotations
from typing import Dict, Optional
import time
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR


def train_one_epoch(
        model: nn.Module,
        loader,
        device: torch.device,
        optim: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        grad_accum: int,
        amp_enabled: bool,
        sched: Optional[torch.optim.lr_scheduler._LRScheduler],
        ema, # ModelEMA or None
        log_every: int,
        exp, # ExperimentLogger
        epoch: int,
        global_step: int,
) -> Dict[str, float]:
    model.train().to(device)
    optim.zero_grad(set_to_none=True)
    epoch_loss, seen = 0.0, 0
    t0 = time.time()
    accum = 0
    n_batches = len(loader)
    is_onecycle = isinstance(sched, OneCycleLR)

    for it, batch in enumerate(loader, 1):
        if len(batch) == 5:
            vids, vid_lens, labels, lab_lens, _ = batch
        else:
            vids, vid_lens, labels, lab_lens = batch

        vids, vid_lens = vids.to(device, non_blocking=True), vid_lens.to(device)
        labels, lab_lens = labels.to(device), lab_lens.to(device)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            out = model(vids, vid_lens, label=labels, label_lgt=lab_lens)
            loss = model.compute_loss(out, labels, lab_lens)
            if loss.dim() > 0: loss = loss.mean()

        loss_scaled = loss.div(max(1, grad_accum))
        (scaler.scale(loss_scaled) if amp_enabled else loss_scaled).backward()
        accum += 1

        if accum ==  grad_accum:
            if amp_enabled:
                scaler.step(optim); scaler.update()
            else:
                optim.step()
            optim.zero_grad(set_to_none=True)
            
            if is_onecycle and sched is not None:
                sched.step() # per-step for OneCycleLR
            if ema is not None: ema.update(model)
            accum = 0

        # stats/logging
        bs = vids.size(0)
        epoch_loss += float(loss.item()) * bs
        seen += bs
        global_step += 1

        if it % log_every == 0:
            samples_per_sec = seen / (time.time() -  t0 + 1e-6)
            lr = optim.param_groups[0]["lr"]
            exp.log_info(f"[epoch {epoch} it {it}/{n_batches}] loss {loss.item():.4f} | samples/s {samples_per_sec:.1f} | lr {lr: .6f}")
            exp.log_scalars(
                {
                    "train/loss": float(loss.item()),
                    "train/samples_per_sec": float(samples_per_sec),
                    "train/lr": float(lr),
                    "meta/epoch": epoch
                },
                step=global_step
            )

    # flush remaining grads if any
    if accum > 0:
        if amp_enabled: scaler.step(optim); scaler.update()
        else: optim.step()
        optim.zero_grad(set_to_none=True)
        
        if is_onecycle and sched is not None:
            sched.step()
        if ema is not None: ema.update(model)

    avg = epoch_loss / max(1, seen)
    return {
        "avg_loss": avg,
        "global_step": global_step,
        "seen": seen
    }