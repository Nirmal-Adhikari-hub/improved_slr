import torch
from pathlib import Path
from cslr.utils.logger import ExperimentLogger
import torch.nn as nn


CKPT_LAST = "last.pth"
CKPT_BEST = "best.pth"
CKPT_BEST_EMA = "best_ema.pth"

def _save_checkpoint(save_dir: Path, epoch: int, model, optim, sched, ema,
                     best_dev_wer: float, best_dev_wer_ema: float):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()),
        "optimizer_state_dict": optim.state_dict(),
        "best_dev_wer": best_dev_wer,
        "best_dev_wer_ema": best_dev_wer_ema,
    }
    if sched is not None:
        ckpt["scheduler_state_dict"] = sched.state_dict()
    if ema is not None:
        ckpt["ema_state_dict"] = ema.state_dict()
    torch.save(ckpt, str(save_dir / CKPT_LAST))

def _load_checkpoint_if_any(model, optim, sched, ema, save_dir: Path, exp: ExperimentLogger):
    ckpt_path = save_dir / CKPT_LAST
    if not ckpt_path.exists():
        exp.log_info("No checkpoint found. Starting from scratch.")
        return 0, float("inf"), float("inf")
    d = torch.load(str(ckpt_path), map_location="cpu")
    (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(d["model_state_dict"])
    if "optimizer_state_dict" in d: optim.load_state_dict(d["optimizer_state_dict"])
    if sched is not None and "scheduler_state_dict" in d: sched.load_state_dict(d["scheduler_state_dict"])
    if ema is not None and "ema_state_dict" in d: ema.load_state_dict(d["ema_state_dict"])
    best_dev_wer = d.get("best_dev_wer", float("inf"))
    best_dev_wer_ema = d.get("best_dev_wer_ema", float("inf"))
    start_epoch = int(d.get("epoch", 0))
    exp.log_info(f"Resumed from epoch {start_epoch}. Best dev WER: raw={best_dev_wer:.3f}, ema={best_dev_wer_ema:.3f}")
    return start_epoch, best_dev_wer, best_dev_wer_ema
