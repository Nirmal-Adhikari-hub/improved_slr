# cslr/utils/logger.py
import logging, sys, datetime
from pathlib import Path
from typing import Optional, Mapping, Any

# ---------- basic python logger ----------
def setup_logger(save_dir: str, name: str = "train", level=logging.INFO) -> logging.Logger:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    try:
        import colorlog
        cfmt = "%(log_color)s" + fmt
        ch.setFormatter(colorlog.ColoredFormatter(cfmt))
    except Exception:
        ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    fh = logging.FileHandler(Path(save_dir) / f"{name}.log")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    logger.propagate = False
    return logger


# ---------- unified experiment logger (TB + W&B) ----------
class ExperimentLogger:
    """
    One place for all logging:
      - Python logging (console+file)
      - TensorBoard (writer)
      - Weights & Biases (run, scalars, artifacts)
    TB is the source of truth when sync_tb=True (W&B mirrors TB automatically).
    """

    def __init__(self, cfg: Mapping[str, Any], save_dir: str, name: str = "train_min",
                 run_name: Optional[str] = None) -> None:
        self.cfg = cfg
        self.save_dir = Path(save_dir)
        self.logger = setup_logger(save_dir, name=name)

        lg = dict(cfg.get("logging", {}))
        self.use_tb: bool = bool(lg.get("tensorboard", True))
        self.use_wandb: bool = bool(lg.get("use_wandb", True))
        self.sync_tb: bool = bool(lg.get("sync_tb", True))
        self.project: str = lg.get("wandb_project", "cslr-slowfast")
        self.tags = lg.get("wandb_tags", [cfg.get("data", {}).get("dataset", "phoenix2014"),
                                          cfg.get("model", {}).get("name", "model")])
        self.watch_gradients: bool = bool(lg.get("wandb_watch_gradients", False))
        self.watch_log_freq: int = int(lg.get("wandb_watch_log_freq", 200))
        self.artifact_policy: str = str(lg.get("wandb_artifact_policy", "end")).lower()  # none|end|every_epoch

        # ---- TensorBoard ----
        self.tb_writer = None
        if self.use_tb:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.save_dir / "tb"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))

        # ---- W&B ----
        self.wandb_run = None
        if self.use_wandb:
            try:
                import wandb
                if run_name is None:
                    run_name = f"{cfg.get('model', {}).get('name','model')}-" + \
                               datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                self.wandb_run = wandb.init(
                    project=self.project,
                    name=run_name,
                    config=dict(cfg),
                    tags=self.tags,
                    sync_tensorboard=self.sync_tb,
                    dir=str(self.save_dir),
                )
            except Exception as e:
                self.logger.warning(f"Failed to init W&B: {e}. Continuing without W&B.")
                self.wandb_run = None

    def watch_model(self, model) -> None:
        if self.wandb_run and self.watch_gradients:
            try:
                import wandb
                wandb.watch(model, log="gradients", log_freq=self.watch_log_freq)
            except Exception as e:
                self.logger.warning(f"W&B watch failed: {e}")

    def log_scalars(self, scalars: Mapping[str, float], step: Optional[int] = None) -> None:
        # TB (source of truth when sync_tb=True)
        if self.tb_writer is not None:
            for k, v in scalars.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, float(v), step if step is not None else 0)
        # Direct W&B only if not mirroring TB
        if self.wandb_run is not None and not self.sync_tb:
            try:
                import wandb
                wandb.log(dict(scalars), step=step)
            except Exception as e:
                self.logger.warning(f"W&B log failed: {e}")

    def log_info(self, msg: str) -> None:
        self.logger.info(msg)

    # policy helper
    def _should_upload_epoch(self, epoch: int, final_epoch: int) -> bool:
        if self.wandb_run is None:
            return False
        if self.artifact_policy == "none":
            return False
        if self.artifact_policy == "every_epoch":
            return True
        # default: "end"
        return epoch == final_epoch

    def maybe_log_checkpoint(self, file_path: str, epoch: int, final_epoch: int) -> None:
        if not self._should_upload_epoch(epoch, final_epoch):
            return
        try:
            import wandb
            art = wandb.Artifact("slr_checkpoint", type="model")
            art.add_file(str(file_path))
            wandb.log_artifact(art)
        except Exception as e:
            self.logger.warning(f"W&B artifact upload failed: {e}")

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
