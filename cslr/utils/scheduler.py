from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR

def create_scheduler(optim, cfg: dict):
    name = cfg.get("name", "none").lower()
    if name in ("", "none", None):
        return None
    if name == "step":
        return StepLR(
            optim, 
            step_size=cfg.get("step_size", 30), 
            gamma=cfg.get("gamma", 0.1)
            )
    if name == "cosine":
        return CosineAnnealingLR(
            optim, 
            T_max=cfg.get("t_max", 100), 
            eta_min=cfg.get("eta_min", 1e-6)
            )
    if name == "onecycle":
        return OneCycleLR(
            optim,
            max_lr=cfg.get("max_lr", 1e-3),
            epochs=cfg["epochs"],
            steps_per_epoch=cfg["steps_per_epoch"],
            pct_start=cfg.get("pct_start", 0.1)
            )
    raise ValueError(f"Unknown scheduler: {name}")