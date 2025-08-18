from cslr.models.legacy_slowfast_adapter import LegacySlowFastSLR

def build_model(cfg):
    mcfg = cfg.model
    if mcfg.name == "legacy_slowfast":
        model = LegacySlowFastSLR(
            mcfg,
            gloss_dict_path=cfg.data.gloss_dict_path,
        )
    else:
        raise ValueError(f"Only legacy_slowfast included in this step.")
    return model


if __name__ == "__main__":
    from box.box import Box
    from cslr.utils.cli import apply_overrides
    from cslr.utils.parse_args import parse_args
    from cslr.utils.config import load_config
    from pathlib import Path

    args = parse_args()
    raw = apply_overrides(load_config(args.config), args.override)
    cfg = Box(raw)

    save_dir = Path(cfg.trainer.save_dir)
    save_dir = save_dir / "new"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "new_model.txt", "w") as f:
        model = build_model(cfg)
        f.write(str(model))