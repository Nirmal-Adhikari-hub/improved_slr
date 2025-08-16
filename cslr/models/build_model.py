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