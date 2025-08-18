import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Train CSLR")
    p.add_argument("-c","--config", 
                   type=str,
                   default="configs/model_legacy_slowfast.yaml",
                   help="Path to YAML config (default: configs/model_legacy_slowfast.yaml)")
    p.add_argument("--override", nargs="*", default=[], help="Dot-key overrides: key value ...")
    return p.parse_args()
