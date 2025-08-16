import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Train CSLR")
    p.add_argument("-c","--config", required=True, type=str)
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()
