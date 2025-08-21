from __future__ import annotations
import argparse
from pathlib import Path
import torch
from box.box import Box

from cslr.utils.config import load_config
from cslr.utils.cli import apply_overrides
from cslr.models.build_model import build_model
from cslr.data_loader.build_dataloader import build_loaders
from cslr.engine.authors_eval import evaluate_single_sample_authors

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--mode", choices=["train", "dev", "test"], default="dev")
    ap.add_argument("--file_id", required=True, help="Exact STM ID (eg, 29July_2010_Thursday_heute_default-7)")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Box(apply_overrides(load_config(args.config), args.override))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device).eval()
    train_loader, dev_loader, test_loader = build_loaders(cfg, kernel_spec=model.kernel_spec)
    loader = {"trainer": train_loader, "dev": dev_loader, "test": test_loader}[args.mode]

    save_dir = Path(cfg.trainer.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    wer = evaluate_single_sample_authors(
        cfg,
        model,
        device,
        args.mode,
        args.file_id,
        loader,
        work_dir=str(save_dir)
    )
    print(f"[single-{args.mode}] {args.file_id}  WER: {wer:.2f}%")


if __name__ == "__main__":
    main()