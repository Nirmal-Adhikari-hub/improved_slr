from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from box.box import Box
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict

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


def main():
    args = parse_args()
    cfg = Box(apply_overrides(load_config(args.config), args.override))

    save_dir = Path(cfg.trainer.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    model = build_model(cfg).to(device)

    ckpt_path = Path(cfg.sanity_checks.author_ckpt_path)
    if ckpt_path.exists():
        print(f"Loading the checkpoint from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        state_dict = state_dict['model_state_dict']
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            # Handle cases where the original checkpoint might have a 'module.' prefix
            # from DataParallel, which we should remove first.
            if k.startswith('module.'):
                k = k[len('module.'):]
            name = 'inner.' + k  # Add the 'inner.' prefix
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        print("✅ Checkpoint loaded successfully!")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    print(f"Loading data ...")
    train_loader, val_loader, test_loader = build_loaders(cfg, kernel_spec=model.kernel_spec)
    
    if cfg.sanity_checks.set == "test": loader = test_loader
    elif cfg.sanity_checks.set == "dev": loader = val_loader
    else: loader = train_loader

    video_index_to_eval = int(cfg.sanity_checks.index)

    print(  loader.dataset[video_index_to_eval][-1])

    single_item_dataset = Subset(loader.dataset, [video_index_to_eval])
    single_item_loader = DataLoader(
        single_item_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=loader.collate_fn
    )

    print(f"Evaluating sample at index {video_index_to_eval} from the test set...")

    metrics = evaluate_epoch(
        model=model,
        loader=single_item_loader,
        device=device,
        compute_loss=False,
        log_examples=1
    )

    print("\n--- Evaluation Complete ---")
    print(f"Metrics for video at index {video_index_to_eval}:")
    print(f"Word Error Rate (WER): {metrics.get('wer', 'N/A'):.2f}%")
    print("\n✅ Pipeline check complete. Compare the 'Ref' and 'Hyp' above with your expected output.")

if __name__ == "__main__":
    main()
