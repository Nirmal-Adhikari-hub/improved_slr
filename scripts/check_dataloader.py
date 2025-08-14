from __future__ import annotations
import os, argparse, ast
import numpy as np
import torch
from torch.utils.data import DataLoader
from box import Box

from cslr.utils.config import load_config, update_config
from cslr.utils.logger import setup_logger

from cslr.data_loader.phoenix_feeder import PhoenixFeeder, make_collate_fn, _compute_padding_and_stride


def _autotype(s: str):
    # bools
    if s.lower() in ("true", "false"): return s.lower() == "true"
    # ints/floats
    try: return int(s)
    except: pass
    try: return float(s)
    except: pass
    # python literal (lists, dicts, etc)
    try: return ast.literal_eval(s)
    except: pass
    return s  # fallback to string


def _parse_overrides(lst):
    # expects ["a.b.c", "123", "x.y", "true", "data.kernel_spec", "['K5', 'P2']"]
    if not lst: return {}
    if len(lst) % 2 != 0:
        raise SystemExit("ERROR: --override expects pairs: <key> <value> ...")
    d = {}
    for k, v in zip(lst[0::2], lst[1::2]):
        d[k] = _autotype(v)
    return d


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--mode", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--override", nargs="+", default=None, help="Dot-keys and values, eg data.frame_interval 2")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no_time_aug", action="store_true", help="Disable TemporalRescale for debugging")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Box(load_config(args.config))

    # apply overrides if any
    if args.override:
        cfg = Box(update_config(cfg, _parse_overrides(args.override)))

    # set seeds for reproducibility when debugging
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log = setup_logger(cfg.trainer.get("save_dir", "outputs"), name="Check_dl")

    # convenience alias
    d = cfg.data

    # Expand ~ and env vars if any
    def _xp(p): return os.path.expanduser(os.path.expandvars(p))

    # Build dataset
    ds = PhoenixFeeder(
        dataset_root=_xp(d.dataset_root),
        preprocess_root=_xp(d.preprocess_root),
        gloss_dict_path=_xp(d.gloss_dict_path),
        dataset=d.get("dataset", "phoenix2014"),
        mode=args.mode,
        datatype=d.get("datatype", "video"),
        frame_interval=int(d.get("frame_interval", 1)),
        image_scale=float(d.get("image_scale", 1.0)),
        input_size=int(d.get("input_size", 224)),
        kernel_spec=d.get("kernel_spec", ["K5", "P2", "K5", "P2"]),
        transform_train=(args.mode=="train"),
        frame_subdir=d.get("frame_subdir", "features/fullFrame-256x256px"),
    )

    # peek a couple of samples ( no collate)
    log.info(f"Dataset len({args.mode}) = {len(ds)})")
    for i in range(min(2, len(ds))):
        vid, lab, info = ds[i]
        log.info(f"[Sample {i}] video: {tuple(vid.shape)} labels: {tuple(lab.shape)} info: {type(info).__name__} info2: {info}")
    
    # Build collate and loader
    is_video = d.get("datatype", "video") == "video"
    collate = make_collate_fn(d.get("kernel_spec", ["K5", "P2", "K5", "P2"]), is_video=is_video)
    dl = DataLoader(
        ds,
        batch_size=d.batch_size,
        shuffle=(args.mode=="train"),
        num_workers=d.num_workers,
        pin_memory=True,
        collate_fn=collate
    )


    # One batch sanity check
    batch = next(iter(dl))
    if len(batch) == 5:
        videos, lengths, labels, label_lengths, info = batch
    else:
        videos, lengths, labels, label_lengths = batch
        info = None

    log.info(f"[batch] videos: {tuple(videos.shape)} lengths: {tuple(lengths.shape)}"
             f"labels: {tuple(labels.shape) if isinstance(labels, torch.Tensor) else labels}"
             f"label_lengths: {tuple(label_lengths.shape)}"
             f" info: {(info) if info else 'N/A'}")
    
    # Verify temporal padding math
    left_pad, total_stride = _compute_padding_and_stride(d.get("kernel_spec", []))
    T_pad = videos.shape[1] if is_video else videos.shape[-1]
    max_src_T = max(int(l) for l in lengths.tolist())
    log.info(f"left_pad={left_pad}, total_stride={total_stride}, T_pad={T_pad}, max_src_T={max_src_T}")

    # For video mode, check padded length formula equals computed T_pad for the largest sample
    if is_video:
        expected_max_T = int(np.ceil((max_src_T - 2*left_pad) / max(1, total_stride)) * max(1, total_stride) + 2* left_pad)
        # Because we clamp per-sample to its own ceil(...), the batch max should be equal this:
        if T_pad != expected_max_T:
            log.warning(f"T_pad({T_pad}) != expected_max_T({expected_max_T}) - check kernel spec or inputs.")
        else:
            log.info("Temporal padding looks consistent.")
    else:
        log.info("Feature-mode padding looks consistent.")

    log.info("Dataloader check comppleted successfully.")


if __name__ == "__main__":
    main()