# CSLR — Modern SlowFast-Compatible Training Stack

A clean, modular training pipeline for **Continuous Sign Language Recognition (CSLR)** that reproduces the SlowFast dataloading semantics **1:1** while modernizing the codebase for clarity, extensibility, and experiment velocity.

---

## What’s New (Aug 13–14, 2025)

**Major additions since the previous update:**

- **PhoenixFeeder (exact parity):** `data_loaders/phoenix_feeder.py` mirrors the original SlowFast dataloading (paths, aug order, normalization, label mapping, stride-aware padding).
- **Kernel-aware collate:** `make_collate_fn()` uses `kernel_spec` (e.g., `["K5","P2","K5","P2"]`) to compute **left_pad** and **total_stride** exactly like the authors’ repo.
- **Sanity checker:** `scripts/check_dataloader.py` validates shapes, per-sample lengths, and padding math. Supports **CLI overrides** and a `--no_time_aug` flag.
- **Cleaner config UX:** `utils/config.py` + `--override` lets you change machine/dataset parameters without editing YAML.
- **Logging:** Structured console + file logging via `utils/logger.py`.

> Verified on Phoenix2014: train transforms, dev/test transforms, normalization, CTC-style label concat, and collate behavior match the original pipeline.

---

## What this repo offers

- Train/evaluate a CSLR model with **config-driven** hyperparameters.
- Use the **same data protocol** as the original SlowFast pipeline (augmentation order, normalization, label mapping, collate logic).
- Swap datasets/machines by editing a **single config file** (or CLI `--override`).
- **Logging now:** console + file logs.  
  **Coming next:** TensorBoard, automatic best checkpoint saving, and full trainer harness.

---

## What’s Implemented (So Far)

**Directory Structure**

```bash
improved_slr/
  configs/
    dataset_phoenix.yaml        # dataset + dataloader config (edit paths here)
  data_loaders/
    video_aug.py                # port of used ops from utils/video_augmentation.py
    phoenix_feeder.py           # modern port of BaseFeeder + stride-aware collate
  scripts/
    check_dataloader.py         # sanity checker for shapes/padding/labels
  utils/
    config.py                   # YAML/JSON loader + CLI dot-overrides
    logger.py                   # coloured console + file logs
    scheduler.py                # LR schedulers (cosine, step, one-cycle)
    metrics.py                  # WER utilities (DP Levenshtein)
  outputs/                      # logs, (TB/ckpts will land here)
Model/trainer files will be added next; dataloader parity is locked in first to de-risk training.
```

## Exact SlowFast Data Semantics

- Reads `preprocess/{dataset}/{split}_info.npy` (same structure).

- Loads `gloss_dict.npy` mapping token → `[id, count]`; uses `[0]` as class ID.

- **Frame sampling**: random offset + `frame_interval` stride.

- **Augmentation order (train)**: `RandomCrop` → `RandomHorizontalFlip(0.5)` → `Resize` → `ToTensor` → `TemporalRescale`.

-** Dev/Test aug**: CenterCrop → Resize → ToTensor.

- **Normalization**: ((x/255) - 0.45) / 0.225.

- **Collate**: length-sort → pad with **left-pad** and edge-pad using kernel_spec.

- **Labels**: concatenated (CTC-style), not padded; companion label_lengths.

## Config-First UX + CLI Overrides

Edit YAML:

```yaml
# configs/dataset_phoenix.yaml
seed: 42
data:
  dataset: "phoenix2014"
  dataset_root: /nas/Dataset/Phoenix/phoenix2014-release/phoenix-2014-multisigner
  frame_subdir: features/fullFrame-256x256px
  preprocess_root: ./preprocess
  gloss_dict_path: ./preprocess/phoenix2014/gloss_dict.npy

  datatype: video
  frame_interval: 1
  image_scale: 1.0
  input_size: 224
  kernel_spec: ["K5","P2","K5","P2"]

  batch_size: 4
  num_workers: 4
```
Override fields inline:
```bash
# change frame interval
PYTHONPATH=. python scripts/check_dataloader.py -c configs/dataset_phoenix.yaml \
  --mode train --override data.frame_interval 2

# switch to 210x260 frames
PYTHONPATH=. python scripts/check_dataloader.py -c configs/dataset_phoenix.yaml \
  --mode train --override data.frame_subdir features/fullFrame-210x260px

# disable TemporalRescale in train (stable T for debugging)
PYTHONPATH=. python scripts/check_dataloader.py -c configs/dataset_phoenix.yaml \
  --mode train --no_time_aug
```
---
## Dataloader Sanity Check

Run:
```bash
# Train mode (TemporalRescale active; T varies per run)
PYTHONPATH=. python scripts/check_dataloader.py -c configs/dataset_phoenix.yaml --mode train

# Dev/Test mode (no TemporalRescale; stable T)
PYTHONPATH=. python scripts/check_dataloader.py -c configs/dataset_phoenix.yaml --mode dev
```

**Reading the logs**

- Different `T` for the **same sample** across runs in **train** is expected (TemporalRescale randomizes time).

- `left_pad` and `total_stride` come from `kernel_spec`.
For `["K5","P2","K5","P2"]` → `left_pad=6`, `total_stride=4`.

- Batched `T_pad` is computed via:
  ```ini
  right_pad = ceil(max_len/stride)*stride - max_len + left_pad
  T_pad     = max_len + left_pad + right_pad
  ```


- The checker verifies `T_pad` consistency; warnings indicate spec/input mismatches.

---

## SlowFast Code Comparison (1:1 Parity)
| Original SlowFast repo                    | This repo (modern port)          |
| ----------------------------------------- | -------------------------------- |
| `dataset/dataloader_video.py::BaseFeeder` | `data_loaders/phoenix_feeder.py` |
| `utils/video_augmentation.py` (used ops)  | `data_loaders/video_aug.py`      |
| `*_info.npy` loading                      | Same                             |
| `gloss_dict.npy` `[0]` → class id         | Same                             |
| Random offset + `frame_interval`          | Same                             |
| Phoenix/CSL/CSL-Daily path rules          | Same (`frame_subdir` toggle)     |
| `(x/255 - 0.45)/0.225` normalization      | Same                             |
| Collate with kernel-spec padding/stride   | Same                             |
| `(video, len, labels, label_lens, info)`  | Same tensors + order             |

---

## Data Layout Requirements
**Dataset Root**

```bash
/nas/Dataset/Phoenix/phoenix2014-release/phoenix-2014-multisigner/
  features/
    fullFrame-210x260px/
    fullFrame-256x256px/
      train/01April.../*.png
      dev/*.png
      test/*.png
```

**Preprocess Root**

```bash
preprocess/
  phoenix2014/
    train_info.npy
    dev_info.npy
    test_info.npy
    gloss_dict.npy
```

---

## Kernel Spec Decoding

Example:

```yaml
kernel_spec: ["K5","P2","K5","P2"]
```


Meaning:

- `K5` → conv1d with kernel 5

- `P2` → temporal downsample (stride 2)

- Sequence: conv5 → pool2 → conv5 → pool2

Computed:

- `left_pad = Σ ((k-1)/2) * (product of previous strides)`

- `total_stride = product of all pool strides`

Used to determine batch `T_pad` and CTC alignment.

---

## Default Model (Starter)

- (Coming next) Minimal baseline: per-frame CNN → temporal conv → (optional) BiLSTM → CTC.

- Adapter for authors’ **SlowFast R101** backbone will plug into the same trainer.

---

## Roadmap
### Phase 1 — Complete Parity

* [x] Port dataset/aug/collate with exact SlowFast semantics

* [x]  Config-driven everything + CLI --override

* [x]  Structured logging (console + file)

* [ ]  Full trainer (AMP, grad-accum, TensorBoard, best-ckpt)

* [ ]  Auto-derive kernel_spec from the temporal model

* [ ]  Plug in SlowFast backbone from slr_network.py

* [ ]  Replicate loss dict (SeqCTC, ConvCTC, SeqKD)

* [ ]  Add BiLSTM matching authors

* [ ]  Reproduce authors’ WER on dev/test

### Phase 2 — Diagnostics & Improvements

* [ ]  Grad-CAM hooks (hands/face/pose)

* [ ]  TensorBoard CAM grid export

* [ ]  Evaluation scripts: train_eval / dev / test

* [ ]  Beam-search decoder parity

* [ ]  DDP + DeepSpeed launcher

### Phase 3 — Research Features

* [ ]  ROI-aware diffusion restoration (hands/face)

* [ ]  Feature-only training path

* [ ]  Curriculum over frame_interval, seq_len, augmentation

---

