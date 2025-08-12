# CSLR — Modern SlowFast-Compatible Training Stack

A clean, modular training pipeline for **Continuous Sign Language Recognition (CSLR)** that reproduces the SlowFast dataloading semantics **1:1** while modernizing the codebase for clarity, extensibility, and experiment velocity.

---

## 🚀 What this repo offers

* Train/evaluate a CSLR model with config-driven hyperparameters.
* Use the **same data protocol** as the original SlowFast pipeline (augmentation order, normalization, label mapping, collate logic).
* Swap datasets/machines by editing a **single config file**.
* Integrated logging via TensorBoard, automatic best checkpoint saving, and clean run management.

---

## 📊 What’s Implemented (So Far)

**Directory Structure**:

```bash
cslr_project/
  configs/
    dataset_phoenix.yaml      # dataset + training config (edit paths here)
    default.yaml              # generic starter config
  cslr/
    data_loader/
      video_aug.py            # port of utils/video_augmentation.py (used ops)
      phoenix_feeder.py       # modern port of BaseFeeder + collate
    models/
      conv2d_backbone.py      # per-frame ResNet backbone (starter)
      temporal_conv.py        # simple 1D temporal conv stack (starter)
      classifier.py           # weight-normalised linear head
      decoder.py              # greedy CTC decoder (baseline)
      slr_model.py            # wiring backbone + temporal + head
    trainers/
      trainer.py              # AMP, grad-accum, TB logging, best ckpt
    utils/
      config.py               # YAML/JSON loader + dot-override
      logger.py               # coloured console + file logs
      scheduler.py            # LR schedulers (cosine, step, one-cycle)
      metrics.py              # WER/CER utilities (DP Levenshtein)
  scripts/
    train.py                  # entry point
  outputs/                    # runs, logs, checkpoints
```

---

## 📖 Exact SlowFast Data Semantics

* Reads `preprocess/{dataset}/{split}_info.npy` (same format).
* Loads `gloss_dict.npy` mapping token → `[id, count]`, uses `[0]` as class ID.
* **Frame sampling**: random offset + frame\_interval stride (same).
* **Augmentation order (train)**:

  * `RandomCrop` → `RandomHorizontalFlip(0.5)` → `Resize` → `ToTensor` → `TemporalRescale`
* **Test aug**: `CenterCrop` → `Resize` → `ToTensor`
* **Normalization**: `((x/255) - 0.45) / 0.225` (same as original).
* **Collate**: sorted by length, padded with left-padding and edge-padding based on `kernel_spec`.
* **Labels**: concatenated (CTC-style), not padded.

---

## 📃 Config-First UX + CLI Override

Edit YAML configs like:

```yaml
# configs/dataset_phoenix.yaml
...
data:
  dataset_root: /nas/Dataset/Phoenix/phoenix2014-release/phoenix-2014-multisigner
  frame_subdir: features/fullFrame-256x256px
  preprocess_root: ./preprocess
  gloss_dict_path: ./preprocess/phoenix2014/gloss_dict.npy
```

Override any field inline at the CLI:

```bash
python cslr_project/scripts/train.py -c configs/dataset_phoenix.yaml \
  --override trainer.epochs=1 data.batch_size=2
```

---

## 🚀 Training QoL Features

* AMP mixed-precision support
* Gradient accumulation
* TensorBoard scalars
* Periodic and best model checkpointing
* Clean and informative logging (console + file)

---

## ✅ SlowFast Code Comparison (1:1 Parity)

| Original SlowFast Repo                         | This Repo (Modern Port)                             |
| ---------------------------------------------- | --------------------------------------------------- |
| `dataset/dataloader_video.py::BaseFeeder`      | `cslr/data_loader/phoenix_feeder.py::PhoenixFeeder` |
| Augmentations in `utils/video_augmentation.py` | `cslr/data_loader/video_aug.py`                     |
| `inputs_list = np.load(.../{mode}_info.npy)`   | Same (configurable path via `preprocess_root`)      |
| `gloss_dict.npy` mapping via `[0]`             | Same                                                |
| Random offset + frame\_interval subsampling    | Same                                                |
| Phoenix/CSL/CSL-Daily path rules               | Same (toggle via `frame_subdir`)                    |
| Normalization `(x/255 - 0.45)/0.225`           | Same                                                |
| Collate function based on `kernel_spec`        | Same (fully dynamic, no globals)                    |
| Return `(video, len, label, label_len, info)`  | Same tensors + order                                |

---

## 📁 Data Layout Requirements

**Dataset Root**:

```
/nas/Dataset/Phoenix/phoenix2014-release/phoenix-2014-multisigner/
  features/
    fullFrame-210x260px/  # or fullFrame-256x256px
      train/01April.../*.png
```

**Preprocess Root**:

```
preprocess/
  phoenix2014/
    train_info.npy
    dev_info.npy
    test_info.npy
    gloss_dict.npy
```

---

## 📊 Kernel Spec Decoding

A kernel spec like:

```yaml
kernel_spec: ["K5", "P2", "K5", "P2"]
```

Means:

* conv1d(kernel=5) → pool(stride=2) → conv1d(kernel=5) → pool(stride=2)

We compute:

* `left_pad = sum of (kernel_size-1)/2 * total_previous_stride`
* `total_stride = product of all pool strides`

This determines exact shape matching for CTC-compatible batching.

---

## 🔧 Default Model (Starter)

* Per-frame ResNet backbone → (B, T, D)
* 1D temporal conv stack → (T, B, F)
* Classifier (weight-normalised linear)
* Greedy CTC decoder for baseline eval

---

## 🕊️ Roadmap

### Phase 1 — Complete Parity

* [x] Port dataset/aug/collate w/ exact SlowFast semantics
* [x] Config-driven everything
* [x] AMP, grad-accum, TB, best-ckpt
* [x] Auto-derive kernel\_spec from temporal model
* [ ] Plug in SlowFast backbone from `slr_network.py`
* [ ] Replicate loss dict (SeqCTC, ConvCTC, SeqKD)
* [ ] Add BiLSTM matching authors
* [ ] Reproduce authors' WER on dev/test

### Phase 2 — Diagnostics & Improvements

* [ ] Grad-CAM hooks for hand/face/pose
* [ ] TensorBoard CAM grid export
* [ ] Evaluation scripts: train\_eval / dev / test
* [ ] Beam-search decoder parity
* [ ] DDP + DeepSpeed launcher

### Phase 3 — Research Features

* [ ] ROI-aware diffusion restoration (hands/face)
* [ ] Feature-only training path
* [ ] Curriculum over frame\_interval, seq\_len, and augmentation

---

## 🔗 TensorBoard

```bash
tensorboard --logdir cslr_project/outputs/tb
```

---

Happy training! 🚀
