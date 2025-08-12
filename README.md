CSLR — Modern SlowFast‑Compatible Training Stack
A clean, modular training pipeline for Continuous Sign Language Recognition (CSLR) that reproduces SlowFast dataloading semantics 1:1 while modernizing the codebase for clarity, extensibility, and experiment velocity.

<div align="center">
Train / evaluate with config‑driven hyper‑params · Exact SlowFast data protocol · Plug‑and‑play backbones · TensorBoard & best‑ckpts

</div>
Highlights
Exact SlowFast dataloading semantics (same augmentation order, normalization, label mapping, collate logic).

Config‑first: switch datasets/machines by editing a single YAML.

Modern trainer: AMP, grad accumulation, TensorBoard, smart checkpoints.

Modular: swap backbones (e.g., SlowFast) and temporal heads without touching the trainer.

What’s Implemented
graphql
Copy
Edit
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
      scheduler.py            # step/cosine/one-cycle LR schedulers
      metrics.py              # WER utilities (DP Levenshtein)
  scripts/
    train.py                  # entry point
  outputs/                    # runs, logs, checkpoints
Exact SlowFast Data Semantics (phoenix, CSL, CSL‑Daily)
Reads preprocess/{dataset}/{split}_info.npy (same structure as original).

Reads gloss_dict.npy and maps gloss → id via the first element [0] (same).

Frame sampling: random offset + frame_interval stride (same).

Augmentations (train):
RandomCrop → RandomHorizontalFlip(0.5) → Resize(image_scale) → ToTensor → TemporalRescale(0.2, frame_interval)
Test: CenterCrop → Resize → ToTensor

Normalization: ((x/255) - 0.45) / 0.225 (same, per channel).

Collate: sort by length, left‑pad + edge‑pad using kernel_spec (e.g., ["K5","P2","K5","P2"]); final total_stride matches original.
Labels are concatenated (CTC style), not padded.

Data Layout Expected
Dataset root (set in configs/dataset_phoenix.yaml → data.dataset_root):

bash
Copy
Edit
/path/to/phoenix2014-release/phoenix-2014-multisigner/
  features/
    fullFrame-256x256px/   # or: fullFrame-210x260px
      {train,dev,test}/
        01April_2010_Thursday_heute_default-0/1/*.png (or *.jpg)
        ...
Preprocess root (set in config → data.preprocess_root):

markdown
Copy
Edit
preprocess/
  phoenix2014/
    train_info.npy
    dev_info.npy
    test_info.npy
    gloss_dict.npy
*_info.npy example entry:

python
Copy
Edit
{
  'fileid': '01April_2010_Thursday_heute_default-0',
  'folder': 'train/01April_2010_Thursday_heute_default-0/1/*.png',
  'signer': 'Signer04',
  'label': 'GLOSS1 GLOSS2 ...',
  'num_frames': 176,
  'original_info': '...'
}
gloss_dict.npy: token -> [id, count] (we use [0] as the integer class id).

Switching resolution
Toggle data.frame_subdir:

features/fullFrame-256x256px (default), or

features/fullFrame-210x260px (matches many *_info.npy entries using *.png).

Install & Run
Environment
Python 3.9+

PyTorch (CUDA) + torchvision

Optional: colorlog (coloured logs)

Edit config paths
Open: cslr_project/configs/dataset_phoenix.yaml

data.dataset_root: dataset base path (machine‑specific)

data.preprocess_root: path to preprocess/

data.gloss_dict_path: path to gloss_dict.npy

data.frame_subdir: choose 210×260 or 256×256

Kick off a quick sanity run
bash
Copy
Edit
python cslr_project/scripts/train.py \
  -c cslr_project/configs/dataset_phoenix.yaml \
  --override trainer.epochs 1 data.batch_size 2
TensorBoard
bash
Copy
Edit
tensorboard --logdir cslr_project/outputs/tb
Shapes & Batching (CTC‑Ready)
Per sample from dataset:

video: (T, C, H, W) after aug; normalization applied.

labels: LongTensor[L] (no padding).

original_info: passthrough metadata.

Collate output:

padded_video: (B, T_pad, C, H, W) for videos (left‑pad + edge‑pad computed from temporal stack).

video_length: LongTensor[B] = ceil(T/total_stride)*total_stride + 2*left_pad.

labels_concat: LongTensor[sum(L_i)] (single 1D concat; CTC expects this).

label_length: LongTensor[B].

Kernel spec (must match temporal model)

["K5","P2","K5","P2"] ⇒ conv1d(k=5), pool/stride=2, conv1d(k=5), pool/stride=2.

We compute:

left_pad = Σ ((k_i − 1)/2 × product_of_previous_strides)

total_stride = Π (pool strides)

Current Model (Starter)
Default SLRModel (baseline for sanity):

Per‑frame 2D backbone (ResNet18/34/50) → (B, T, D)

Temporal 1D conv stack → (T, B, F) (time‑major for CTC)

Classifier with weight‑normalised linear head

Greedy CTC decoder (quick checks)

We will swap this backbone for your SlowFast and wire in the same temporal/loss heads as in the original slr_network.py.

How This Matches the SlowFast Repo
Original (SlowFast)	This repo (modern port)
dataset/dataloader_video.py::BaseFeeder	cslr/data_loader/phoenix_feeder.py::PhoenixFeeder
utils/video_augmentation.py	cslr/data_loader/video_aug.py (ported operators)
inputs_list = np.load(.../{mode}_info.npy).item()	Same (config: data.preprocess_root, data.dataset)
gloss_dict.npy via [0]	Same
Random offset + frame_interval	Same
Phoenix / CSL / CSL‑Daily path rules	Same (frame_subdir toggle)
((x/255) - 0.45) / 0.225	Same
Collate with kernel_sizes = ['K5','P2',...]	make_collate_fn(kernel_spec) (no global state)
Return (video, len, labels_concat, label_len, info)	Same shapes + order

Roadmap
Phase 1 — Complete Parity

Auto‑derive kernel_spec from the actual temporal model (no manual duplication).

Plug in SlowFast backbone (from slowfast_modules/*) with your YAML config.

Replicate loss dictionary (SeqCTC, ConvCTC, distillation/SeqKD) + BiLSTM head to match slr_network.py.

Reproduce authors’ WER (dev/test) end‑to‑end.

Phase 2 — Diagnostics & Improvements

Grad‑CAM hooks for hands/face/pose; CAM grids to TensorBoard.

Eval scripts mirroring “train_eval/dev/test” protocols + beam‑search parity.

DDP / torchrun launcher + optional DeepSpeed config.

Phase 3 — Research Features

ROI‑aware diffusion restoration (hands/face) with data‑consistent priors.

Optional feature‑only training path (pre‑extracted frame features).

Curriculum over frame_interval, sequence length, augmentation strength.