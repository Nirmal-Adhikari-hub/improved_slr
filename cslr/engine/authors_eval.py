# cslr/engine/authors_eval.py
from __future__ import annotations
from typing import List, Tuple, Dict
from pathlib import Path
import torch

from cslr.models.slowfast.evaluation.slr_eval.wer_calculation import evaluate as authors_evaluate
from cslr.models.slowfast.evaluation.slr_eval.python_wer_evaluation import wer_calculation as py_wer
from cslr.data_loader.phoenix_feeder import make_collate_fn

@torch.no_grad()
def _extract_file_ids(info_batch) -> List[str]:
    out = []
    for info in info_batch:
        # Authors used: file_name.split("|")[0]
        s = info if isinstance(info, str) else str(info)
        out.append(s.split("|")[0])
    return out

def _write_ctm(ctm_path: str, file_ids: List[str], decoded: List[List[Tuple[str,int]]]):
    with open(ctm_path, "w", encoding="utf-8") as f:
        for i, sent in enumerate(decoded):
            fid = file_ids[i]
            for w_idx, (word, _) in enumerate(sent):
                start = w_idx * 1.0 / 100.0
                dur   = (w_idx + 1) * 1.0 / 100.0
                f.write(f"{fid} 1 {start:.2f} {dur:.2f} {word}\n")

@torch.no_grad()
def evaluate_split_authors(
    cfg,
    loader,
    model,
    device,
    mode: str,                 # "dev" or "test"
    work_dir: str,
    python_evaluate: bool = True,
    tag: str = "",
) -> Dict[str, float]:
    """
    Exact authors flow: model -> CTM -> authors_evaluate(STM).
    Returns {'wer': <LSTM-head WER>}
    """
    model.eval().to(device)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    eval_dir    = cfg.dataset_info["evaluation_dir"]
    eval_prefix = cfg.dataset_info["evaluation_prefix"]

    total_ids: List[str] = []
    total_sent: List[List[Tuple[str,int]]] = []
    total_conv: List[List[Tuple[str,int]]] = []

    sum_loss = 0.0
    n_items = 0

    for batch in loader:
        if len(batch) == 5:
            vids, vid_lens, labels, label_lens, info = batch
        else:
            vids, vid_lens, labels, label_lens = batch
            info = None

        vids = vids.to(device, non_blocking=True)
        vid_lens = vid_lens.to(device)
        labels = labels.to(device)
        label_lens = label_lens.to(device)

        out = model(vids, vid_lens, label=labels, label_lgt=label_lens)

        # author's loss (SeqCTC + optional Slow/Fast/ConvCTC/Dist if enabled in cfg)
        loss = model.compute_loss(out, labels, label_lens)
        if loss.dim() > 0: loss = loss.mean()
        sum_loss += float(loss.item()) * vids.size(0)
        n_items += vids.size(0)

        rec = out.get("recognized_sents") or [[] for _ in range(vids.size(0))]
        cnv = out.get("conv_sents")       or [[] for _ in range(vids.size(0))]
        ids = _extract_file_ids(info) if info is not None else [f"sample_{len(total_ids)+i}" for i in range(vids.size(0))]

        total_ids.extend(ids)
        total_sent.extend(rec)
        total_conv.extend(cnv)

    suffix   = f"-{tag}" if tag else ""
    ctm_lstm = str(Path(work_dir) / f"output-hypothesis-{mode}{suffix}.ctm")
    ctm_conv = str(Path(work_dir) / f"output-hypothesis-{mode}{suffix}-conv.ctm")

    _write_ctm(ctm_lstm, total_ids, total_sent)
    _write_ctm(ctm_conv, total_ids, total_conv)

    # ensure prefix ends with '/'
    prefix = work_dir if work_dir.endswith("/") else (work_dir + "/")

    _ = authors_evaluate(
            prefix=prefix,
            mode=mode,
            output_file=Path(ctm_conv).name,   # e.g., "output-hypothesis-dev-<tag>-conv.ctm"
            evaluate_dir=eval_dir,
            evaluate_prefix=eval_prefix,
            output_dir=None,
            python_evaluate=python_evaluate,
            triplet=False,
        )

    lstm_wer = authors_evaluate(
        prefix=prefix,
        mode=mode,
        output_file=Path(ctm_lstm).name,
        evaluate_dir=eval_dir,
        evaluate_prefix=eval_prefix,
        output_dir=None,
        python_evaluate=python_evaluate,
        triplet=True,
    )
    avg_loss = (sum_loss / max(1, n_items))
    return {"wer": float(lstm_wer), "loss": float(avg_loss)}

@torch.no_grad()
def evaluate_single_by_index_authors(
    cfg,
    model,
    device,
    mode: str,              # "train" | "dev" | "test"
    index: int,             # dataset index within that split
    loader,                 # DataLoader for the same split
    work_dir: str,
    ):
    """
    Evaluate exactly one sample by DATASET INDEX (authors pipeline).
    We build a 1-sample batch with the same collate (kernel_spec-aware),
    write a 1-line CTM, filter STM to 1 line, and call the authors' python WER.
    """
    model.eval().to(device)
    ds = loader.dataset
    is_video = (getattr(cfg.data, "datatype", "video") == "video")

    # ---- get the raw sample from the dataset ----
    video_t, label_ids, info = ds[index]  # returns (T,C,H,W) tensor, label tensor, and original_info
    
    fid = (info if isinstance(info, str) else str(info)).split("|")[0]

    # ---- collate a batch of size 1 with the same padding spec ----
    collate = make_collate_fn(getattr(model, "kernel_spec", ["K5","P2","K5","P2"]), is_video=is_video)
    padded_video, video_length, labels, label_lengths, infos = collate([(video_t, label_ids, info)])

    vids      = padded_video.to(device, non_blocking=True)
    vid_lens  = video_length.to(device)

    # ---- forward & decode (beam inside the model) ----
    out = model(vids, vid_lens, label=None, label_lgt=None)
    rec = out.get("recognized_sents") or [[]]
    # write single-entry CTM
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    ctm = str(Path(work_dir) / f"single-{mode}.ctm")
    _write_ctm(ctm, [fid], [rec[0]])

    # build single-entry STM by filtering official STM
    gt_dir = Path(cfg.dataset_info["evaluation_dir"])
    gt_stm = gt_dir / f"{cfg.dataset_info['evaluation_prefix']}-{mode}.stm"
    single_gt = str(Path(work_dir) / f"single-{mode}.stm")
    with open(gt_stm, "r", encoding="utf-8") as fin, open(single_gt, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.split()[0] == fid:
                fout.write(line)
                break

    # compute WER via authors' python evaluator
    wer = py_wer(single_gt, ctm)
    return {"wer": float(wer), "info": info}
