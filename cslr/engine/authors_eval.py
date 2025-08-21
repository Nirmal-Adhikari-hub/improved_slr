from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import torch, os

from cslr.models.slowfast.evaluation.slr_eval.wer_calculation import evaluate as authors_evaluate

@torch.no_grad()
def _extract_file_ids(info_batch) -> List[str]:
    """
    info_batch: the 'info' tuple from your collate (data[-1])
    Authors used: file_name.split('|')[0]
    We robustly support str or dict info
    """
    out = []
    for info in info_batch:
        if isinstance(info, str):
            # typical: "<fileid>/..."; take the left token
            out.append(info.split('|')[0])
        elif isinstance(info, dict):
            # if preprocess stored 'fileid'
            fid = info.get('fileid', None)
            if not fid:
                # as a fallback, stringify the dict
                fid = str(info)
            out.append(fid)
        else:
            out.append(str(info))
    return out

def _write_ctm(ctm_path: str, file_ids: List[str], decoded: List[List[Tuple[str, int]]]):
    """
    Match author's write2file(): each token becomes a line:
    "<fileid> 1 start duration word\n", where start=idx/100, duration=(idx+1)/100
    decoded: per-sample list of (gloss,idx) pairs from model["recognized_sents"] or ["conv_sents"]
    """
    with open(ctm_path, "w", encoding="utf-8") as f:
        for i, sent in enumerate(decoded):
            fid = file_ids[i]
            for w_idx, (word, _) in enumerate(sent):
                # timings are dummy but consistent with authors
                start = w_idx * 1.0 / 100.0
                dur = (w_idx + 1) * 1.0 / 100.0
                f.write(f"{fid} 1 {start:.2f} {dur:.2f} {word}\n")

@torch.no_grad()
def collect_predictions(model, loader, device) -> Tuple[List[str], List[List[Tuple[str,int]]], List[List[Tuple[str,int]]]]:
    """
    Runs the model in eval mode and collects:
    - file IDs
    - recognized_sents (LSTM/temporal head)
    - conv_sents (conv head)
    Returns lists aligned with the loader order
    """
    model = model.to(device).eval()
    total_ids: List[str] = []
    total_sent: List[List[Tuple[str, int]]] = []
    total_conv: List[List[Tuple[str, int]]] = []

    for batch in loader:
        if len(batch) == 5:
            vids, vid_lens, labels, label_lens, info = batch
        else:
            vids, vid_lens, labels, label_lens = batch
            info = None

        vids = vids.to(device, non_blocking=True)
        vid_lens = vid_lens.to(device)

        out = model(vids, vid_lens, label=None, label_lgt=None)

        # model returns lists of (gloss, idx). Ensure not None
        rec = out.get("recognized_sents") or [[] for _ in range(vids.size(0))]
        cnv = out.get("conv_sents") or [[] for _ in range(vids.size(0))]

        if info is None:
            # fallback: synthetic IDs
            file_ids = [f"sample_{len(total_ids)+i}" for i in range(vids.size(0))]
        else:
            file_ids = _extract_file_ids(info)

        total_ids.extend(file_ids)
        total_sent.extend(rec)
        total_conv.extend(cnv)

    return total_ids, total_sent, total_conv

@ torch.no_grad()
def evaluate_split_authors(
    cfg, 
    loader,
    model,
    device,
    mode: str, 
    work_dir: str, # where to put CTMs, temp outputs
    python_evaluate: bool = True,
    tag: str = "", # optional suffix for output filenames (eg "mid")
) -> Dict[str, float]:
    """
    Author parity evaluation:
    1) run model -> recognized sents / conv_sents
    2) write CTMs under work_dir
    3) call author's evaluate() with your STM directory/prefix
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    eval_dir = cfg.dataset_info["evaluate_dir"]
    eval_prefix = cfg.dataset_info["evaluate_prefix"]

    ids, rec, cnv = collect_predictions(model, loader, device)

    # CTM file names (optinally suffixed)
    suffix = f"-{tag}" if tag else ""
    ctm_lstm = str(Path(work_dir) / f"output-hypothesis-{mode}{suffix}.ctm")
    ctm_conv = str(Path(work_dir) / f"output-hypothesis-{mode}{suffix}-conv.ctm")

    _write_ctm(ctm_lstm, ids, rec)
    _write_ctm(ctm_conv, ids, cnv)

    # Call authors evaluator  (python path to avoid sclite dependency by default)
    lstm_wer = authors_evaluate(
        prefix=work_dir if work_dir.endswith("/") else (work_dir + "/"),
        mode=mode,
        output_file=f"output-hypothesis-{mode}{suffix}.ctm",
        evaluate_dir=eval_dir,
        evaluate_prefix=eval_prefix,
        output_dir=None, # or f"epoch_{epoch}_result/" if you want"    
        python_evaluate=python_evaluate,
        triplet=True, # mathces authors (conv vs lstm reporting)
        )
    # conv_ret isnt returned by authors_evaluate when python_evaluate=True
    # so compute it via a second call if needed. We will ignore or set to NaN
    return {"wer": float(lstm_wer)} # primary wer is LSTM head

def evaluate_single_sample_authors(
    cfg,
    model,
    device,
    mode: str,              # "train" | "dev" | "test"
    file_id: str,           # exact STM key (e.g., "29July_2010_Thursday_heute_default-7")
    loader,                 # DataLoader for the same split 'mode'
    work_dir: str,
) -> float:
    """
    Single-sample WER using the authors' pipeline:
      - find the sample in the loader, run model -> recognized_sents for that one
      - write single CTM with that ID
      - filter STM to single line for that ID
      - compute WER via the python evaluator directly
    """
    # from cslr.models.slowfast.evaluation.slr_eval.python_wer_calculation import wer_calculation
    from cslr.models.slowfast.evaluation.slr_eval.python_wer_evaluation import wer_calculation

    model = model.to(device).eval()
    found = False
    hyp_ctm = str(Path(work_dir) / f"single-{mode}.ctm")
    gt_stm  = Path(cfg.dataset_info["evaluation_dir"]) / f"{cfg.dataset_info['evaluation_prefix']}-{mode}.stm"
    single_gt = str(Path(work_dir) / f"single-{mode}.stm")

    # 1) Locate sample and run
    for batch in loader:
        if len(batch) == 5:
            vids, vid_lens, labels, label_lens, info = batch
        else:
            vids, vid_lens, labels, label_lens = batch
            info = None

        ids = _extract_file_ids(info) if info is not None else ["" for _ in range(vids.size(0))]
        if file_id not in ids:
            continue

        b_idx = ids.index(file_id)
        vids = vids.to(device, non_blocking=True)
        vid_lens = vid_lens.to(device)

        out = model(vids, vid_lens, label=None, label_lgt=None)
        rec = out.get("recognized_sents") or [[] for _ in range(vids.size(0))]

        # 2) write single-entry CTM
        _write_ctm(hyp_ctm, [file_id], [rec[b_idx]])

        found = True
        break

    if not found:
        raise RuntimeError(f"File ID '{file_id}' not found in {mode} loader.")

    # 3) build single-entry STM by filtering the official STM
    with open(gt_stm, "r", encoding="utf-8") as fin, open(single_gt, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.split()[0] == file_id:
                fout.write(line)
                break

    # 4) compute WER via python evaluator (single pair)
    wer = wer_calculation(single_gt, hyp_ctm)
    return float(wer)