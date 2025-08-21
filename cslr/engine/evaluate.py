from __future__ import annotations
from typing import Dict, List, Tuple
import torch
from cslr.eval.metrics import wer_percent


def _split_labels(labels_flat: torch.Tensor, label_lens: torch.Tensor) -> List[List[int]]:
    out: List[int[int]] = []
    start = 0
    for ln in label_lens.tolist():
        ln = int(ln)
        out.append(labels_flat[start:start + ln].tolist() if ln > 0 else [])
        start += ln
    return out

def _inv_gloss_dict_from_model(model) -> Dict[int, str]:
    inv: Dict[int, str] = {}
    if hasattr(model, "gloss_dict"):
        for g, ids in model.gloss_dict.items():
            if ids: inv[int(ids[0])] = g
    return inv

@torch.no_grad()
def evaluate_epoch(
    model,
    loader,
    device: torch.device,
    compute_loss: bool = True,
    log_examples: int = 0,
) -> Dict[str, float]:
    """
    Use the model's own integrated decoder (beam via ctcdecode) to get hypothesis.
    Returns {"wer": float, "loss":float?}.
    """
    model = model.to(device).eval()
    inv_gloss = _inv_gloss_dict_from_model(model)

    total_S = total_D = total_I = total_N = 0
    loss_sum = 0.0
    seen = 0
    shown = 0

    for batch in loader:
        if len(batch) == 5:
            vids, vid_lens, labels_flat, label_lens, _ = batch
        else:
            vids, vid_lens, labels_flat, label_lens = batch

        vids = vids.to(device, non_blocking=True)
        vid_lens = vid_lens.to(device)
        labels_flat = labels_flat.to(device)
        label_lens = label_lens.to(device)

        # Forward in eval mode triggers authors' beam decoder outputs
        out = model(vids, vid_lens, label=None, label_lgt=None)

        # Hypothesis from model's integrated decoder
        # recognized_sents is: List[List[(gloss: str, idx: int)]]
        hyp_gloss_batch: List[List[str]] = []
        if "recognized_sents" in out and out["recognized_sents"] is not None:
            for seq in out["recognized_sents"]:
                hyp_gloss_batch.append([g for (g, _) in seq])
        else:
            # Rare fallback: no decoder output; treat as empty hyps
            hyp_gloss_batch= [[] for _ in range(vid_lens.shape[0])]

        # References (ids -> gloss)
        refs_ids_batch = _split_labels(labels_flat, label_lens)
        refs_gloss_batch: List[List[str]] = []
        for ref_ids in refs_ids_batch:
            refs_gloss_batch.append([inv_gloss[i] for i in ref_ids if i in inv_gloss])
        
        # WER
        for ref, hyp in zip(refs_gloss_batch, hyp_gloss_batch):
            wer, (S,D,I,N) = wer_percent(ref, hyp)
            total_S += S
            total_D += D
            total_I += I
            total_N += N
            if log_examples and shown < log_examples:
                print(f"Ref: {' '.join(ref)}")
                print(f"Hyp: {' '.join(hyp)}")
                shown += 1

        # Optional average loss on eval sets (authors' CTC loss)
        if compute_loss and hasattr(model, "compute_loss"):
            # Need logits in ret_dict + label tensors
            out_for_loss = out # forward already computed logits
            loss = model.compute_loss(out_for_loss, labels_flat, label_lens)
            if loss.dim() > 0: loss = loss.mean()
            B = vids.size(0)
            loss_sum += float(loss.item()) * B
            seen += B

    wer_total = 0.0 if total_N == 0 else 100.0 * (total_S + total_D + total_I) / total_N
    result = {"wer": wer_total}
    if compute_loss and seen > 0:
        result["loss"] = loss_sum / seen
    return result