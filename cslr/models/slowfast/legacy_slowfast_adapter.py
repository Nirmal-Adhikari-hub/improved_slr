import os
import numpy as np
import torch
import torch.nn as nn
from box import Box


class LegacySlowFastSLR(nn.Module):
    """
    Adapter that wraps slr_network.SLRModel so it works in the trainer of current framework.
    Exposes:
        - .kernel_spec: list[str] like ["K5", "P2", ...], for collate padding
        - forward(...) : returns dict with keys we expect
        - compute_loss(...) : calls author's criterion on outputs 
    """
    def __init__(self, cfg, gloss_dict_path: str):
        super().__init__()
        self.gloss_dict = np.load(gloss_dict_path, allow_pickle=True).item()
        num_classes = cfg.num_classes
        blank_id = cfg.get("blank_id", 0)
        self.blank_id = blank_id

        # ----- import authors model -----------
        # Expect your repo in PYTHONPATH, eg run with `PYTHONPATH=.`
        from cslr.models.slowfast.slr_network_multi import SLRModel

        # unpack model args from cfg
        c2d_type = cfg.get("c2d_type", "slowfast101")
        conv_type = cfg.get("conv_type", "conv1d")
        load_pkl = cfg.get("load_pkl", True)
        slowfast_cfg = cfg.get("slowfast_config", "SLOWFAST_64x2_R101_50_50.yaml")
        slowfast_args = cfg.get("slowfast_args", []) # e.g. ["SLOWFAST.ALPHA",4,"MODEL.DROPOUT_RATE",0.5]
        use_bn = cfg.get("use_bn", False)
        hidden_size = cfg.get("hidden_size", 1024)
        loss_weights = cfg.get("loss_weights", {"SeqCTC": 1.0}) # same defaults as authors
        weight_norm = cfg.get("weight_norm", True)
        share_classifier = cfg.get("share_classifier", True)

        # -------- instantiate authors model -----------------
        self.inner = SLRModel(
            num_classes=num_classes,
            c2d_type=c2d_type,
            conv_type=conv_type,
            load_pkl=load_pkl,
            slowfast_config=slowfast_cfg,
            use_bn=use_bn,
            slowfast_args=slowfast_args,
            hidden_size=hidden_size,
            gloss_dict=self.gloss_dict,
            loss_weights=loss_weights,
            weight_norm=weight_norm,
            share_classifier=share_classifier,
        )

        # ------- kernel spec for collate (exactly what their dataloading expects) -------
        ks = getattr(self.inner, "conv1d", None)
        self.kernel_spec = None
        if ks is not None and hasattr(ks, "kernel_size"):
            # In the author's code, Processor did: model.conv1d.kernel_size
            self.kernel_spec = ks.kernel_size # usually like ["K5", "P2", "K5", "P2"]
        if not self.kernel_spec:
            # Fallback to config (keeps us safe if module API differs)
            self.kernel_spec = cfg.get("kernel_spec", ["K5", "P2", "K5", "P2"])

        # store for loss
        self.loss_weights = loss_weights


    @torch.no_grad()
    def decode(self, logits, feat_len):
        # authors attach a decoder inside SLRModel; use their non-training path
        # forward() returns sentences when not training; in training we dont decode
        pass

    def forward(self, x, len_x, label=None, label_lgt=None):
        """
        x: (B, T, C, H, W) for video (matches our feeder output)
        len_x: LongTensor(B,) time lengths (post-collate accounting)
        """
        ret = self.inner(x, len_x, label, label_lgt=label_lgt)
        # map to keys our trainer expects
        out = {
            "feat_len": ret["feat_len"], # (B,)
            "conv_logits": ret["conv_logits"], # (T, B,C) or (B,T,C) depends on authors; they use batch_first=False
            "sequence_logits" : ret["sequence_logits"], # same layout as authors use in CTCLoss call
        }

        # optional extas for eval/printing
        if "recognized_sents" in ret:
            out["recognized_sents"] = ret["recognized_sents"]
        if "conv_sents" in ret:
            out["conv_sents"] = ret["conv_sents"]
        return out
    
    def compute_loss(self, ret_dict, labels, labels_lengths):
        """
        Use the author's criterion calculation for exact parity
        """
        return self.inner.criterion_calculation(ret_dict, labels, labels_lengths)
    
    @property
    def blank_id(self):
        return self._blank_id
    
    @blank_id.setter
    def blank_id(self, v):
        self._blank_id = v