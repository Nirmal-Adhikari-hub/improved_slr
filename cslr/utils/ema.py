from copy import deepcopy
import torch

class ModelEMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay: float = 0.999, device=None):
        self.decay = float(decay)
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.ema.to(device)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + (1.0 - self.decay) * mv.detach())
            else:
                v.copy_(mv)

    def to(self, device):
        self.ema.to(device)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)
