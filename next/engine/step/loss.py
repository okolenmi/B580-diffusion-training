"""The training loss: masked per-sample MSE, weighted per sample by a function of the noise level."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

T_THIRDS = ("loss_t_low", "loss_t_mid", "loss_t_high")


def per_sample_mse(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean squared error per sample; with a validity mask, only valid positions count."""
    squared = (pred.float() - target.float()).pow(2)
    if mask is None:
        return squared.flatten(1).mean(dim=1)
    m = mask.to(squared.dtype).expand_as(squared)
    return (squared * m).flatten(1).sum(dim=1) / m.flatten(1).sum(dim=1).clamp(min=1.0)


def thirds_of_range(per_sample: Tensor, t_frac: Tensor) -> dict[str, float]:
    """Mean loss over the low, middle and high third of the timestep range (a third with no samples is omitted)."""
    losses, fracs = per_sample.detach().flatten().float().cpu(), t_frac.detach().flatten().float().cpu()
    out = {}
    for i, name in enumerate(T_THIRDS):
        pick = (fracs >= i / 3) & ((fracs < (i + 1) / 3) | (i == 2))
        if bool(pick.any()):
            out[name] = float(losses[pick].mean())
    return out


class LossWeighting(ABC):
    @abstractmethod
    def weight(self, sigma: Tensor) -> Tensor:
        """One weight per sample from its noise level (k-diffusion sigma, so SNR is ``1 / sigma**2``)."""

    def apply(self, per_sample: Tensor, sigma: Tensor, extra: Tensor | None = None) -> Tensor:
        """Mean of the weighted losses; ``extra`` is one more per-sample multiplier (the time balance's)."""
        weighted = per_sample * self.weight(sigma.float().flatten().to(per_sample.device))
        return (weighted if extra is None else weighted * extra.to(weighted.device)).mean()


class Uniform(LossWeighting):
    def weight(self, sigma):
        return torch.ones_like(sigma)


class MinSNR(LossWeighting):
    """``min(snr, gamma) / snr`` for eps prediction, ``min(snr, gamma) / (snr + 1)`` for v prediction."""

    def __init__(self, gamma: float = 5.0, parameterization: str = "eps"):
        if parameterization not in ("eps", "v"):
            raise ValueError(f"Min-SNR is defined for 'eps' and 'v', got {parameterization!r}")
        self.gamma, self.parameterization = gamma, parameterization

    def weight(self, sigma):
        snr = 1.0 / (sigma ** 2 + 1e-8)
        clipped = torch.clamp(snr, max=self.gamma)
        return clipped / (snr + 1.0) if self.parameterization == "v" else clipped / snr


class P2(LossWeighting):
    def __init__(self, k: float = 1.0, gamma: float = 1.0):
        self.k, self.gamma = k, gamma

    def weight(self, sigma):
        return 1.0 / (self.k + 1.0 / (sigma ** 2 + 1e-8)) ** self.gamma


WEIGHTINGS: dict[str, type[LossWeighting]] = {"uniform": Uniform, "min_snr": MinSNR, "p2": P2}
