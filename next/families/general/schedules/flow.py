"""Rectified flow (Cosmos / Anima style): a straight line between data and noise.

``x_s = (1 - s) * x0 + s * noise`` for ``s`` in [0, 1]; the network predicts the velocity
``noise - x0``. Defaults for shift and sampling are the commonly published ones and are
unverified for Anima until measured (rework/audit/anima-model-facts.md).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

SAMPLERS = ("uniform", "sigmoid", "logit_normal", "cosmap")
WEIGHTINGS = ("uniform", "sigma_sqrt", "cosmap")


def shift_sigma(s: Tensor, shift: float) -> Tensor:
    """Push mass toward noise for large images: s' = shift*s / (1 + (shift-1)*s)."""
    if shift == 1.0:
        return s
    return shift * s / (1.0 + (shift - 1.0) * s)


def sample_sigma(n: int, mode: str = "sigmoid", *, shift: float = 1.0, scale: float = 1.0,
                 generator: torch.Generator | None = None, device="cpu") -> Tensor:
    """Draw ``n`` flow positions in (0, 1)."""
    if mode not in SAMPLERS:
        raise ValueError(f"unknown flow sampler {mode!r}; choose from {SAMPLERS}")
    if mode == "uniform":
        s = torch.rand(n, generator=generator, device=device)
    elif mode in ("sigmoid", "logit_normal"):
        s = torch.sigmoid(torch.randn(n, generator=generator, device=device) * scale)
    else:
        u = torch.rand(n, generator=generator, device=device)
        s = 1.0 - 1.0 / (torch.tan(math.pi * u / 2.0) + 1.0)
    return shift_sigma(s, shift).clamp(1e-5, 1.0 - 1e-5)


def noisy(x0: Tensor, noise: Tensor, s: Tensor) -> tuple[Tensor, Tensor]:
    """(x_s, velocity target) for flow positions ``s`` of shape (B,)."""
    s = s.to(x0.dtype).view(-1, *([1] * (x0.ndim - 1)))
    return (1.0 - s) * x0 + s * noise, noise - x0


def to_x0(velocity: Tensor, x_s: Tensor, s: Tensor) -> Tensor:
    s = s.to(x_s.dtype).view(-1, *([1] * (x_s.ndim - 1)))
    return x_s - s * velocity


def loss_weight(s: Tensor, name: str = "uniform") -> Tensor:
    if name == "uniform":
        return torch.ones_like(s)
    if name == "sigma_sqrt":
        return 1.0 / (s ** 2)
    if name == "cosmap":
        return 2.0 / (math.pi * (1.0 - 2.0 * s + 2.0 * s ** 2))
    raise ValueError(f"unknown flow weighting {name!r}; choose from {WEIGHTINGS}")
