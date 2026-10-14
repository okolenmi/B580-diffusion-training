"""Drawing integer timesteps for discrete schedules, one at a time or a batch at a time."""

from __future__ import annotations

import math
import random

import torch
from torch import Tensor

MODES = ("uniform", "low", "mid", "high", "logit")
_BETA = {"low": (1.0, 3.0), "mid": (2.0, 2.0), "high": (3.0, 1.0)}


def sample_timestep(rng: random.Random, mode: str, low: int, high: int) -> int:
    """One timestep in [low, high]. low/mid/high are Beta(1,3)/(2,2)/(3,1); logit is logit-normal."""
    if mode not in MODES:
        raise ValueError(f"unknown timestep mode {mode!r}; choose from {MODES}")
    if mode == "uniform":
        return rng.randint(low, high)
    if mode == "logit":
        p = 1.0 / (1.0 + math.exp(-rng.gauss(0.0, 1.0)))
    else:
        a, b = _BETA[mode]
        x, y = rng.gammavariate(a, 1.0), rng.gammavariate(b, 1.0)
        p = x / (x + y)
    return max(low, min(high, int(round(low + p * (high - low)))))


def sample_timesteps(n: int, mode: str, low: int, high: int, generator: torch.Generator | None = None) -> Tensor:
    """A batch of timesteps, the same distributions drawn with torch."""
    if mode not in MODES:
        raise ValueError(f"unknown timestep mode {mode!r}; choose from {MODES}")
    if mode == "uniform":
        return torch.randint(low, high + 1, (n,), generator=generator)
    if mode == "logit":
        p = torch.sigmoid(torch.randn(n, generator=generator))
    else:
        a, b = _BETA[mode]
        p = torch.distributions.Beta(torch.tensor(a), torch.tensor(b)).sample((n,))
    return (low + p * (high - low)).round().long().clamp(low, high)
