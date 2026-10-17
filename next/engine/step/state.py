"""What flows through the phases of one micro-step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass
class Inputs:
    x_t: Tensor
    t: Tensor
    sigma: Tensor
    target: Tensor
    t_frac: Tensor
    mask: Tensor | None = None


@dataclass
class StepState:
    step: int
    micro: int
    device: Any
    batch: dict | None = None
    inputs: Inputs | None = None
    conditioning: Any = None
    prediction: Tensor | None = None
    loss: Tensor | None = None
    per_sample: Tensor | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    phase: str = ""  # the phase running (or that failed)
