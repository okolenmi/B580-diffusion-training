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
    scale: float = 1.0  # share of the optimizer step's gradient this pass carries (a chunk of a split batch)


def split_batch(batch: dict, sizes: list[int]) -> list[dict]:
    """The batch as consecutive chunks of ``sizes`` samples; whatever is not per-sample is shared by every chunk."""
    count = sum(sizes)
    if len(sizes) == 1:
        return [batch]
    chunks: list[dict] = [dict(batch) for _ in sizes]
    start = 0
    for chunk, size in zip(chunks, sizes):
        for key, value in batch.items():
            if isinstance(value, Tensor) and value.ndim >= 1 and value.shape[0] == count or isinstance(value, list) and len(value) == count:
                chunk[key] = value[start:start + size]
        start += size
    return chunks
