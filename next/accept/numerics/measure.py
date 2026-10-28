"""Comparisons between tensors: how many values differ bit for bit, and how far apart two valid results are."""

from __future__ import annotations

import torch
from torch import Tensor


def differing(a: Tensor, b: Tensor) -> int:
    """Values that are not identical (a NaN against a NaN counts as identical; ``-0.0`` against ``0.0`` does not)."""
    if a.shape != b.shape:
        return a.numel()
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if a.dtype != b.dtype:
        b = b.to(a.dtype)
    bits = {2: torch.int16, 4: torch.int32, 8: torch.int64, 1: torch.int8}[a.element_size()]
    return int((a.view(bits) != b.view(bits)).sum())


def relative(a: Tensor, b: Tensor) -> float:
    """``|a - b| / |b|`` in float64, 0 for two zero tensors, inf when only the reference is zero."""
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    ref = b.norm().item()
    gap = (a - b).norm().item()
    return 0.0 if gap == 0.0 else (gap / ref if ref else float("inf"))


def finite(t: Tensor) -> bool:
    return bool(torch.isfinite(t.detach().float()).all())


def fingerprint(t: Tensor) -> str:
    """A short digest of the bytes, to show two tensors are the same without printing them."""
    import hashlib
    return hashlib.sha1(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:12]


def digest(t: Tensor) -> tuple[int, int]:
    """Two sums over the bytes of a tensor (plain, and weighted by position), computed where the tensor lives: equal digests mean the same bytes for any practical purpose."""
    flat = t.detach().contiguous().view(-1).view(torch.uint8)
    total = weighted = 0
    step = 1 << 24
    for start in range(0, flat.numel(), step):
        part = flat[start:start + step].to(torch.int64)
        position = torch.arange(start, start + part.numel(), device=part.device, dtype=torch.int64) % 1000003 + 1
        total += int(part.sum())
        weighted += int((part * position).sum())
    return total, weighted


def spread(runs: list[Tensor], first: Tensor) -> tuple[int, float]:
    """How many values of ``runs`` differ bit for bit from ``first``, and the largest relative difference of one run."""
    return sum(differing(r, first) for r in runs), max((relative(r, first) for r in runs), default=0.0)
