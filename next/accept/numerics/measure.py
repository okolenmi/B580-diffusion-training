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
