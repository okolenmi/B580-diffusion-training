"""Sinusoidal embeddings shared by the denoisers and by SDXL's size conditioning."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def timestep_embedding(timesteps: Tensor, dim: int, max_period: float = 10000) -> Tensor:
    """Sinusoidal features of 1-D ``timesteps`` (may be fractional), shape (N, dim), always float32.

    cos and sin halves are concatenated; an odd ``dim`` gets a trailing zero column.
    """
    if dim < 2:
        raise ValueError(f"embedding dim must be at least 2, got {dim}")
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must be 1-D, got shape {tuple(timesteps.shape)}")
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    out = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return torch.cat([out, torch.zeros_like(out[:, :1])], dim=-1) if dim % 2 else out
