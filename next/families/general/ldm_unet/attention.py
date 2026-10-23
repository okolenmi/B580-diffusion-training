"""Transformer blocks of the latent-diffusion UNet. Key names follow the published checkpoints."""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import nn


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


_replacement = None  # an alternative to the stock kernel while a ``replace_sdpa`` block is open


@contextlib.contextmanager
def replace_sdpa(function):
    """Within the block every attention call goes through ``function(q, k, v)`` (tensors of shape batch, heads, length, head_dim)."""
    global _replacement
    previous, _replacement = _replacement, function
    try:
        yield
    finally:
        _replacement = previous


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v) if _replacement is None else _replacement(q, k, v)


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
    """Multi-head attention over (B, N, heads * head_dim) tensors; heads are split out of the channel axis."""
    b, n, width = q.shape

    def split(t: torch.Tensor) -> torch.Tensor:
        return t.view(b, -1, heads, width // heads).transpose(1, 2)

    out = sdpa(split(q), split(k), split(v))
    return out.transpose(1, 2).reshape(b, n, width)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4, glu: bool = False,
                 dropout: float = 0.0) -> None:
        super().__init__()
        inner = int(dim * mult)
        project_in = GEGLU(dim, inner) if glu else nn.Sequential(nn.Linear(dim, inner), nn.GELU())
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner, dim if dim_out is None else dim_out))

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, heads: int, dim_head: int, dropout: float = 0.0,
                 context_dim: int | None = None) -> None:
        super().__init__()
        inner = heads * dim_head
        self.heads, self.query_dim = heads, query_dim
        self.context_dim = query_dim if context_dim is None else context_dim
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(self.context_dim, inner, bias=False)
        self.to_v = nn.Linear(self.context_dim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None):
        if context is None:
            if self.context_dim != self.query_dim:
                raise ValueError(f"no context given, but this attends over {self.context_dim} channels, not {self.query_dim}")
            context = x
        return self.to_out(attend(self.to_q(x), self.to_k(context), self.to_v(context), self.heads))


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, d_head: int, dropout: float = 0.0, context_dim: int | None = None,
                 gated_ff: bool = True) -> None:
        super().__init__()
        self.attn1 = CrossAttention(dim, n_heads, d_head, dropout)
        self.attn2 = CrossAttention(dim, n_heads, d_head, dropout, context_dim)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.checkpointer = None

    def forward(self, x, context=None):
        if self.checkpointer is None:
            return self._forward(x, context)
        inputs = (x,) if context is None else (x, context)
        return self.checkpointer(self._forward, inputs, self.parameters())

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        return self.ff(self.norm3(x)) + x


class SpatialTransformer(nn.Module):
    """Image features as a sequence: norm, project in, ``depth`` transformer blocks, project out, residual.

    ``use_linear`` swaps the 1x1 convolutions for linear layers (SDXL). ``context`` may be one tensor for every
    block or a list with one tensor per block.
    """

    def __init__(self, in_channels: int, n_heads: int, d_head: int, depth: int, dropout: float = 0.0,
                 context_dim: int | None = None, use_linear: bool = False) -> None:
        super().__init__()
        self.use_linear = use_linear
        inner = n_heads * d_head
        self.norm = group_norm(in_channels)
        self.proj_in = nn.Linear(in_channels, inner) if use_linear else nn.Conv2d(in_channels, inner, kernel_size=1)
        self.transformer_blocks = nn.ModuleList(
            BasicTransformerBlock(inner, n_heads, d_head, dropout, context_dim) for _ in range(depth))
        self.proj_out = nn.Linear(inner, in_channels) if use_linear else nn.Conv2d(inner, in_channels, kernel_size=1)

    def forward(self, x, context=None):
        if isinstance(context, list):
            if len(context) != len(self.transformer_blocks):
                raise ValueError(f"expected {len(self.transformer_blocks)} context tensors, got {len(context)}")
            pairs = list(zip(self.transformer_blocks, context))
        else:
            pairs = [(block, context) for block in self.transformer_blocks]
        b, _, h, w = x.shape
        residual = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.movedim(1, 3).flatten(1, 2).contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for block, ctx in pairs:
            x = block(x, context=ctx)
        if self.use_linear:
            x = self.proj_out(x)
        x = x.reshape(b, h, w, -1).movedim(3, 1).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + residual
