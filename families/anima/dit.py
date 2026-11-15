"""The Anima transformer: a Cosmos-Predict2 style DiT for single-frame images, with its text adapter attached.

Key names and shapes follow the files (checked against real headers in ``check_anima``): ``net.blocks.N.{self_attn,
cross_attn,mlp,adaln_modulation_*}``, ``x_embedder.proj.1``, ``t_embedder.1``, ``final_layer`` and ``llm_adapter``.
A latent is ``(B, 16, 1, H, W)``; a padding-mask channel is appended and 2x2 patches become tokens.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapter import LLMAdapter

Checkpointer = Callable[[Callable[..., Any], tuple, Iterable[nn.Parameter]], Any]


@dataclass(frozen=True)
class DiTConfig:
    """Everything that is not a weight. ``rope_extrapolation`` stretches the rotary frequencies (height, width, time)."""

    in_channels: int = 16
    out_channels: int = 16
    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    context_channels: int = 1024
    patch: int = 2
    adaln_lora_dim: int = 256
    adapter_blocks: int = 6
    rope_extrapolation: tuple[float, float, float] = (4.0, 4.0, 1.0)

    @classmethod
    def from_shapes(cls, blocks: int, width: int, in_embed: int, adapter_blocks: int = 6) -> DiTConfig:
        """From what a header says: block count, token width, and the embedder's input size (``(channels + 1) * patch^2``)."""
        return cls(in_channels=in_embed // 4 - 1, model_channels=width, num_blocks=blocks, num_heads=16 if width == 2048 else 40,
                   adapter_blocks=adapter_blocks)


class RotaryEmbedding3D(nn.Module):
    """Rotary angles for a (time, height, width) token grid; the head dimension is split between the three axes."""

    def __init__(self, head_dim: int, extrapolation: tuple[float, float, float]):
        super().__init__()
        dim_h = head_dim // 6 * 2
        dim_t = head_dim - 2 * dim_h
        self.dim_h, self.dim_t = dim_h, dim_t
        self.theta = tuple(10000.0 * r ** (d / (d - 2)) for r, d in zip(extrapolation, (dim_h, dim_h, dim_t)))

    def angles(self, t: int, h: int, w: int, device) -> Tensor:
        """(t*h*w, head_dim/2) angles, ordered time, height, width along the last dimension."""
        h_theta, w_theta, t_theta = self.theta
        # computed here, not kept as buffers: a network built on the meta device has no way to fill them in
        range_hw = torch.arange(0, self.dim_h, 2, device=device).float()[: self.dim_h // 2] / self.dim_h
        range_t = torch.arange(0, self.dim_t, 2, device=device).float()[: self.dim_t // 2] / self.dim_t
        f_h = 1.0 / (h_theta ** range_hw)
        f_w = 1.0 / (w_theta ** range_hw)
        f_t = 1.0 / (t_theta ** range_t)
        pos = lambda n: torch.arange(n, dtype=torch.float32, device=device)  # noqa: E731
        a_t = torch.outer(pos(t), f_t)[:, None, None, :].expand(t, h, w, -1)
        a_h = torch.outer(pos(h), f_h)[None, :, None, :].expand(t, h, w, -1)
        a_w = torch.outer(pos(w), f_w)[None, None, :, :].expand(t, h, w, -1)
        return torch.cat([a_t, a_h, a_w], dim=-1).reshape(t * h * w, -1)


def rotate_split_half(x: Tensor, angles: Tensor) -> Tensor:
    """Rotate the pairs ``(x[i], x[i + D/2])`` by ``angles`` (L, D/2); ``x`` is (B, L, heads, D)."""
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class Attention(nn.Module):
    """Multi-head attention with per-head RMS norm on queries and keys; rotary angles go to self-attention only."""

    def __init__(self, query_dim: int, context_dim: int | None, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, query_dim // heads
        context_dim = context_dim or query_dim
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.output_proj = nn.Linear(query_dim, query_dim, bias=False)

    def forward(self, x: Tensor, context: Tensor | None = None, angles: Tensor | None = None) -> Tensor:
        context = x if context is None else context
        b, n, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, n, self.heads, self.head_dim))
        k = self.k_norm(self.k_proj(context).view(b, context.shape[1], self.heads, self.head_dim))
        v = self.v_proj(context).view(b, context.shape[1], self.heads, self.head_dim)
        if angles is not None:
            q, k = rotate_split_half(q, angles), rotate_split_half(k, angles)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.output_proj(out.transpose(1, 2).reshape(b, n, -1))


class Modulation(nn.Sequential):
    """SiLU, a low-rank projection and the expansion to (shift, scale, gate) or (shift, scale)."""

    def __init__(self, width: int, rank: int, chunks: int):
        super().__init__(nn.SiLU(), nn.Linear(width, rank, bias=False), nn.Linear(rank, chunks * width, bias=False))


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.layer1 = nn.Linear(width, hidden, bias=False)
        self.layer2 = nn.Linear(hidden, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer2(F.gelu(self.layer1(x)))


class Block(nn.Module):
    """Self-attention, cross-attention and feed-forward, each behind a layer norm modulated by the timestep embedding."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        d = cfg.model_channels
        self.checkpointer: Checkpointer | None = None
        norm = lambda: nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)  # noqa: E731
        self.layer_norm_self_attn, self.layer_norm_cross_attn, self.layer_norm_mlp = norm(), norm(), norm()
        self.self_attn = Attention(d, None, cfg.num_heads)
        self.cross_attn = Attention(d, cfg.context_channels, cfg.num_heads)
        self.mlp = FeedForward(d, int(d * cfg.mlp_ratio))
        self.adaln_modulation_self_attn = Modulation(d, cfg.adaln_lora_dim, 3)
        self.adaln_modulation_cross_attn = Modulation(d, cfg.adaln_lora_dim, 3)
        self.adaln_modulation_mlp = Modulation(d, cfg.adaln_lora_dim, 3)

    def forward(self, x: Tensor, emb: Tensor, lora: Tensor, context: Tensor, angles: Tensor) -> Tensor:
        if self.checkpointer is not None and torch.is_grad_enabled():
            return self.checkpointer(self._forward, (x, emb, lora, context, angles), self.parameters())
        return self._forward(x, emb, lora, context, angles)

    def _forward(self, x: Tensor, emb: Tensor, lora: Tensor, context: Tensor, angles: Tensor) -> Tensor:
        """``x`` is (B, L, D) tokens; ``emb`` and ``lora`` are (B, 1, D) and (B, 1, 3D)."""
        compute = emb.dtype
        shift, scale, gate = (self.adaln_modulation_self_attn(emb) + lora).chunk(3, dim=-1)
        h = (self.layer_norm_self_attn(x) * (1 + scale) + shift).to(compute)
        x = torch.addcmul(x, gate.to(x.dtype), self.self_attn(h, angles=angles).to(x.dtype))
        shift, scale, gate = (self.adaln_modulation_cross_attn(emb) + lora).chunk(3, dim=-1)
        h = (self.layer_norm_cross_attn(x) * (1 + scale) + shift).to(compute)
        x = torch.addcmul(x, gate.to(x.dtype), self.cross_attn(h, context).to(x.dtype))
        shift, scale, gate = (self.adaln_modulation_mlp(emb) + lora).chunk(3, dim=-1)
        h = (self.layer_norm_mlp(x) * (1 + scale) + shift).to(compute)
        return torch.addcmul(x, gate.to(x.dtype), self.mlp(h).to(x.dtype))


class TimestepEmbedding(nn.Module):
    """Sinusoidal features of the flow position, then the low-rank expansion the blocks add to their own modulation."""

    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.linear_1 = nn.Linear(width, width, bias=False)
        self.linear_2 = nn.Linear(width, 3 * width, bias=False)

    def features(self, t: Tensor) -> Tensor:
        half = self.width // 2
        exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        phase = t.float().flatten()[:, None] * torch.exp(exponent)[None, :]
        return torch.cat([phase.cos(), phase.sin()], dim=-1)

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        sample = self.features(t)
        return sample, self.linear_2(F.silu(self.linear_1(sample.to(self.linear_1.weight.dtype))))


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, width: int, patch: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Sequential(nn.Identity(), nn.Linear(in_channels * patch * patch, width, bias=False))  # key `proj.1`

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, H, W) to (B, H/p * W/p, D), pixels of one patch ordered channel-major (c, row, column)."""
        b, c, h, w = x.shape
        p = self.patch
        x = x.view(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // p) * (w // p), c * p * p)
        return self.proj[1](x)


class FinalLayer(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        d = cfg.model_channels
        self.layer_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(d, cfg.patch * cfg.patch * cfg.out_channels, bias=False)
        self.adaln_modulation = Modulation(d, cfg.adaln_lora_dim, 2)
        self.width = d

    def forward(self, x: Tensor, emb: Tensor, lora: Tensor) -> Tensor:
        shift, scale = (self.adaln_modulation(emb) + lora[..., : 2 * self.width]).chunk(2, dim=-1)
        return self.linear(self.layer_norm(x) * (1 + scale) + shift)


class AnimaDiT(nn.Module):
    """Latents ``(B, C, H, W)`` (one frame), flow positions in [0, 1] and a ``(B, N, 1024)`` context in, velocity out."""

    def __init__(self, cfg: DiTConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or DiTConfig()
        d = cfg.model_channels
        self.pos_embedder = RotaryEmbedding3D(d // cfg.num_heads, cfg.rope_extrapolation)
        self.t_embedder = nn.Sequential(nn.Identity(), TimestepEmbedding(d))  # key `t_embedder.1`
        self.x_embedder = PatchEmbed(cfg.in_channels + 1, d, cfg.patch)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.num_blocks))
        self.final_layer = FinalLayer(cfg)
        self.t_embedding_norm = nn.RMSNorm(d, eps=1e-6)
        self.llm_adapter = LLMAdapter(num_layers=cfg.adapter_blocks)

    def forward(self, x: Tensor, t: Tensor, context: Tensor) -> Tensor:
        cfg, (b, _, h, w) = self.cfg, x.shape
        p = cfg.patch
        if h % p or w % p:
            raise ValueError(f"latent size {h}x{w} is not a multiple of the patch size {p}")
        x = torch.cat([x, x.new_zeros(b, 1, h, w)], dim=1)  # the padding-mask channel, nothing padded
        tokens = self.x_embedder(x)
        angles = self.pos_embedder.angles(1, h // p, w // p, x.device)
        features, lora = self.t_embedder[1](t.view(b, 1))
        emb = self.t_embedding_norm(features.to(tokens.dtype))[:, None, :]
        lora = lora[:, None, :]
        if tokens.dtype == torch.float16:  # the residual stream is large; keep it in float32
            tokens = tokens.float()
        for block in self.blocks:
            tokens = block(tokens, emb, lora, context, angles)
        out = self.final_layer(tokens.to(context.dtype), emb, lora)
        out = out.view(b, h // p, w // p, p, p, cfg.out_channels).permute(0, 5, 1, 3, 2, 4)
        return out.reshape(b, cfg.out_channels, h, w)

    def encode_text(self, hidden: Tensor, t5_ids: Tensor, hidden_mask: Tensor | None = None,
                    ids_mask: Tensor | None = None) -> Tensor:
        """The adapter's output for Qwen3 hidden states and T5 token ids, padded to the 512 positions the DiT trains on.

        Masks (1 where a token is real) let prompts of different lengths share a batch; positions past a prompt's own T5
        tokens come out zero, as when each prompt runs alone and is padded afterwards.
        """
        out = self.llm_adapter(hidden, t5_ids, ids_mask, hidden_mask)
        if ids_mask is not None:
            out = out * ids_mask.to(out.dtype)[..., None]
        if out.shape[1] < 512:
            out = F.pad(out, (0, 0, 0, 512 - out.shape[1]))
        return out

    def checkpoint_blocks(self) -> list[tuple[str, nn.Module]]:
        return [(f"blocks.{i}", block) for i, block in enumerate(self.blocks)]

    def set_checkpointed(self, checkpointer: Checkpointer | None, flags: Iterable[bool]) -> None:
        flags = list(flags)
        if len(flags) != len(self.blocks):
            raise ValueError(f"{len(self.blocks)} blocks need {len(self.blocks)} flags, got {len(flags)}")
        for block, on in zip(self.blocks, flags):
            block.checkpointer = checkpointer if on else None

    def set_checkpointer(self, checkpointer: Checkpointer | None, fraction: float = 1.0) -> None:
        """A share of the blocks, spread evenly, runs through ``checkpointer`` (1 every block, 0 none)."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        n = len(self.blocks)
        self.set_checkpointed(checkpointer, [int((i + 1) * fraction) > int(i * fraction) for i in range(n)])
