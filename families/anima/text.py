"""Qwen3-0.6B as Anima's text encoder: token ids in, last-layer hidden states (after the final norm) out.

Key layout follows the ``qwen_3_06b_base`` file (``model.embed_tokens``, ``model.layers.N.*``, ``model.norm``); the language-model
head is not part of it. Only the forward pass over a whole prompt is implemented: no cache, no generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class QwenConfig:
    vocab: int = 151936
    width: int = 1024
    layers: int = 28
    heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    mlp_width: int = 3072
    rope_theta: float = 1_000_000.0
    eps: float = 1e-6


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


class _Attention(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.cfg = cfg
        inner, kv = cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.width, inner, bias=False)
        self.k_proj = nn.Linear(cfg.width, kv, bias=False)
        self.v_proj = nn.Linear(cfg.width, kv, bias=False)
        self.o_proj = nn.Linear(inner, cfg.width, bias=False)
        self.q_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.eps)
        self.k_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        cfg = self.cfg
        b, n, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, n, cfg.heads, cfg.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, n, cfg.kv_heads, cfg.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, n, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        # Explicit repeat: the fused grouped-query path is not available on every backend.
        group = cfg.heads // cfg.kv_heads
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(b, n, -1))


class _Mlp(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.width, cfg.mlp_width, bias=False)
        self.up_proj = nn.Linear(cfg.width, cfg.mlp_width, bias=False)
        self.down_proj = nn.Linear(cfg.mlp_width, cfg.width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(cfg.width, eps=cfg.eps)
        self.self_attn = _Attention(cfg)
        self.post_attention_layernorm = nn.RMSNorm(cfg.width, eps=cfg.eps)
        self.mlp = _Mlp(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class _Stack(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab, cfg.width)
        self.layers = nn.ModuleList(_Layer(cfg) for _ in range(cfg.layers))
        self.norm = nn.RMSNorm(cfg.width, eps=cfg.eps)


class QwenEncoder(nn.Module):
    def __init__(self, cfg: QwenConfig | None = None):
        super().__init__()
        self.cfg = cfg or QwenConfig()
        self.model = _Stack(self.cfg)

    def _rotary(self, length: int, device, dtype) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=device).float() / cfg.head_dim))
        freqs = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]

    def forward(self, ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """``ids`` (B, N); ``attention_mask`` (B, N) with 1 for real tokens. Returns (B, N, 1024)."""
        stack = self.model
        x = stack.embed_tokens(ids)
        n = x.shape[1]
        allowed = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()[None, None]
        if attention_mask is not None:
            allowed = allowed & attention_mask.to(torch.bool)[:, None, None, :]
        cos, sin = self._rotary(n, x.device, x.dtype)
        for layer in stack.layers:
            x = layer(x, cos, sin, allowed)
        return stack.norm(x)
