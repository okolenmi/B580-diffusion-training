"""The text adapter inside the Anima file: T5 token ids attend to Qwen3 hidden states and become the 1024-wide context.

Unlike the DiT's attention, rotary angles go to queries *and* keys of both self- and cross-attention here (positions of the
T5 tokens for queries, of the Qwen3 tokens for keys). Layout follows ``net.llm_adapter.*`` in the files.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

T5_VOCAB = 32128


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


class _Rotary(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, length: int, device, dtype) -> tuple[Tensor, Tensor]:
        inv = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        freqs = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]


class _Attention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, query_dim // heads
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.o_proj = nn.Linear(query_dim, query_dim, bias=False)

    def forward(self, x: Tensor, context: Tensor, rope_q, rope_k, mask: Tensor | None) -> Tensor:
        b, n, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, n, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(context).view(b, context.shape[1], self.heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(context).view(b, context.shape[1], self.heads, self.head_dim).transpose(1, 2)
        q = q * rope_q[0][:, None] + _rotate_half(q) * rope_q[1][:, None]
        k = k * rope_k[0][:, None] + _rotate_half(k) * rope_k[1][:, None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(b, n, -1))


class _Block(nn.Module):
    def __init__(self, source_dim: int, width: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_self_attn = nn.RMSNorm(width, eps=1e-6)
        self.self_attn = _Attention(width, width, heads)
        self.norm_cross_attn = nn.RMSNorm(width, eps=1e-6)
        self.cross_attn = _Attention(width, source_dim, heads)
        self.norm_mlp = nn.RMSNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, int(width * mlp_ratio)), nn.GELU(), nn.Linear(int(width * mlp_ratio), width))

    def forward(self, x: Tensor, context: Tensor, rope_x, rope_c, target_mask, source_mask) -> Tensor:
        normed = self.norm_self_attn(x)
        x = x + self.self_attn(normed, normed, rope_x, rope_x, target_mask)
        x = x + self.cross_attn(self.norm_cross_attn(x), context, rope_x, rope_c, source_mask)
        return x + self.mlp(self.norm_mlp(x))


class LLMAdapter(nn.Module):
    def __init__(self, source_dim: int = 1024, width: int = 1024, num_layers: int = 6, heads: int = 16):
        super().__init__()
        self.embed = nn.Embedding(T5_VOCAB, width)
        self.rotary_emb = _Rotary(width // heads)
        self.blocks = nn.ModuleList(_Block(source_dim, width, heads) for _ in range(num_layers))
        self.out_proj = nn.Linear(width, width)
        self.norm = nn.RMSNorm(width, eps=1e-6)

    @staticmethod
    def _mask(mask: Tensor | None) -> Tensor | None:
        return None if mask is None else mask.to(torch.bool)[:, None, None, :]

    def forward(self, hidden: Tensor, t5_ids: Tensor, target_mask: Tensor | None = None,
                source_mask: Tensor | None = None) -> Tensor:
        """``hidden`` (B, S, 1024) Qwen3 states, ``t5_ids`` (B, T) token ids; masks are 1 where a token is real."""
        x = self.embed(t5_ids).to(hidden.dtype)
        rope_x = self.rotary_emb(x.shape[1], x.device, x.dtype)
        rope_c = self.rotary_emb(hidden.shape[1], x.device, x.dtype)
        target_mask, source_mask = self._mask(target_mask), self._mask(source_mask)
        for block in self.blocks:
            x = block(x, hidden, rope_x, rope_c, target_mask, source_mask)
        return self.norm(self.out_proj(x))
