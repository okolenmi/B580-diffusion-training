"""Ways to compute attention that can run inside a captured device graph, chosen by name.

The stock kernel on XPU (flash) cannot be captured, so a captured step runs the math backend, which converts bf16 inputs to fp32 first.
The modes here trade that for bf16 matrix products. Both round the attention scores to bf16 before the softmax (the stock math backend does not),
which is what most bf16 attention outside flash does; the softmax itself runs in fp32.
"""

from __future__ import annotations

import contextlib
import functools
from contextlib import AbstractContextManager
from collections.abc import Callable

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .attention import replace_sdpa


class ChunkedAttention(torch.autograd.Function):
    """Softmax attention in bf16 matrix products, over blocks of query rows so the score matrix never exists whole.

    Only q, k, v and the output are kept for backward; the probabilities are recomputed per block there (so memory does not grow with the
    score matrix). The key and value gradients add up across blocks in fp32. Nothing here synchronises or allocates outside the allocator,
    so it replays from a graph.
    """

    CHUNK_ELEMENTS = 1 << 25  # scores per block (batch * heads * rows * keys)

    @classmethod
    def rows_per_block(cls, q: torch.Tensor, k: torch.Tensor) -> int:
        batch_heads = q.shape[0] * q.shape[1]
        return max(1, min(q.shape[-2], cls.CHUNK_ELEMENTS // max(batch_heads * k.shape[-2], 1)))

    @staticmethod
    def _probabilities(q_block: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
        scores = torch.matmul(q_block * scale, k.transpose(-1, -2))
        return torch.softmax(scores, dim=-1, dtype=torch.float32)

    @staticmethod
    def forward(ctx, q, k, v, rows: int):
        scale = q.shape[-1] ** -0.5
        parts = [torch.matmul(ChunkedAttention._probabilities(q[..., lo:lo + rows, :], k, scale).to(q.dtype), v) for lo in range(0, q.shape[-2], rows)]
        out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-2)
        ctx.save_for_backward(q, k, v, out)
        ctx.rows, ctx.scale = rows, scale
        return out

    @staticmethod
    def backward(ctx, dy):
        q, k, v, out = ctx.saved_tensors
        rows, scale = ctx.rows, ctx.scale
        want_q, want_k, want_v, _ = ctx.needs_input_grad
        dq_parts: list[torch.Tensor] = []
        dk = dv = None
        for lo in range(0, q.shape[-2], rows):
            q_block, dy_block = q[..., lo:lo + rows, :], dy[..., lo:lo + rows, :]
            p = ChunkedAttention._probabilities(q_block, k, scale)
            dp = torch.matmul(dy_block, v.transpose(-1, -2)).float()
            delta = (dy_block.float() * out[..., lo:lo + rows, :].float()).sum(-1, keepdim=True)
            ds = (p * (dp - delta)).to(q.dtype)
            if want_q:
                dq_parts.append(torch.matmul(ds, k) * scale)
            if want_k:
                step = torch.matmul(ds.transpose(-1, -2), q_block).float()
                dk = step if dk is None else dk.add_(step)
            if want_v:
                step = torch.matmul(p.to(q.dtype).transpose(-1, -2), dy_block).float()
                dv = step if dv is None else dv.add_(step)
        dq = (dq_parts[0] if len(dq_parts) == 1 else torch.cat(dq_parts, dim=-2)) if want_q else None
        return dq, None if dk is None else (dk * scale).to(k.dtype), None if dv is None else dv.to(v.dtype), None


def chunked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return ChunkedAttention.apply(q, k, v, ChunkedAttention.rows_per_block(q, k))


@contextlib.contextmanager
def _math_with_bf16_products():
    """The math backend, told it may keep bf16/fp16 products instead of converting its inputs to fp32."""
    backends = torch.backends.cuda
    setter, getter = getattr(backends, "allow_fp16_bf16_reduction_math_sdp", None), getattr(backends, "fp16_bf16_reduction_math_sdp_allowed", None)
    if setter is None or getter is None:
        with sdpa_kernel(SDPBackend.MATH):
            yield
        return
    previous = getter()
    setter(True)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        setter(previous)


@functools.cache
def _fused_module():
    """``xpu_kernels``' Triton attention, or None when the installed package predates it."""
    try:
        from xpu_kernels.triton import attention
    except ImportError:
        return None
    return attention


def fused_attention_status(device) -> str:
    """Empty when the ``triton`` mode runs the fused kernel on ``device``, otherwise why it runs the chunked mode instead."""
    kernels = _fused_module()
    if kernels is None:
        return "the installed xpu-kernels has no Triton attention (it needs a release newer than 0.3)"
    probe = torch.zeros((1, 1, 16, 64), dtype=torch.bfloat16, device=device)
    if not kernels.usable(probe, probe, probe):
        return "the fused attention is not usable here (no matrix units exposed to Triton, or a verification gate closed)"
    return ""


def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The fused, capturable kernel where it can run these tensors; the chunked bf16 mode where it cannot (the choice is per shape, so a captured graph is consistent)."""
    kernels = _fused_module()
    if kernels is not None and kernels.usable(q, k, v):
        return kernels.attention(q, k, v)
    return chunked_attention(q, k, v)


MODES: dict[str, Callable[[], AbstractContextManager]] = {
    "math": lambda: sdpa_kernel(SDPBackend.MATH),
    "math_bf16": _math_with_bf16_products,
    "chunked_bf16": lambda: replace_sdpa(chunked_attention),
    "triton": lambda: replace_sdpa(fused_attention),
}
