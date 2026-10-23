"""The NF4 linear as an autograd op, and the small object layers use to call it."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .nf4_backends import Nf4Backend


class Nf4LinearFn(torch.autograd.Function):
    """``x @ W.T + bias`` against an NF4 weight, rebuilt (dequantized) for the call and dropped again; backward rebuilds it once more.

    Only the quantized tensors are kept for backward, so the memory the format saves stays saved. The weight is frozen: only ``dL/dx`` returns.
    """

    @staticmethod
    def forward(ctx, x: Tensor, packed: Tensor, absmax: Tensor, scale: Tensor | None, offset: Tensor | None, bias: Tensor | None,
                backend: Nf4Backend, shape: tuple[int, ...]) -> Tensor:
        ctx.backend, ctx.shape, ctx.dtype = backend, shape, x.dtype
        ctx.save_for_backward(packed, absmax, scale, offset)
        weight = backend.dequant(packed, absmax, scale, offset, shape, x.dtype)
        return F.linear(x, weight, None if bias is None else bias.to(x.dtype))

    @staticmethod
    def backward(ctx, grad: Tensor):
        packed, absmax, scale, offset = ctx.saved_tensors
        weight = ctx.backend.dequant(packed, absmax, scale, offset, ctx.shape, grad.dtype)
        return (grad @ weight).to(ctx.dtype), None, None, None, None, None, None, None


class Nf4Weight:
    """What both layer kinds share: how the packed weight is read back, and the backend that does it."""

    def __init__(self, backend: Nf4Backend, shape: tuple[int, ...], double_quant: bool):
        self.backend, self.shape, self.double_quant = backend, tuple(shape), double_quant

    def quantize(self, weight: Tensor) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        return self.backend.quantize(weight, self.double_quant)

    def apply(self, x: Tensor, packed: Tensor, absmax: Tensor, scale: Tensor | None, offset: Tensor | None, bias: Tensor | None) -> Tensor:
        return Nf4LinearFn.apply(x, packed, absmax, scale, offset, bias, self.backend, self.shape)

    def dense(self, packed: Tensor, absmax: Tensor, scale: Tensor | None, offset: Tensor | None, dtype: torch.dtype = torch.bfloat16) -> Tensor:
        return self.backend.dequant(packed, absmax, scale, offset, self.shape, dtype)

    @staticmethod
    def stored_bytes(*tensors: Tensor | None) -> int:
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)
