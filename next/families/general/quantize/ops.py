"""The int8 linear as an autograd op, and the small object layers use to call it."""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

from .backends import Int8Backend


class Int8LinearFn(torch.autograd.Function):
    """``x @ W.T + bias`` against an int8 weight. The weight is frozen, so backward only returns ``dL/dx``.

    ``dL/dx = dL/dy @ W`` needs the dense weight; it is rebuilt from int8 for this one layer and dropped again, so the
    memory the quantization saved stays saved.
    """

    @staticmethod
    def forward(ctx, x: Tensor, w8: Tensor, scale: Tensor, bias: Tensor | None, backend: Int8Backend) -> Tensor:
        ctx.backend, ctx.dtype = backend, x.dtype
        ctx.save_for_backward(w8, scale)
        return backend.linear(x, w8, scale, bias)

    @staticmethod
    def backward(ctx, grad: Tensor):
        w8, scale = ctx.saved_tensors
        dense = ctx.backend.dequant(w8, scale)
        return (grad.to(dense.dtype) @ dense).to(ctx.dtype), None, None, None, None


class Int8Weight:
    """What both layer kinds share: the int8 weight, its scale and the backend that understands them."""

    def __init__(self, backend: Int8Backend):
        self.backend = backend
        self._warned = False

    def apply(self, x: Tensor, w8: Tensor, scale: Tensor, bias: Tensor | None) -> Tensor:
        problem = self.backend.call_problem(x, w8)
        if problem is None:
            return Int8LinearFn.apply(x, w8, scale, bias, self.backend)
        if not self._warned:
            self._warned = True
            warnings.warn(f"int8 linear falls back to the dense weight: {problem}", stacklevel=3)
        return F.linear(x, self.backend.dequant(w8, scale).to(x.dtype), None if bias is None else bias.to(x.dtype))
