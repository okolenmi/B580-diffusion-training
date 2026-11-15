"""The int8 linear as an autograd op, and the small object layers use to call it."""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

from .backends import Int8Backend


class Int8LinearFn(torch.autograd.Function):
    """``x @ W.T + bias`` against an int8 weight. The weight is frozen, so backward only returns ``dL/dx``.

    ``dL/dx = dL/dy @ W``. ``dense`` rebuilds this one layer's bf16 weight and drops it again, so the memory the
    quantization saved stays saved, and gives the exact gradient of the int8 network. ``int8`` keeps the weight in int8
    and quantizes ``dL/dy`` instead: no dense weight at all, a little more gradient error.
    """

    @staticmethod
    def forward(ctx, x: Tensor, w8: Tensor, scale: Tensor, bias: Tensor | None, backend: Int8Backend, grad: str) -> Tensor:
        ctx.backend, ctx.dtype, ctx.grad = backend, x.dtype, grad
        ctx.save_for_backward(w8, scale)
        return backend.linear(x, w8, scale, bias)

    @staticmethod
    def backward(ctx, grad: Tensor):
        w8, scale = ctx.saved_tensors
        if ctx.grad == "int8":
            dx = ctx.backend.dx_int8(grad, w8, scale)
        else:
            dense = ctx.backend.dequant(w8, scale)
            dx = grad.to(dense.dtype) @ dense
        return dx.to(ctx.dtype), None, None, None, None, None


class Int8Weight:
    """What both layer kinds share: the int8 weight, its scale and the backend that understands them."""

    GRADS = ("dense", "int8")

    def __init__(self, backend: Int8Backend, grad: str = "dense"):
        if grad not in self.GRADS:
            raise ValueError(f"unknown int8 gradient path {grad!r}; choose from {self.GRADS}")
        self.backend, self.grad = backend, grad
        self._warned = False

    def apply(self, x: Tensor, w8: Tensor, scale: Tensor, bias: Tensor | None) -> Tensor:
        problem = self.backend.call_problem(x, w8)
        if problem is None:
            return Int8LinearFn.apply(x, w8, scale, bias, self.backend, self.grad)
        if not self._warned:
            self._warned = True
            warnings.warn(f"int8 linear falls back to the dense weight: {problem}", stacklevel=3)
        return F.linear(x, self.backend.dequant(w8, scale).to(x.dtype), None if bias is None else bias.to(x.dtype))
