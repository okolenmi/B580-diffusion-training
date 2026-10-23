"""Frozen weights stored in int8: the autograd op, the module for plain linears, and the pass that converts a network."""

from __future__ import annotations

from torch import Tensor, nn

from ..adapters.layers import AdaptedLinear
from .backends import Int8Backend, make_backend, shape_problem
from .frozen import FrozenQuantizer
from .ops import Int8Weight


class Int8Linear(nn.Module):
    """A frozen ``nn.Linear`` that keeps its weight in int8."""

    def __init__(self, original: nn.Linear, backend: Int8Backend, grad: str = "dense"):
        super().__init__()
        self.in_features, self.out_features = original.in_features, original.out_features
        self.quantized = Int8Weight(backend, grad)
        w8, scale = backend.prepare(original.weight.detach())
        self.register_buffer("weight_int8", w8)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", original.bias.detach() if original.bias is not None else None)

    def forward(self, x: Tensor) -> Tensor:
        return self.quantized.apply(x, self.weight_int8, self.weight_scale, self.bias)


class Int8Quantizer(FrozenQuantizer):
    label = "int8"

    def __init__(self, backend: Int8Backend, min_params: int = 1 << 20, grad: str = "dense"):
        super().__init__(backend, min_params)
        self.grad = grad

    def shape_problem(self, weight_shape) -> str | None:
        return shape_problem(*weight_shape)

    def weight_error(self, weight) -> float:
        w8, scale = self.backend.prepare(weight)
        return ((self.backend.dequant(w8, scale).float() - weight.float()).norm() / weight.float().norm()).item()

    def convert_adapted(self, layer: AdaptedLinear) -> int:
        layer.use_int8(self.backend, self.grad)
        return layer.base_weight.numel() + layer.base_scale.numel() * 4

    def replacement(self, layer: nn.Linear) -> tuple[nn.Module, int]:
        module = Int8Linear(layer, self.backend, self.grad)
        return module, module.weight_int8.numel() + module.weight_scale.numel() * 4


def quantize_frozen(module: nn.Module, *, backend: str = "auto", device="cpu", min_params: int = 1 << 20, grad: str = "dense") -> QuantizeReport:
    """Convert the eligible linear layers of ``module`` to int8 frozen weights. ``device`` is where it computes (decides ``auto``)."""
    return Int8Quantizer(make_backend(backend, device), min_params, grad).run(module)
