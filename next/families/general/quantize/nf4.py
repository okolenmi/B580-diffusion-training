"""Frozen weights stored as NF4: the module for plain linears and the pass that converts a network."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..adapters.layers import AdaptedLinear
from .frozen import FrozenQuantizer
from .nf4_backends import Nf4Backend, make_nf4_backend, shape_problem
from .nf4_ops import Nf4Weight


class Nf4Linear(nn.Module):
    """A frozen ``nn.Linear`` that keeps its weight as NF4."""

    def __init__(self, original: nn.Linear, backend: Nf4Backend, double_quant: bool = False):
        super().__init__()
        self.in_features, self.out_features = original.in_features, original.out_features
        self.quantized = Nf4Weight(backend, original.weight.shape, double_quant)
        packed, absmax, scale, offset = self.quantized.quantize(original.weight)
        self.register_buffer("weight_nf4", packed)
        self.register_buffer("weight_absmax", absmax)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("weight_offset", offset)
        self.register_buffer("bias", original.bias.detach() if original.bias is not None else None)

    @property
    def stored_bytes(self) -> int:
        return Nf4Weight.stored_bytes(self.weight_nf4, self.weight_absmax, self.weight_scale, self.weight_offset)

    def forward(self, x: Tensor) -> Tensor:
        return self.quantized.apply(x, self.weight_nf4, self.weight_absmax, self.weight_scale, self.weight_offset, self.bias)


class Nf4Quantizer(FrozenQuantizer):
    label = "nf4"

    def __init__(self, backend: Nf4Backend, min_params: int = 1 << 20, double_quant: bool = False):
        super().__init__(backend, min_params)
        self.double_quant = double_quant

    def shape_problem(self, weight_shape) -> str | None:
        return shape_problem(weight_shape)

    def weight_error(self, weight) -> float:
        tensors = self.backend.quantize(weight, self.double_quant)
        dense = self.backend.dequant(*tensors, weight.shape, torch.float32)
        return ((dense - weight.float()).norm() / weight.float().norm()).item()

    def convert_adapted(self, layer: AdaptedLinear) -> int:
        layer.use_nf4(self.backend, self.double_quant)
        return Nf4Weight.stored_bytes(layer.base_weight, layer.base_scale, layer.base_scale2, layer.base_offset)

    def replacement(self, layer: nn.Linear) -> tuple[nn.Module, int]:
        module = Nf4Linear(layer, self.backend, self.double_quant)
        return module, module.stored_bytes


def quantize_frozen(module: nn.Module, *, backend: str = "auto", device="cpu", min_params: int = 1 << 20, double_quant: bool = False):
    """Convert the eligible linear layers of ``module`` to NF4 frozen weights. ``device`` is where it computes (decides ``auto``)."""
    return Nf4Quantizer(make_nf4_backend(backend, device), min_params, double_quant).run(module)
