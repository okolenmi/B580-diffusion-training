"""Frozen weights stored in int8: the autograd op, the module for plain linears, and the pass that converts a network."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor, nn

from ..adapters.layers import AdaptedLinear
from .backends import Int8Backend, make_backend, shape_problem
from .ops import Int8Weight


class Int8Linear(nn.Module):
    """A frozen ``nn.Linear`` that keeps its weight in int8."""

    def __init__(self, original: nn.Linear, backend: Int8Backend):
        super().__init__()
        self.in_features, self.out_features = original.in_features, original.out_features
        self.quantized = Int8Weight(backend)
        w8, scale = backend.prepare(original.weight.detach())
        self.register_buffer("weight_int8", w8)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", original.bias.detach() if original.bias is not None else None)

    def forward(self, x: Tensor) -> Tensor:
        return self.quantized.apply(x, self.weight_int8, self.weight_scale, self.bias)


@dataclass
class QuantizeReport:
    backend: str
    layers: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    skipped: dict[str, int] = field(default_factory=dict)  # reason -> layers

    @property
    def saved_mb(self) -> float:
        return (self.bytes_before - self.bytes_after) / 2**20

    def summary(self) -> str:
        left = "".join(f"; {count} left as they were ({why})" for why, count in sorted(self.skipped.items()))
        return f"int8 base weights ({self.backend}): {self.layers} layers, {self.saved_mb:.0f} MB saved{left}"


class Int8Quantizer:
    """Converts the eligible linear layers of a network to int8, one layer at a time (peak is one dense weight)."""

    def __init__(self, backend: Int8Backend, min_params: int = 1 << 20):
        self.backend, self.min_params = backend, min_params

    def _why_not(self, layer: nn.Module) -> str | None:
        if isinstance(layer, AdaptedLinear):
            if layer.kind != "lora":
                return "DoRA reads the dense weight every step"
            if layer.is_int8:
                return "already int8"
            weight = layer.base_weight
        else:
            weight = layer.weight
        if weight.numel() < self.min_params:
            return f"under {self.min_params:,} parameters"
        return shape_problem(*weight.shape)

    def run(self, module: nn.Module) -> QuantizeReport:
        problem = self.backend.problem()
        if problem is not None:
            raise RuntimeError(f"int8 base weights: {problem}")
        report = QuantizeReport(self.backend.name)
        for parent in list(module.modules()):
            for name, child in list(parent.named_children()):
                if not isinstance(child, (nn.Linear, AdaptedLinear)):
                    continue
                why = self._why_not(child)
                if why is not None:
                    if why != "already int8":
                        report.skipped[why] = report.skipped.get(why, 0) + 1
                    continue
                if isinstance(child, AdaptedLinear):
                    before = child.base_weight.numel() * child.base_weight.element_size()
                    child.use_int8(self.backend)
                    after = child.base_weight.numel() + child.base_scale.numel() * 4
                else:
                    before = child.weight.numel() * child.weight.element_size()
                    replacement = Int8Linear(child, self.backend)
                    setattr(parent, name, replacement)
                    after = replacement.weight_int8.numel() + replacement.weight_scale.numel() * 4
                report.layers += 1
                report.bytes_before += before
                report.bytes_after += after
        return report


def quantize_frozen(module: nn.Module, *, backend: str = "auto", device="cpu", min_params: int = 1 << 20) -> QuantizeReport:
    """Convert the eligible linear layers of ``module`` to int8 frozen weights. ``device`` is where it computes (decides ``auto``)."""
    return Int8Quantizer(make_backend(backend, device), min_params).run(module)
