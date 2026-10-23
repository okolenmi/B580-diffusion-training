"""What every frozen-weight quantizer shares: the report and the walk over a network's linear layers."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn

from ..adapters.layers import AdaptedLinear


@dataclass
class QuantizeReport:
    backend: str
    label: str = "int8"
    layers: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    skipped: dict[str, int] = field(default_factory=dict)  # reason -> layers

    @property
    def saved_mb(self) -> float:
        return (self.bytes_before - self.bytes_after) / 2**20

    def summary(self) -> str:
        left = "".join(f"; {count} left as they were ({why})" for why, count in sorted(self.skipped.items()))
        return f"{self.label} base weights ({self.backend}): {self.layers} layers, {self.saved_mb:.0f} MB saved{left}"


class FrozenQuantizer:
    """Converts the eligible linear layers of a network one layer at a time (the peak is one dense weight).

    A subclass says which layers can convert (``shape_problem``), how an adapted layer converts (``convert_adapted``, returns the bytes it
    now holds) and how a plain ``nn.Linear`` converts (``replacement``, returns the new module and its bytes).
    """

    label = "?"

    def __init__(self, backend, min_params: int):
        self.backend, self.min_params = backend, min_params

    def shape_problem(self, weight_shape) -> str | None:
        raise NotImplementedError

    def weight_error(self, weight) -> float:
        """Relative error of this layer's weight after quantizing and rebuilding it (nothing is changed)."""
        raise NotImplementedError

    def convert_adapted(self, layer: AdaptedLinear) -> int:
        raise NotImplementedError

    def replacement(self, layer: nn.Linear) -> tuple[nn.Module, int]:
        raise NotImplementedError

    def _why_not(self, layer: nn.Module) -> str | None:
        if isinstance(layer, AdaptedLinear):
            if layer.kind != "lora":
                return "DoRA reads the dense weight every step"
            if layer.is_quantized:
                return "already quantized"
            weight = layer.base_weight
        else:
            weight = layer.weight
        if weight.numel() < self.min_params:
            return f"under {self.min_params:,} parameters"
        return self.shape_problem(weight.shape)

    def run(self, module: nn.Module) -> QuantizeReport:
        problem = self.backend.problem()
        if problem is not None:
            raise RuntimeError(f"{self.label} base weights: {problem}")
        report = QuantizeReport(self.backend.name, self.label)
        for parent in list(module.modules()):
            for name, child in list(parent.named_children()):
                if not isinstance(child, (nn.Linear, AdaptedLinear)):
                    continue
                why = self._why_not(child)
                if why is not None:
                    if why != "already quantized":
                        report.skipped[why] = report.skipped.get(why, 0) + 1
                    continue
                weight = child.base_weight if isinstance(child, AdaptedLinear) else child.weight
                before = weight.numel() * weight.element_size()
                if isinstance(child, AdaptedLinear):
                    after = self.convert_adapted(child)
                else:
                    replacement, after = self.replacement(child)
                    setattr(parent, name, replacement)
                report.layers += 1
                report.bytes_before += before
                report.bytes_after += after
        return report
