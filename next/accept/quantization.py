"""Is a quantized network still the network? A dense model against its own quantized copy, reported by region.

Three questions, in the order they are cheapest to answer: how far each layer's weight moved (before anything runs), how far every block's
output moved on fixed inputs, and how far the gradient arriving at every block moved (the training signal). Sizes are relative: norm of
the difference over norm of the reference, so layers and regions compare with each other.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from collections.abc import Callable

from torch import Tensor, nn

from ..families.general.adapters.layers import AdaptedLinear
from ..families.general.quantize.frozen import FrozenQuantizer, QuantizeReport
from .recorder import BlockRecorder, Recording, is_block
from .regions import Regions


def relative_error(value: Tensor, reference: Tensor) -> float:
    reference = reference.float()
    norm = reference.norm().item()
    return (value.float() - reference).norm().item() / norm if norm > 0 else float("nan")


@dataclass
class RegionError:
    region: str
    blocks: int
    output_mean: float
    output_worst: float
    grad_mean: float
    grad_worst: float
    worst_module: str


@dataclass
class ProbeResult:
    precision: str
    report: QuantizeReport | None
    weights: dict[str, float] = field(default_factory=dict)      # layer path -> relative weight error
    modules: dict[str, dict[str, float]] = field(default_factory=dict)  # block path -> {"output": err, "grad": err}
    regions: list[RegionError] = field(default_factory=list)
    final: dict[str, float] = field(default_factory=dict)         # "output", "input_grad"
    dropped: int = 0

    def to_json(self) -> dict:
        return {"precision": self.precision, "summary": self.report.summary() if self.report else None, "weights": self.weights, "modules": self.modules,
                "regions": [vars(r) for r in self.regions], "final": self.final, "dropped": self.dropped}

    def markdown(self, top: int = 8) -> str:
        out = [f"# Quantization probe: {self.precision}", ""]
        if self.report:
            out += [self.report.summary(), ""]
        if self.final:
            out += ["Whole network, on the fixed inputs: " + ", ".join(f"{k} {v:.4f}" for k, v in self.final.items()) + " (relative error against the dense network).", ""]
        if self.weights:
            values = list(self.weights.values())
            out += [f"Weight error over {len(values)} layers: median {statistics.median(values):.4f}, worst {max(values):.4f}.", ""]
            worst = sorted(self.weights.items(), key=lambda kv: -kv[1])[:top]
            out += ["| layer | weight error |", "|---|---:|"] + [f"| {p} | {e:.4f} |" for p, e in worst] + [""]
        out += ["## By region (blocks' outputs and the gradient arriving at them)", "",
                "| region | blocks | output mean | output worst | gradient mean | gradient worst | worst block |", "|---|---:|---:|---:|---:|---:|---|"]
        out += [f"| {r.region} | {r.blocks} | {r.output_mean:.4f} | {r.output_worst:.4f} | {r.grad_mean:.4f} | {r.grad_worst:.4f} | {r.worst_module} |" for r in self.regions]
        if self.dropped:
            out += ["", f"{self.dropped} blocks were not recorded (memory limit)."]
        return "\n".join(out) + "\n"


class QuantizationProbe:
    """``run_model(model)`` must run the network on fixed inputs with gradients enabled and return ``(output, input_grad)``; it is called once
    on the dense network and once on the converted one, so it has to be repeatable (fixed seeds, no dropout)."""

    def __init__(self, regions: Regions | None = None, limit_mb: float = 2048.0, select: Callable[[nn.Module], bool] = is_block):
        self.regions, self.limit_mb, self.select = regions or Regions(), limit_mb, select

    def weight_errors(self, model: nn.Module, quantizer: FrozenQuantizer) -> dict[str, float]:
        """Per eligible layer, without changing it: quantize the weight, rebuild it, compare."""
        errors = {}
        for path, module in model.named_modules():
            if isinstance(module, (nn.Linear, AdaptedLinear)) and quantizer._why_not(module) is None:
                weight = module.base_weight if isinstance(module, AdaptedLinear) else module.weight
                errors[path] = quantizer.weight_error(weight.detach())
        return errors

    def _record(self, model: nn.Module, run_model) -> tuple[Recording, Tensor, Tensor]:
        with BlockRecorder(model, self.select, self.limit_mb) as recorder:
            output, grad = run_model(model)
        return recorder.recording, output.detach().float().cpu(), grad.detach().float().cpu()

    def run(self, model: nn.Module, quantizer: FrozenQuantizer, run_model, *, precision: str,
            convert: Callable[[nn.Module], QuantizeReport] | None = None) -> ProbeResult:
        result = ProbeResult(precision, None)
        result.weights = self.weight_errors(model, quantizer)
        reference, ref_out, ref_grad = self._record(model, run_model)
        result.report = convert(model) if convert is not None else quantizer.run(model)
        current, out, grad = self._record(model, run_model)
        result.final = {"output": relative_error(out, ref_out), "input_grad": relative_error(grad, ref_grad)}
        result.dropped = max(reference.dropped, current.dropped)
        for path, ref in reference.outputs.items():
            if path in current.outputs:
                errors = {"output": relative_error(current.outputs[path], ref)}
                if path in reference.grads and path in current.grads:
                    errors["grad"] = relative_error(current.grads[path], reference.grads[path])
                result.modules[path] = errors
        result.regions = self._regions(result.modules)
        return result

    def _regions(self, modules: dict[str, dict[str, float]]) -> list[RegionError]:
        grouped: dict = {}
        for path, errors in modules.items():
            grouped.setdefault(self.regions.of(path), []).append((path, errors))
        rows = []
        for region in self.regions.order(grouped):
            items = grouped[region]
            outs = [e["output"] for _, e in items]
            grads = [e["grad"] for _, e in items if "grad" in e] or [float("nan")]
            worst = max(items, key=lambda it: it[1].get("grad", it[1]["output"]))[0]
            rows.append(RegionError(region.name, len(items), statistics.fmean(outs), max(outs), statistics.fmean(grads), max(grads), worst))
        return rows
