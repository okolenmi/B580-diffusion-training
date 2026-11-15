"""What merging an adapter into the weights does to the model's output, against running it beside them."""

from __future__ import annotations

import torch

from .base import Finding
from .measure import relative
from .model import ModelCheck, Reference


class MergeEffect(ModelCheck):
    name = "merge-effect"
    purpose = ("the adapter's effect on the model's output when it is added into the bf16 weights (what a UI does at generation) against run beside them (how it "
               "trained); needs host memory for a copy of the adapted weights (about 5 GB for SDXL)")

    STRENGTHS = (0.5, 1.0)

    def run(self, ctx) -> list[Finding]:
        subject = ctx.subject
        if not (ctx.lora or ctx.tiny):
            return [self.skip("merge effect", "needs --lora (the other checks use random factors that do not look like a trained adapter)")]
        layers = [layer for layer in subject.layers if layer.kind == "lora" and not layer.is_quantized]
        inputs = subject.inputs()
        subject.set_compute("float32")
        saved_weights = [layer.base_weight.detach().cpu().clone() for layer in layers]
        saved_strength = [layer.training_weight for layer in layers]
        try:
            self._strength(layers, 0.0)
            base = subject.output(inputs)
            found = []
            for strength in self.STRENGTHS:
                self._strength(layers, strength)
                beside = subject.output(inputs)
                self._merge(layers)
                merged = subject.output(inputs)
                self._restore(layers, saved_weights)
                found.append(self._finding(ctx, strength, len(layers), base, beside, merged))
        finally:
            self._restore(layers, saved_weights)
            for layer, value in zip(layers, saved_strength, strict=True):
                layer.training_weight = value
        return found

    @staticmethod
    def _strength(layers: list, value: float) -> None:
        for layer in layers:
            layer.training_weight = value

    @staticmethod
    def _merge(layers: list) -> None:
        with torch.no_grad():
            for layer in layers:
                weight = layer.base_weight
                weight.copy_((weight.float() + layer._delta().float()).to(weight.dtype))
                layer.training_weight = 0.0

    @staticmethod
    def _restore(layers: list, saved: list) -> None:
        with torch.no_grad():
            for layer, weight in zip(layers, saved, strict=True):
                layer.base_weight.copy_(weight.to(layer.base_weight.device))

    def _finding(self, ctx, strength: float, count: int, base, beside, merged) -> Finding:
        effect = beside - base
        size = effect.double().norm().item()
        delivered = float((merged - base).double().flatten() @ effect.double().flatten()) / max(size * size, 1e-30)
        error = relative(merged, beside) * beside.double().norm().item() / max(size, 1e-30)
        jitter = Reference.of(ctx).out * base.double().norm().item() / max(size, 1e-30)
        detail = (f"{count} layers merged at strength {strength}: the adapter changes the output by {size / max(base.double().norm().item(), 1e-30):.1%}; "
                  f"the merged model delivers {delivered:.0%} of that change and differs from the adapter run beside the weights by {error:.0%} of it "
                  f"(one bf16 step in the input moves the output by {jitter:.0%} of it, for scale)")
        return self.measured(f"strength {strength}", detail, {"effect": size, "delivered": delivered, "error_over_effect": error, "reference_over_effect": jitter},
                             warn_if=not 0.9 <= delivered <= 1.1 or error > max(0.25, 2 * jitter))
