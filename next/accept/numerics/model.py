"""Checks on a whole denoiser with an adapter: the same forward and backward run in ways that must agree, and the differences are shown by region."""

from __future__ import annotations

import contextlib
import copy

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ...families.general.ldm_unet.fused import FusedGeglu
from ...families.general.quantize.int8 import quantize_frozen
from ..regions import Regions
from .base import Finding, NumericCheck
from .context import Snapshot
from .measure import differing, finite, relative

BF16_STEP = 2.0 ** -8  # the size of one bf16 rounding step relative to the value
WARN_ABOVE = 0.05  # a relative difference this large is not rounding any more


def grad_difference(a: Snapshot, b: Snapshot) -> tuple[float, dict[str, float]]:
    """Relative difference of all adapter gradients together, and of each region's."""
    regions = Regions(coarse=True)
    sums: dict[str, list[float]] = {}
    total = [0.0, 0.0]
    for key, ref in b.grads.items():
        gap = (a.grads[key].double() - ref.double()).pow(2).sum().item()
        size = ref.double().pow(2).sum().item()
        group = sums.setdefault(regions.of(key).name, [0.0, 0.0])
        group[0] += gap
        group[1] += size
        total[0] += gap
        total[1] += size

    def share(gap: float, size: float) -> float:
        return 0.0 if gap == 0.0 else (gap / size) ** 0.5 if size else float("inf")

    return share(*total), {name: share(*pair) for name, pair in sorted(sums.items())}


class ModelCheck(NumericCheck):
    needs_model = True

    def compare(self, name: str, got: Snapshot, ref: Snapshot, note: str = "") -> Finding:
        grads, by_region = grad_difference(got, ref)
        out, dx = relative(got.out, ref.out), relative(got.dx, ref.dx)
        regions = " ".join(f"{k} {v:.1e}" for k, v in by_region.items())
        detail = f"output {out:.1e}, input gradient {dx:.1e}, adapter gradients {grads:.1e} ({regions}){note}"
        bad = not (finite(got.out) and finite(got.dx)) or max(out, dx, grads) > WARN_ABOVE
        return self.measured(name, detail, {"output": out, "input_gradient": dx, "adapter_gradients": grads, "by_region": by_region}, warn_if=bad)


class ModelRepeat(ModelCheck):
    name = "model-repeat"
    purpose = "the same forward and backward, run again, gives the same output and gradients"

    def run(self, ctx) -> list[Finding]:
        subject = ctx.subject
        inputs = subject.inputs()
        first = subject.evaluate(inputs)
        out_bad = dx_bad = grad_bad = 0
        runs = 4
        for _ in range(runs):
            again = subject.evaluate(inputs)
            out_bad += differing(again.out, first.out)
            dx_bad += differing(again.dx, first.dx)
            grad_bad += sum(differing(again.grads[k], v) for k, v in first.grads.items())
        grads = sum(v.numel() for v in first.grads.values()) * runs
        found = [self.exact("output", out_bad, first.out.numel() * runs), self.exact("input gradient", dx_bad, first.dx.numel() * runs)]
        found.append(self.measured("adapter gradients", f"{grad_bad} of {grads} values differ", {"differing": grad_bad}, warn_if=grad_bad > 0))
        return found


class ComputeDtype(ModelCheck):
    name = "compute-dtype"
    purpose = "the LoRA branch in float32 against in the activation dtype: how far apart the outputs and the adapter gradients are, by region"

    def run(self, ctx) -> list[Finding]:
        subject = ctx.subject
        inputs = subject.inputs()
        subject.set_compute("float32")
        reference = subject.evaluate(inputs)
        subject.set_compute("activation")
        activation = subject.evaluate(inputs)
        subject.set_compute("float32")
        nudged = copy.copy(inputs)
        nudged.x = (inputs.x.float() * (1 + BF16_STEP)).to(inputs.x.dtype)  # every input value one bf16 step larger
        scale = subject.evaluate(nudged)
        return [self.compare("activation against float32", activation, reference),
                self.compare("scale: the input made 1/256 larger, float32 branch", scale, reference,
                             " <- what a change of one bf16 step in the input does; a difference between the two modes of this size is rounding noise")]


class ChunkedGradients(ModelCheck):
    name = "chunks"
    purpose = "a batch of two run whole, and as two batches of one with the gradients added (what memory splitting does)"

    def run(self, ctx) -> list[Finding]:
        subject = ctx.subject
        whole_inputs = subject.inputs(batch=2)
        whole = subject.evaluate(whole_inputs)
        parts = []
        for i in range(2):
            part = copy.copy(whole_inputs)
            for name in ("x", "t", "context", "adm", "dy"):
                setattr(part, name, getattr(whole_inputs, name)[i:i + 1])
            parts.append(subject.evaluate(part))
        summed = Snapshot(torch.cat([p.out for p in parts]), torch.cat([p.dx for p in parts]),
                          {k: parts[0].grads[k] + parts[1].grads[k] for k in parts[0].grads})
        return [self.compare("two chunks of one against the whole batch", summed, whole)]


class CheckpointingEquality(ModelCheck):
    name = "checkpointing"
    purpose = "recomputing activations in backward gives the same gradients as keeping them"

    def run(self, ctx) -> list[Finding]:
        subject = ctx.subject
        inputs = subject.inputs()
        subject.set_checkpointing(False)
        kept = subject.evaluate(inputs)
        subject.set_checkpointing(True)
        recomputed = subject.evaluate(inputs)
        subject.set_checkpointing(False)
        bad = differing(recomputed.out, kept.out) + sum(differing(recomputed.grads[k], v) for k, v in kept.grads.items())
        found = [self.compare("recomputed against kept", recomputed, kept)]
        found.append(self.measured("bit for bit", f"{bad} output and gradient values differ", {"differing": bad}))
        return found


class AttentionModes(ModelCheck):
    name = "attention"
    purpose = "every attention path against the stock math backend: how far each is from it"

    def run(self, ctx) -> list[Finding]:
        subject = ctx.subject
        inputs = subject.inputs()

        def math():
            return sdpa_kernel(SDPBackend.MATH)

        scopes = {"default kernel": contextlib.nullcontext, **subject.denoiser.attention_scopes()}
        with math():
            reference = subject.evaluate(inputs)
        found = []
        for name, scope in scopes.items():
            with scope():
                snapshot = subject.evaluate(inputs)
            found.append(self.compare(f"{name} against math", snapshot, reference))
        return found


class FusedGegluEquality(ModelCheck):
    name = "fused-geglu"
    purpose = "the fused feed-forward activation against the eager chain, forward and backward"

    def run(self, ctx) -> list[Finding]:
        subject = ctx.subject
        inputs = subject.inputs()
        FusedGeglu.disable()
        eager = subject.evaluate(inputs)
        reason = FusedGeglu.enable(ctx.device)
        if reason:
            return [self.skip("fused against eager", reason)]
        try:
            fused = subject.evaluate(inputs)
        finally:
            FusedGeglu.disable()
        bad = differing(fused.out, eager.out) + sum(differing(fused.grads[k], v) for k, v in eager.grads.items())
        return [self.compare("fused against eager", fused, eager), self.measured("bit for bit", f"{bad} output and gradient values differ", {"differing": bad})]


class ModelOffload(ModelCheck):
    name = "model-offload"
    purpose = "the whole model (weights and adapter) moved to the host and back gives the same output and gradients"

    def run(self, ctx) -> list[Finding]:
        subject = ctx.subject
        inputs = subject.inputs()
        before = subject.evaluate(inputs)
        subject.denoiser.to("cpu")
        ctx.release()
        subject.denoiser.to(ctx.device)
        after = subject.evaluate(inputs)
        return [self.exact("output", differing(after.out, before.out), before.out.numel()),
                self.exact("adapter gradients", sum(differing(after.grads[k], v) for k, v in before.grads.items()), sum(v.numel() for v in before.grads.values()), "values")]


class ModelInt8(ModelCheck):
    name = "model-int8"
    purpose = "the same model with int8 frozen weights (converts the subject, so it runs last): its error, and its offload round trip"

    def run(self, ctx) -> list[Finding]:
        subject = ctx.subject
        inputs = subject.inputs()
        dense = subject.evaluate(inputs)
        try:
            report = quantize_frozen(subject.denoiser.module(), backend="auto", device=ctx.device, min_params=0 if ctx.tiny else 1 << 20)
        except RuntimeError as error:
            return [self.skip("int8", f"unavailable here: {str(error)[:120]}")]
        quantized = subject.evaluate(inputs)
        subject.denoiser.to("cpu")
        ctx.release()
        subject.denoiser.to(ctx.device)
        again = subject.evaluate(inputs)
        return [self.compare(f"int8 ({report.backend}, {report.layers} layers) against bf16", quantized, dense),
                self.exact("int8 model, offload and reload, output", differing(again.out, quantized.out), quantized.out.numel()),
                self.exact("int8 model, offload and reload, adapter gradients", sum(differing(again.grads[k], v) for k, v in quantized.grads.items()),
                           sum(v.numel() for v in quantized.grads.values()), "values")]
