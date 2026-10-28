"""Checks on a whole denoiser with an adapter: the same forward and backward run in ways that must agree, and the differences are shown by region."""

from __future__ import annotations

import contextlib
import copy
import warnings
from dataclasses import dataclass

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ...families.general.ldm_unet.fused import FusedGeglu
from ...families.general.quantize.int8 import quantize_frozen
from ..regions import Regions
from .base import FAIL, Finding, NumericCheck
from .context import Snapshot
from .measure import differing, finite, relative, spread

BF16_STEP = 2.0 ** -8  # the size of one bf16 rounding step relative to the value
WARN_ABOVE = 2.0  # times the reference: a difference this much past what one bf16 step of the input does is not rounding any more


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


@dataclass
class Reference:
    """What a change of one bf16 step in every input value does to the output and gradients: the size of difference rounding alone produces."""

    out: float
    dx: float
    grads: float
    by_region: dict[str, float]

    @classmethod
    def of(cls, ctx) -> Reference:
        if "reference" not in ctx.cache:
            subject = ctx.subject
            inputs = subject.inputs()
            subject.set_compute("float32")
            base = subject.evaluate(inputs)
            nudged = copy.copy(inputs)
            nudged.x = (inputs.x.float() * (1 + BF16_STEP)).to(inputs.x.dtype)
            moved = subject.evaluate(nudged)
            grads, by_region = grad_difference(moved, base)
            ctx.cache["reference"] = cls(relative(moved.out, base.out), relative(moved.dx, base.dx), grads, by_region)
        return ctx.cache["reference"]


class ModelCheck(NumericCheck):
    needs_model = True

    def compare(self, ctx, name: str, got: Snapshot, ref: Snapshot, note: str = "") -> Finding:
        """Two valid runs: shown next to the reference; WARN only when one of them is past WARN_ABOVE times it."""
        floor = Reference.of(ctx)
        grads, by_region = grad_difference(got, ref)
        out, dx = relative(got.out, ref.out), relative(got.dx, ref.dx)
        times = {"output": out / max(floor.out, 1e-12), "input gradient": dx / max(floor.dx, 1e-12), "adapter gradients": grads / max(floor.grads, 1e-12)}
        regions = " ".join(f"{k} {v:.1e}" for k, v in by_region.items())
        detail = (f"output {out:.1e} (x{times['output']:.1f}), input gradient {dx:.1e} (x{times['input gradient']:.1f}), "
                  f"adapter gradients {grads:.1e} (x{times['adapter gradients']:.1f}; {regions}){note}")
        numbers = {"output": out, "input_gradient": dx, "adapter_gradients": grads, "by_region": by_region, "times_reference": times}
        if not (finite(got.out) and finite(got.dx)):
            return Finding(self.name, name, FAIL, "not finite: " + detail, numbers)
        return self.measured(name, detail, numbers, warn_if=max(times.values()) > WARN_ABOVE)


@contextlib.contextmanager
def deterministic_algorithms(on: bool):
    """The switch ``runtime.deterministic`` turns on; yields the operations that said they have no deterministic kernel."""
    was = torch.are_deterministic_algorithms_enabled()
    missing: set[str] = set()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.use_deterministic_algorithms(on, warn_only=True)
            yield missing
        missing.update(str(w.message).split(" does not have a deterministic")[0].strip("'\" ")[:60] for w in caught if "deterministic" in str(w.message))
    finally:
        torch.use_deterministic_algorithms(was)


class ModelRepeat(ModelCheck):
    name = "model-repeat"
    purpose = "the same forward and backward, run again: is the output the same, and is the backward? (default kernels and math attention, with and without deterministic algorithms)"

    def run(self, ctx) -> list[Finding]:
        subject = ctx.subject
        inputs = subject.inputs()
        floor = Reference.of(ctx)
        math = lambda: sdpa_kernel(SDPBackend.MATH)  # noqa: E731
        found = []
        for label, scope, deterministic in (("default kernels", contextlib.nullcontext, False), ("math attention", math, False),
                                            ("default kernels, deterministic algorithms", contextlib.nullcontext, True),
                                            ("math attention, deterministic algorithms", math, True)):
            with scope(), deterministic_algorithms(deterministic) as missing:
                first = subject.evaluate(inputs)
                runs = [subject.evaluate(inputs) for _ in range(3)]
            out_bad, _ = spread([r.out for r in runs], first.out)
            dx_bad, dx_gap = spread([r.dx for r in runs], first.dx)
            grad_bad, grad_gap = 0, 0.0
            for r in runs:
                grad_bad += sum(differing(r.grads[k], v) for k, v in first.grads.items())
                grad_gap = max(grad_gap, grad_difference(r, first)[0])
            total = sum(v.numel() for v in first.grads.values()) * len(runs)
            found.append(self.exact(f"{label}: output", out_bad, first.out.numel() * len(runs)))
            note = f"; operations without a deterministic kernel: {', '.join(sorted(missing))}" if missing else ""
            detail = (f"input gradient: {dx_bad} of {first.dx.numel() * len(runs)} values differ, up to {dx_gap:.1e} (x{dx_gap / max(floor.dx, 1e-12):.2f} of the reference); "
                      f"adapter gradients: {grad_bad} of {total} differ, up to {grad_gap:.1e} (x{grad_gap / max(floor.grads, 1e-12):.2f}){note}")
            repeatable = dx_bad == 0 and grad_bad == 0
            found.append(self.measured(f"{label}: gradients", "identical in every repeat" if repeatable else detail,
                                       {"input_gradient": dx_gap, "adapter_gradients": grad_gap, "dx_differing": dx_bad, "grads_differing": grad_bad},
                                       warn_if=not repeatable and deterministic and label.startswith("math")))
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
        floor = Reference.of(ctx)
        regions = " ".join(f"{k} {v:.1e}" for k, v in floor.by_region.items())
        return [self.compare(ctx, "activation against float32", activation, reference),
                self.measured("reference: the input made 1/256 larger", f"output {floor.out:.1e}, input gradient {floor.dx:.1e}, adapter gradients {floor.grads:.1e} ({regions}) "
                              "<- what a change of one bf16 step in the input does; the (x) factors of every comparison are against this",
                              {"output": floor.out, "input_gradient": floor.dx, "adapter_gradients": floor.grads, "by_region": floor.by_region})]


class ChunkedGradients(ModelCheck):
    name = "chunks"
    purpose = "a batch of two run whole, and as two batches of one with the gradients added (what memory splitting does); where they differ, whether the batch position or the sample causes it"

    FIELDS = ("x", "t", "context", "adm", "dy")

    @classmethod
    def pick(cls, inputs, rows: list[int]):
        part = copy.copy(inputs)
        for name in cls.FIELDS:
            setattr(part, name, getattr(inputs, name)[rows])
        return part

    def run(self, ctx) -> list[Finding]:
        subject = ctx.subject
        whole_inputs = subject.inputs(batch=2)
        found = []
        for label, scope in (("default kernels", contextlib.nullcontext), ("math attention", lambda: sdpa_kernel(SDPBackend.MATH))):
            with scope():
                whole = subject.evaluate(whole_inputs)
                parts = [subject.evaluate(self.pick(whole_inputs, [i])) for i in range(2)]
            summed = Snapshot(torch.cat([p.out for p in parts]), torch.cat([p.dx for p in parts]),
                              {k: parts[0].grads[k] + parts[1].grads[k] for k in parts[0].grads})
            per_sample = ", ".join(f"sample {i}: output {relative(summed.out[i], whole.out[i]):.1e}, input gradient {relative(summed.dx[i], whole.dx[i]):.1e}" for i in range(2))
            found.append(self.compare(ctx, f"two chunks of one against the whole batch, {label}", summed, whole, f"; {per_sample}"))
        return found + [self._diagnose(subject, whole_inputs, whole, parts)]

    def _diagnose(self, subject, inputs, whole: Snapshot, parts: list[Snapshot]) -> Finding:
        """Sample 1 four ways: alone, at position 1 of the batch, at position 0 of the swapped batch, and alone with a nudged input."""
        swapped = subject.evaluate(self.pick(inputs, [1, 0]))
        alone = self.pick(inputs, [1])
        nudged = copy.copy(alone)
        nudged.x = (alone.x.float() * (1 + BF16_STEP)).to(alone.x.dtype)
        moved = subject.evaluate(nudged)

        def pair(a: Snapshot, i: int, b: Snapshot, j: int) -> str:
            return f"output {relative(a.out[i], b.out[j]):.1e} input gradient {relative(a.dx[i], b.dx[j]):.1e}"

        detail = (f"sample 1 alone against at position 1 of the batch: {pair(parts[1], 0, whole, 1)}; alone against at position 0 of the swapped batch: {pair(parts[1], 0, swapped, 0)}; "
                  f"position 1 against position 0 of the swapped batch: {pair(whole, 1, swapped, 0)}; the reference for this sample alone (input 1/256 larger): {pair(moved, 0, parts[1], 0)}")
        return self.measured("sample 1, by batch position", detail, {})


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
        found = [self.compare(ctx, "recomputed against kept", recomputed, kept)]
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
            found.append(self.compare(ctx, f"{name} against math", snapshot, reference))
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
        return [self.compare(ctx, "fused against eager", fused, eager), self.measured("bit for bit", f"{bad} output and gradient values differ", {"differing": bad})]


class OffloadRoundTrip(ModelCheck):
    """The model moved to the host and back: every stored tensor and the output must be the same bits. Gradients are held to the spread
    two plain repeats have, since the backward may not repeat bit for bit on a device."""

    def round_trip(self, ctx, label: str) -> list[Finding]:
        subject = ctx.subject
        inputs = subject.inputs()
        before_digest = subject.weights_digest()
        first = subject.evaluate(inputs)
        repeat = subject.evaluate(inputs)
        subject.denoiser.to("cpu")
        ctx.release()
        subject.denoiser.to(ctx.device)
        after_digest = subject.weights_digest()
        after = subject.evaluate(inputs)
        changed = [k for k, v in before_digest.items() if after_digest.get(k) != v]
        floor, moved = grad_difference(repeat, first)[0], grad_difference(after, first)[0]
        return [self.exact(f"{label}stored tensors (weights, codes, scales, adapter)", len(changed), len(before_digest), "tensors"),
                self.exact(f"{label}output", differing(after.out, first.out), first.out.numel()),
                self.measured(f"{label}adapter gradients against the spread of a plain repeat", f"after the round trip {moved:.1e}, between two plain repeats {floor:.1e}",
                              {"after": moved, "repeat": floor}, warn_if=moved > max(3 * floor, 1e-6))]


class ModelOffload(OffloadRoundTrip):
    name = "model-offload"
    purpose = "the whole model (weights and adapter) moved to the host and back: the same stored bytes, output and gradients"

    def run(self, ctx) -> list[Finding]:
        return self.round_trip(ctx, "")


class ModelInt8(OffloadRoundTrip):
    name = "model-int8"
    purpose = "the same model with int8 frozen weights (converts the subject, so it runs last): its error, and its offload round trip"

    def run(self, ctx) -> list[Finding]:
        subject = ctx.subject
        inputs = subject.inputs()
        Reference.of(ctx)  # before the subject changes
        dense = subject.evaluate(inputs)
        try:
            report = quantize_frozen(subject.denoiser.module(), backend="auto", device=ctx.device, min_params=0 if ctx.tiny else 1 << 20)
        except RuntimeError as error:
            return [self.skip("int8", f"unavailable here: {str(error)[:120]}")]
        quantized = subject.evaluate(inputs)
        return [self.compare(ctx, f"int8 ({report.backend}, {report.layers} layers) against bf16", quantized, dense), *self.round_trip(ctx, "int8 model, offload and reload: ")]
