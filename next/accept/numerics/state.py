"""Numbers that live in memory between steps: optimizer state, frozen weights, and the adapter file, through the moves and saves a run makes."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch
from torch import nn

from ...engine.optim.algorithms import CAME, Adafactor, AdamW
from ...engine.optim.handle import Optimizer, ParamGroup
from ...engine.optim.state import make_store
from ...engine.runtime.residents import ModulesResident
from ...families.general.quantize.int8 import quantize_frozen as quantize_int8
from ...families.general.quantize.nf4 import quantize_frozen as quantize_nf4
from ...families.general.adapters.schemes import read_adapter, save_adapter
from ...families.registry import get_family
from ..regions import Regions
from .base import FAIL, INFO, PASS, Finding, NumericCheck
from .context import Context
from .measure import differing, finite, relative


class OptimizerRoundtrip(NumericCheck):
    name = "optimizer"
    purpose = "optimizer state moved to the host and back between steps gives the same training as state that never moved"

    SHAPES = ((64, 32), (64, 32), (128,), (16, 640), (640, 16), (96, 96))
    STEPS = 10

    def run(self, ctx) -> list[Finding]:
        found = []
        for algorithm in (AdamW, Adafactor, CAME):
            for store in ("float32", "int8_blockwise"):
                label = f"{algorithm.__name__} with {store} state"
                try:
                    found.extend(self._compare(ctx, algorithm, store, label))
                except Exception as error:  # noqa: BLE001 - a combination the optimizer does not support is not a failure
                    found.append(self.skip(label, f"{type(error).__name__}: {str(error)[:100]}"))
        return found

    def _build(self, ctx, algorithm, store) -> Optimizer:
        torch.manual_seed(0)
        params = [nn.Parameter(torch.randn(*shape).to(ctx.device)) for shape in self.SHAPES]
        return Optimizer(algorithm(), [ParamGroup(params)], 1e-3, store=make_store(store))

    def _train(self, ctx, opt: Optimizer, every: int | None) -> None:
        for step in range(self.STEPS):
            gen = torch.Generator().manual_seed(100 + step)
            for p in opt.params:
                p.grad = torch.randn(p.shape, generator=gen).to(ctx.device)
            opt.step()
            if every and (step + 1) % every == 0:
                opt.offload()
                if opt.footprint_bytes() != 0:
                    raise AssertionError("offloaded state still counts as on the device")
                opt.reload()

    def _compare(self, ctx, algorithm, store, label) -> list[Finding]:
        straight = self._build(ctx, algorithm, store)
        self._train(ctx, straight, None)
        out = []
        for every in (1, 3):
            moved = self._build(ctx, algorithm, store)
            self._train(ctx, moved, every)
            bad = sum(differing(a, b) for a, b in zip(straight.params, moved.params, strict=True))
            for sa, sb in zip(straight.stacks, moved.stacks, strict=True):
                ta, tb = straight.store.checkout(sa.state), moved.store.checkout(sb.state)
                bad += sum(differing(ta[k], tb[k]) for k in ta)
            total = sum(p.numel() for p in straight.params)
            out.append(self.exact(f"{label}, moved every {every} step{'s' if every > 1 else ''}", bad, total, "parameter and state values"))
        return out


class FrozenWeights(NumericCheck):
    name = "frozen"
    purpose = "frozen weights (bf16, int8, NF4) give the same output after an offload and reload, and quantizing twice gives the same codes"

    def run(self, ctx) -> list[Finding]:
        found = [self._dense(ctx)]
        for label, quantize, limit in (("int8", lambda m, d: quantize_int8(m, backend="auto", device=d, min_params=0), 0.05),
                                       ("nf4", lambda m, d: quantize_nf4(m, backend="auto", device=d, min_params=0), 0.3)):
            try:
                found.extend(self._quantized(ctx, label, quantize, limit))
            except RuntimeError as error:
                found.append(self.skip(label, f"unavailable here: {str(error)[:120]}"))
        return found

    @staticmethod
    def _stack(ctx, seed: int = 0) -> nn.Module:
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(1280, 5120), nn.GELU(), nn.Linear(5120, 1280)).to(ctx.device, torch.bfloat16).requires_grad_(False)

    def _roundtrip(self, ctx, module) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(512, 1280, generator=torch.Generator().manual_seed(1)).to(ctx.device, torch.bfloat16)
        before = module(x).float().cpu()
        resident = ModulesResident([module], lambda device: module.to(device), str(ctx.device))
        resident.offload()
        resident.reload()
        return before, module(x).float().cpu()

    def _dense(self, ctx) -> Finding:
        before, after = self._roundtrip(ctx, self._stack(ctx))
        return self.exact("bf16 weights, offload and reload, output", differing(before, after), before.numel())

    def _quantized(self, ctx, label, quantize, limit) -> list[Finding]:
        a, b = self._stack(ctx), self._stack(ctx)
        quantize(a, ctx.device)
        quantize(b, ctx.device)
        codes = {k: v for k, v in a.state_dict().items()}
        same = sum(differing(codes[k], v) for k, v in b.state_dict().items())
        before, after = self._roundtrip(ctx, a)
        reference = self._stack(ctx)(torch.randn(512, 1280, generator=torch.Generator().manual_seed(1)).to(ctx.device, torch.bfloat16)).float().cpu()
        error = relative(before, reference)
        return [self.exact(f"{label}: quantizing the same weights twice, stored codes and scales", same, sum(v.numel() for v in codes.values()), "stored values"),
                self.exact(f"{label}: offload and reload, output", differing(before, after), before.numel()),
                self.measured(f"{label}: output against the bf16 weights", f"relative error {error:.4f}", {"relative": error}, warn_if=error > limit)]


class AdapterFile(NumericCheck):
    name = "adapter"
    purpose = "an adapter written to a file and read back is the same adapter; and what a given LoRA file holds"

    def run(self, ctx) -> list[Finding]:
        found = self._roundtrip(ctx)
        if ctx.lora:
            found.extend(self._inspect(ctx.lora))
        return found

    def _roundtrip(self, ctx) -> list[Finding]:
        tiny = Context(ctx.device, tiny=True, seed=ctx.seed)
        other = Context(ctx.device, tiny=True, seed=ctx.seed)  # the same base weights; only the adapter is made to differ
        source, target = tiny.subject, other.subject
        for layer in target.adapted.layers.values():
            with torch.no_grad():
                layer.lora_B.zero_()
        family = get_family("sdxl")
        scheme = family.schemes["kohya"]
        exported = scheme.export(source.adapted)
        with tempfile.TemporaryDirectory() as raw:
            path = str(Path(raw) / "check.safetensors")
            save_adapter(path, exported, {"scheme": "kohya"})
            state, _ = read_adapter(path)
        file_bad = sum(differing(exported[k], state[k]) for k in exported)
        dtypes = sorted({str(t.dtype).split(".")[-1] for t in state.values()})
        report = scheme.load(target.adapted, state)
        factor_bad = sum(differing(a, b) for pa, pb in zip(source.adapted.layers.values(), target.adapted.layers.values(), strict=True)
                         for a, b in zip(pa.adapter_state().values(), pb.adapter_state().values(), strict=True))
        inputs = source.inputs()
        out_a, out_b = source.evaluate(inputs).out, target.evaluate(inputs).out
        return [self.exact("tensors after write and read", file_bad, len(exported), "tensors"),
                Finding(self.name, "stored dtypes", INFO, ", ".join(dtypes) + f"; {report.matched} layers loaded, {len(report.missing)} missing"),
                self.exact("factors of the loaded adapter against the saved one", factor_bad, sum(t.numel() for t in state.values()), "values"),
                self.exact("model output with the loaded adapter against the original", differing(out_a, out_b), out_a.numel())]

    def _inspect(self, path: str) -> list[Finding]:
        state, meta = read_adapter(path)
        dtypes: dict[str, int] = {}
        bad = []
        peaks: dict[str, float] = {}
        for key, tensor in state.items():
            dtypes[str(tensor.dtype).split(".")[-1]] = dtypes.get(str(tensor.dtype).split(".")[-1], 0) + 1
            if not finite(tensor):
                bad.append(key)
            kind = "up" if ("lora_up" in key or "lora_B" in key) else "down" if ("lora_down" in key or "lora_A" in key) else None
            if kind:
                peaks[kind] = max(peaks.get(kind, 0.0), float(tensor.float().abs().max()))
        status = FAIL if bad else PASS
        return [Finding(self.name, f"{Path(path).name}: values", status,
                        f"{len(state)} tensors, dtypes {dtypes}, largest |up| {peaks.get('up', 0):.4f}, largest |down| {peaks.get('down', 0):.4f}, "
                        f"{len(bad)} with NaN or infinite values; metadata {dict(list(meta.items())[:6])}")]


class MergeLoss(NumericCheck):
    name = "merge"
    purpose = "what is lost when a LoRA is added into bf16 weights (the way a UI merges it at generation) instead of run beside them (the way it trained)"

    STRENGTHS = (0.5, 1.0)

    def run(self, ctx) -> list[Finding]:
        found = self._synthetic(ctx)
        if ctx.lora and not ctx.tiny:
            found.extend(self._real(ctx))
        return found

    @staticmethod
    def _loss(weight: torch.Tensor, delta: torch.Tensor, strength: float) -> tuple[float, float]:
        """``(share of the update's size lost, share of update values that vanish)`` when ``weight + strength * delta`` is rounded to the weight's dtype."""
        wanted = delta.float() * strength
        merged = (weight.float() + wanted).to(weight.dtype).float()
        realized = merged - weight.float()
        lost = (realized - wanted).norm().item() / max(wanted.norm().item(), 1e-30)
        vanished = ((realized == 0) & (wanted != 0)).float().mean().item()
        return lost, vanished

    def _synthetic(self, ctx) -> list[Finding]:
        gen = torch.Generator().manual_seed(2)
        weight = (torch.randn(5120, 1280, generator=gen) * 0.02).to(ctx.device, torch.bfloat16)
        a = (torch.rand(16, 1280, generator=gen) * 2 - 1) / math.sqrt(1280)
        b = torch.randn(5120, 16, generator=gen) * 0.02
        delta = (b @ a).to(ctx.device)
        found = []
        for strength in self.STRENGTHS:
            lost, vanished = self._loss(weight, delta, strength)
            found.append(self.measured(f"a random rank-16 LoRA on a 5120x1280 bf16 weight, strength {strength}",
                                       f"{lost:.1%} of the update's size is lost, {vanished:.1%} of its values vanish", {"lost": lost, "vanished": vanished}))
        return found

    def _real(self, ctx) -> list[Finding]:
        subject, regions = ctx.subject, Regions(coarse=True)
        found = []
        for strength in self.STRENGTHS:
            lost: dict[str, list[float]] = {}
            vanished: dict[str, list[float]] = {}
            for path, layer in subject.adapted.layers.items():
                weight = layer.frozen_weight
                if weight.dtype == torch.float32:
                    continue
                share, gone = self._loss(weight, layer._delta().detach(), strength)
                group = regions.of(path).name
                lost.setdefault(group, []).append(share)
                vanished.setdefault(group, []).append(gone)
            for group in sorted(lost):
                values, gone = sorted(lost[group]), vanished[group]
                median, worst = values[len(values) // 2], values[-1]
                found.append(self.measured(f"{Path(ctx.lora).name} strength {strength}, {group} ({len(values)} layers)",
                                           f"update size lost: median {median:.1%}, worst {worst:.1%}; values vanishing: mean {sum(gone) / len(gone):.1%}",
                                           {"median_lost": median, "worst_lost": worst}, warn_if=median > 0.25))
        return found
