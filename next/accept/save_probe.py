"""Does the saved adapter make a model compute what training computed?

Three predictions of the same inputs (seed, prompt, timestep, noise): the one the trainer's own model made just before the file was
written (``<adapter>.probe.safetensors``, see Save Adapter's probe), the one of a fresh copy of this project's model with the saved file
loaded, and the one of ComfyUI with the same file. Differences are measured against what the adapter does at all (its effect: the
prediction with the adapter minus the prediction without it), because that is the size of what a broken save would damage.
"""

from __future__ import annotations

import gc
import json
import subprocess
import time
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor

from ..families.api.adapters import TrainableSelection
from ..families.general.ldm_unet.fused import FusedGeglu
from ..families.general.adapters.schemes import read_adapter
from ..families.registry import get_family
from ..families.sdxl.adapters import inject_denoiser
from .loading import checkpoint_path, load_denoiser

PASS_COSINE, PASS_RATIO, FAIL_COSINE, FAIL_RATIO = 0.98, 0.05, 0.8, 0.3


@dataclass
class Probe:
    tensors: dict[str, Tensor]
    meta: dict

    @staticmethod
    def read(path: str) -> Probe:
        with safe_open(path, framework="pt") as handle:
            raw = handle.metadata() or {}
            tensors = {k: handle.get_tensor(k) for k in handle.keys()}
        meta = {}
        for key, value in raw.items():
            try:
                meta[key] = json.loads(value)
            except ValueError:
                meta[key] = value
        return Probe(tensors, meta)

    @property
    def cases(self) -> list[dict]:
        return self.meta["cases"]


@dataclass
class Difference:
    """How far ``got`` is from ``want``, in the units of what the adapter does (``want`` minus ``base_want``)."""

    exact: bool
    relative: float  # |got - want| / |want|
    of_effect: float  # |got - want| / |effect of want|
    effect_cosine: float
    effect_ratio: float  # |effect of got| / |effect of want|
    effect: float  # |effect of want| / |want|

    @staticmethod
    def of(got: Tensor, want: Tensor, base_got: Tensor, base_want: Tensor) -> Difference:
        err, eg, ew = (got - want).norm(), got - base_got, want - base_want
        return Difference(bool(torch.equal(got, want)), float(err / want.norm()), float(err / ew.norm().clamp(min=1e-20)),
                          float(torch.nn.functional.cosine_similarity(eg.flatten(), ew.flatten(), dim=0)),
                          float(eg.norm() / ew.norm().clamp(min=1e-20)), float(ew.norm() / want.norm()))

    def verdict(self) -> str:
        if self.exact:
            return "PASS (identical)"
        deviation = abs(self.effect_ratio - 1)
        if self.effect_cosine >= PASS_COSINE and deviation <= PASS_RATIO:
            return "PASS"
        if self.effect_cosine < FAIL_COSINE or deviation > FAIL_RATIO:
            return "FAIL"
        return "WARN"


class Comparison:
    """One row per case of one pair of predictions."""

    def __init__(self, title: str, note: str):
        self.title, self.note, self.rows = title, note, []

    def add(self, case: dict, difference: Difference, floor: float | None = None) -> None:
        """``floor``: the same engines' relative difference without the adapter, what the comparison cannot do better than."""
        self.rows.append((case, difference, floor))

    def worst(self) -> str:
        order = ["PASS (identical)", "PASS", "WARN", "FAIL"]
        return max((d.verdict() for _, d, _ in self.rows), key=order.index) if self.rows else "-"

    def markdown(self) -> str:
        floored = any(f is not None for _, _, f in self.rows)
        lines = [f"### {self.title}: {self.worst()}", "", self.note, "",
                 "| prompt | t | verdict | identical | error / prediction | error / adapter effect | effect cosine | effect size ratio | adapter effect / prediction |"
                 + (" error / the engines' difference without the adapter |" if floored else ""),
                 "|---|---|---|---|---|---|---|---|---|" + ("---|" if floored else "")]
        for case, d, floor in self.rows:
            prompt = (case["prompt"] or "(empty)")[:40]
            lines.append(f"| {prompt} | {case['t']:.0f} | {d.verdict()} | {'yes' if d.exact else 'no'} | {d.relative:.2e} | {d.of_effect:.3f} | "
                         f"{d.effect_cosine:.4f} | {d.effect_ratio:.4f} | {d.effect:.3f} |" + (f" {d.relative / floor:.2f} |" if floored and floor else ""))
        return "\n".join(lines)


class OurReload:
    """A fresh model of this project with the saved file loaded, run on the probe's inputs."""

    def __init__(self, probe: Probe, adapter: str, models: str, file: str, device: str):
        self.probe, self.adapter, self.models, self.file, self.device = probe, adapter, models, file, device
        self.report: dict = {}

    def run(self) -> dict[str, Tensor]:
        meta, family = self.probe.meta, get_family("sdxl")
        state, header = read_adapter(self.adapter)
        denoiser = load_denoiser(self.models, self.file, self.device)
        compute = meta.get("compute", "activation")
        adapted = inject_denoiser(denoiser, TrainableSelection(), rank=int(header.get("rank") or meta["rank"]), alpha=float(header.get("alpha") or meta["rank"]),
                                  kind=header.get("kind") or meta.get("kind", "lora"), compute="float32" if compute == "float32" else "activation")
        scheme = family.schemes.get(header.get("scheme") or "kohya")
        loaded = family.load_checked(scheme, adapted, state)
        self.report = {"layers in the model": len(adapted.layers), "layers loaded from the file": loaded.matched, "model layers the file lacks": len(loaded.missing),
                       "file keys no layer used": len(loaded.unexpected), "fused GEGLU (as in training)": bool(meta.get("fused_geglu"))}
        FusedGeglu.disable()
        if meta.get("fused_geglu"):
            self.report["fused GEGLU state"] = FusedGeglu.enable(torch.device(self.device)) or "on"
        out: dict[str, Tensor] = {}
        t = self.probe.tensors
        denoiser.eval()
        with torch.no_grad():
            for n in range(len(self.probe.cases)):
                args = (t["x"].to(self.device), t[f"t{n}"].to(self.device), _Conditioning(t[f"ctx{n}"].to(self.device), t[f"y{n}"].to(self.device)))
                with adapted.gate.override(None):
                    out[f"live{n}"] = denoiser.predict(*args).float().cpu()
                with adapted.gate.override(torch.zeros(1, device=self.device)):
                    out[f"base{n}"] = denoiser.predict(*args).float().cpu()
        FusedGeglu.disable()
        return out


def _relative(a: Tensor, b: Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp(min=1e-20))


@dataclass
class _Conditioning:
    ctx: Tensor
    y: Tensor


class ComfyRun:
    """ComfyUI's answers: read from ``out``, or made by running ``comfy_forward.py`` with the given Python."""

    def __init__(self, probe_path: str, adapter: str, checkpoint: str, out: str, comfy: str | None, python: str, dtype: str, device: str):
        self.probe_path, self.adapter, self.checkpoint, self.out = probe_path, adapter, checkpoint, out
        self.comfy, self.python, self.dtype, self.device = comfy, python, dtype, device

    def run(self) -> tuple[dict[str, Tensor], dict[str, str]] | None:
        if self.comfy:
            script = str(Path(__file__).with_name("comfy_forward.py"))
            command = [self.python, script, "--comfy", self.comfy, "--checkpoint", self.checkpoint, "--lora", self.adapter, "--probe", self.probe_path,
                       "--out", self.out, "--unet-dtype", self.dtype, "--device", self.device]
            print("running ComfyUI:", " ".join(command), flush=True)
            if subprocess.run(command).returncode:
                raise SystemExit("the ComfyUI run failed (see its output above)")
        if not Path(self.out).exists():
            return None
        with safe_open(self.out, framework="pt") as handle:
            return {k: handle.get_tensor(k) for k in handle.keys()}, dict(handle.metadata() or {})


class SaveFidelity:
    def __init__(self, probe_path: str, models: str, file: str, device: str, adapter: str | None = None, comfy: str | None = None,
                 comfy_python: str | None = None, comfy_dtype: str = "auto", comfy_device: str = "auto"):
        self.probe_path, self.models, self.file, self.device = probe_path, models, file, device
        self.adapter = adapter or probe_path.replace(".probe.safetensors", ".safetensors")
        self.comfy_out = probe_path.replace(".probe.safetensors", ".comfy.safetensors")
        self.comfy = ComfyRun(probe_path, self.adapter, checkpoint_path(models, file), self.comfy_out, comfy, comfy_python or sys.executable, comfy_dtype, comfy_device)

    def file_section(self, probe: Probe) -> str:
        state, header = read_adapter(self.adapter)
        factors = [t for k, t in state.items() if k.endswith((".lora_down.weight", ".lora_up.weight", "lora_A", "lora_B"))]
        bad = sum(1 for t in factors if not torch.isfinite(t.float()).all())
        zero_up = sum(1 for k, t in state.items() if k.endswith(".lora_up.weight") and not t.any())
        dtypes = sorted({str(t.dtype).replace("torch.", "") for t in factors})
        size = lambda path: f"{Path(path).stat().st_size / 1e6:.1f} MB"  # noqa: E731
        rows = [("adapter file read", f"{Path(self.adapter).resolve()} ({size(self.adapter)}, {time.strftime('%Y-%m-%d %H:%M', time.localtime(Path(self.adapter).stat().st_mtime))})"),
                ("probe file (only the recorded inputs and predictions, not the adapter)", f"{Path(self.probe_path).resolve()} ({size(self.probe_path)})"), ("file step / probe step", f"{header.get('step')} / {probe.meta.get('step')}"),
                ("probe names this file", probe.meta.get("adapter")), ("format, rank, alpha", f"{header.get('scheme')}, {header.get('rank')}, {header.get('alpha')}"),
                ("tensors", len(state)), ("factor dtypes in the file", ", ".join(dtypes)), ("tensors with NaN/inf", bad), ("layers whose up factor is all zero", zero_up),
                ("training state recorded", f"compute {probe.meta.get('compute')}, fused GEGLU {probe.meta.get('fused_geglu')}, base {probe.meta.get('base_precision')}")]
        return "\n".join(["## The files", "", "| | |", "|---|---|", *[f"| {k} | {v} |" for k, v in rows]])

    def run(self) -> str:
        probe = Probe.read(self.probe_path)
        live = {k: v for k, v in probe.tensors.items() if k.startswith(("live", "base"))}
        sections, summary = [self.file_section(probe)], []
        reload = OurReload(probe, self.adapter, self.models, self.file, self.device)
        ours = reload.run()
        self.free_device()
        sections.append("\n".join(["## This project's model, loaded from the file", "", "| | |", "|---|---|", *[f"| {k} | {v} |" for k, v in reload.report.items()]]))
        reloaded = Comparison("2. Training forward against this project's model with the saved file",
                              "Same code, same weights if the file holds what training had: the prediction should be identical.")
        for n, case in enumerate(probe.cases):
            reloaded.add(case, Difference.of(ours[f"live{n}"], live[f"live{n}"], ours[f"base{n}"], live[f"base{n}"]))
        control = Comparison("1. Control: the same comparison with the adapter switched off in the reloaded model",
                             "This must FAIL: it shows the comparison notices a model that does not carry the adapter.")
        for n, case in enumerate(probe.cases):
            control.add(case, Difference.of(ours[f"base{n}"], live[f"live{n}"], ours[f"base{n}"], live[f"base{n}"]))
        sections += [control.markdown(), reloaded.markdown()]
        summary += [("control: reloaded model without the adapter (must FAIL)", control.worst()), ("training forward = our model + saved file", reloaded.worst())]
        answer = self.comfy.run()
        if answer is None:
            sections.append("## ComfyUI\n\nNot run: give --comfy <ComfyUI folder> (and --comfy-python for ComfyUI's own Python), or put its output next to the probe.")
        else:
            data, header = answer
            sections.append("\n".join(["## ComfyUI", "", "| | |", "|---|---|", *[f"| {k} | {v} |" for k, v in header.items()]]))
            with_ours = Comparison("3a. Training forward against ComfyUI (model, LoRA loader) fed this project's conditioning",
                                   "Only the UNet and the way the file is applied differ. The difference between the two models without the adapter is "
                                   "the engines' own noise (precision, attention kernels). The last column is the error divided by that noise: near 1 means the adapter adds nothing to the difference.")
            with_own = Comparison("3b. Training forward against ComfyUI with its own text encoders and size embedding",
                                  "Everything differs that can: also tests the text side.")
            for n, case in enumerate(probe.cases):
                base_want = live[f"base{n}"]
                floor = _relative(data[f"comfy_base{n}"], base_want)
                with_ours.add(case, Difference.of(data[f"comfy_ours{n}"], live[f"live{n}"], data[f"comfy_base{n}"], base_want), floor)
                with_own.add(case, Difference.of(data[f"comfy_own{n}"], live[f"live{n}"], data[f"comfy_base{n}"], base_want), floor)
            sections += [with_ours.markdown(), with_own.markdown(), self.text_section(probe, data)]
            summary += [("training forward = ComfyUI + saved file (our conditioning)", with_ours.worst()),
                        ("training forward = ComfyUI + saved file (its own conditioning)", with_own.worst())]
            sections.append(self.floor_section(probe, data, live))
        head = ["# Save fidelity", "", "| comparison | verdict |", "|---|---|", *[f"| {k} | {v} |" for k, v in summary], "",
                f"PASS: effect cosine at least {PASS_COSINE} and effect size within {PASS_RATIO:.0%}; FAIL: cosine below {FAIL_COSINE} or size off by more than {FAIL_RATIO:.0%}; "
                "WARN in between. Identical predictions pass whatever their size.", ""]
        return "\n".join(head + [s + "\n" for s in sections])

    def free_device(self) -> None:
        """ComfyUI runs in another process and needs the card's memory."""
        gc.collect()
        backend = getattr(torch, self.device.split(":")[0], None)
        if hasattr(backend, "empty_cache"):
            backend.empty_cache()

    @staticmethod
    def text_section(probe: Probe, data: dict[str, Tensor]) -> str:
        lines = ["### Text conditioning: this project's against ComfyUI's", "", "| prompt | context error | pooled vector error | size embedding error |", "|---|---|---|---|"]
        seen = set()
        for n, case in enumerate(probe.cases):
            if case["prompt"] in seen:
                continue
            seen.add(case["prompt"])
            ctx, y = probe.tensors[f"ctx{n}"], probe.tensors[f"y{n}"]
            cc, cy = data[f"comfy_ctx{n}"], data[f"comfy_y{n}"]
            lines.append(f"| {(case['prompt'] or '(empty)')[:40]} | {_relative(ctx.float(), cc):.2e} | {_relative(y[:, :1280], cy[:, :1280]):.2e} | {_relative(y[:, 1280:], cy[:, 1280:]):.2e} |")
        return "\n".join(lines)

    @staticmethod
    def floor_section(probe: Probe, data: dict[str, Tensor], live: dict[str, Tensor]) -> str:
        lines = ["### The engines' own difference (no adapter)", "", "| prompt | t | base prediction: ComfyUI against this project's (relative) |", "|---|---|---|"]
        for n, case in enumerate(probe.cases):
            a, b = data[f"comfy_base{n}"], live[f"base{n}"]
            lines.append(f"| {(case['prompt'] or '(empty)')[:40]} | {case['t']:.0f} | {float((a - b).norm() / b.norm()):.2e} |")
        return "\n".join(lines)
