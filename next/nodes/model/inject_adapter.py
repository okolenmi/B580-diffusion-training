"""Inject Adapter: load a denoiser and wrap the chosen layers with trainable adapters."""

from __future__ import annotations

import re
from typing import Any

import torch
from torch import nn

from ...engine.devices import resolve_device, resolve_dtype
from ...engine.handles import TrainableDenoiser
from ...engine.runtime.checkpointing import ActivationCheckpointer
from ...families.api.adapters import GateWindow, TrainableSelection
from ...families.api.parts import AdapterMerge, DenoiserPart
from ...families.registry import get_family
from ...graph import Group, Node, Port

SCALING = ("classic", "rank_stabilized")


def parse_regions(text: str) -> dict[str, float | None]:
    """``prefix=weight`` per line or comma; ``prefix=off`` turns a region off."""
    out: dict[str, float | None] = {}
    for item in re.split(r"[\n,]", text or ""):
        if not item.strip():
            continue
        prefix, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"region {item.strip()!r} needs the form prefix=weight or prefix=off")
        out[prefix.strip()] = None if value.strip().lower() == "off" else float(value)
    return out


def parse_list(text: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in re.split(r"[\n,]", text or "") if p.strip())


class InjectAdapterNode(Node):
    DISPLAY_NAME = "Inject Adapter"
    CATEGORY = "Model"
    READS_SETTINGS = ("device.default", "precision.denoiser")
    GROUPS = (Group("adapter", "Adapter"), Group("targets", "Targets", collapsed=True),
              Group("gate", "Timestep gate", collapsed=True, advanced=True),
              Group("memory", "Memory", collapsed=True, advanced=True))
    INPUTS = {
        "denoiser": Port("denoiser", DenoiserPart),
        "kind": Port("kind", str, required=False, default="lora", choices=("lora", "dora"), group="adapter"),
        "rank": Port("rank", int, required=False, default=16, minimum=1, group="adapter"),
        "alpha": Port("alpha", float, required=False, default=16.0, minimum=0.0, group="adapter"),
        "dropout": Port("dropout", float, required=False, default=0.0, minimum=0.0, maximum=0.9, step=0.01, group="adapter"),
        "scaling": Port("scaling", str, required=False, default="classic", choices=SCALING, group="adapter",
                        doc="classic: alpha / rank. rank_stabilized: alpha / sqrt(rank)."),
        "groups": Port("groups", str, required=False, default="", group="targets",
                       doc="Comma-separated module groups to train; empty uses the family's defaults."),
        "regions": Port("regions", str, required=False, default="", group="targets",
                        doc="prefix=weight or prefix=off, one per line: scale or switch off parts of the network."),
        "include": Port("include", str, required=False, default="", group="targets",
                        doc="Regular expressions (full match on the module path) for extra modules to train."),
        "exclude": Port("exclude", str, required=False, default="", group="targets",
                        doc="Regular expressions for modules to leave out; exclusion wins."),
        "gate": Port("gate", bool, required=False, default=False, group="gate",
                     doc="Fade the adapter out for timesteps outside the range below."),
        "gate_low": Port("gate_low", float, required=False, default=0.0, group="gate", visible_when=("gate", True)),
        "gate_high": Port("gate_high", float, required=False, default=999.0, group="gate", visible_when=("gate", True)),
        "gate_width": Port("gate_width", float, required=False, default=100.0, group="gate", visible_when=("gate", True)),
        "checkpointing": Port("checkpointing", bool, required=False, default=True, group="memory",
                              doc="Recompute activations in backward instead of storing them."),
        "attention_fraction": Port("attention_fraction", float, required=False, default=1.0, minimum=0.0, maximum=1.0,
                                   step=0.05, group="memory", visible_when=("checkpointing", True),
                                   doc="Share of the transformer blocks that are checkpointed (residual blocks always are)."),
    }
    OUTPUTS = {"model": Port("model", TrainableDenoiser, doc="Denoiser with adapters; only the adapters train.")}

    def _merge(self, family, denoiser, merge: AdapterMerge) -> None:
        scheme = family.schemes.get(merge.format)
        if scheme is None or not hasattr(scheme, "key"):
            raise ValueError(f"Merge Adapter: {family.id} cannot merge the {merge.format!r} format; choose from kohya")
        path = self.context.env.service("models").resolve("loras", merge.file)
        state, _ = family.read_adapter(str(path))
        report = family.merge_adapter(denoiser.module(), scheme, state, merge.strength)
        if report.merged == 0:
            raise ValueError(f"Merge Adapter: nothing in {merge.file} matches this network (wrong family or key format?)")

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        part: DenoiserPart = inputs["denoiser"]
        family = get_family(part.family_id)
        device = resolve_device(self.context.settings.get("device.default"))
        precision = part.dtype if part.dtype != "auto" else self.context.settings.get("precision.denoiser")
        dtype = resolve_dtype(precision, torch.bfloat16)
        selection = TrainableSelection(groups=parse_list(inputs.get("groups", "")) or None,
                                       regions=parse_regions(inputs.get("regions", "")),
                                       include=parse_list(inputs.get("include", "")),
                                       exclude=parse_list(inputs.get("exclude", "")))
        rank, alpha = inputs.get("rank") or 16, float(inputs.get("alpha") if inputs.get("alpha") is not None else 16.0)
        with self.context.task("Loading the denoiser"):
            denoiser = family.load_denoiser(part, device=device, dtype=dtype)
        for merge in part.merges:
            with self.context.task(f"Merging {merge.file}"):
                self._merge(family, denoiser, merge)
        with self.context.task("Injecting the adapter"):
            adapted = family.inject_denoiser(denoiser, selection, rank=rank, alpha=alpha, kind=inputs.get("kind") or "lora",
                                             policy=inputs.get("scaling") or "classic", dropout=inputs.get("dropout") or 0.0)
        deterministic = not any(isinstance(m, nn.Dropout) and m.p > 0 for m in denoiser.module().modules())
        checkpointer = ActivationCheckpointer(preserve_rng=not deterministic) if inputs.get("checkpointing", True) else None
        share = inputs.get("attention_fraction", 1.0)
        if checkpointer is not None:
            denoiser.set_checkpointer(checkpointer, share)
        window = None
        if inputs.get("gate"):
            window = GateWindow(inputs.get("gate_low", 0.0), inputs.get("gate_high", 999.0), inputs.get("gate_width", 100.0))
        plan = adapted.plan.to_json()
        return {"model": TrainableDenoiser(family, part, denoiser, adapted, device, rank, alpha, inputs.get("kind") or "lora",
                                           selection, window, info=plan,
                                           checkpointer=checkpointer, checkpoint_share=share, deterministic=deterministic)}
