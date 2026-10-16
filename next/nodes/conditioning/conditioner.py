"""Conditioner: text encoders plus a cache of their output, wired to the trainer."""

from __future__ import annotations

from typing import Any

import torch

from ...data.conditioning_cache import ConditioningCache
from ...data.prompts import PromptTable
from ...engine.conditioner import Conditioner
from ...engine.devices import resolve_device, resolve_dtype
from ...engine.runtime.residents import ModulesResident
from ...families.api.parts import ModelPart
from ...families.registry import get_family
from ...graph import EventIn, Group, Node, Port
from ...graph.events import PromptsChanged


class ConditionerNode(Node):
    DISPLAY_NAME = "Conditioner"
    CATEGORY = "Conditioning"
    READS_SETTINGS = ("device.default", "precision.encoders", "precision.autocast")
    GROUPS = (Group("cache", "Cache", collapsed=True),)
    INPUTS = {
        "encoders": Port("encoders", ModelPart, multi=True, doc="The text-encoder parts of the model family (wire each)."),
        "prompts": Port("prompts", PromptTable, required=False,
                        doc="When wired, every prompt of the coming epoch is encoded as soon as the epoch's prompts are known."),
        "release_encoders": Port("release_encoders", bool, required=False, default=True, group="cache",
                                 doc="Move the encoders off the device once the whole prompt set is cached."),
        "cache_mb": Port("cache_mb", int, required=False, default=4096, minimum=64, unit="MB", group="cache"),
    }
    OUTPUTS = {"conditioner": Port("conditioner", Conditioner)}
    EVENTS_IN = {"prompts_changed": EventIn(PromptsChanged, handler="on_prompts_changed")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        parts = {p.role: p for p in inputs["encoders"]}
        families = {p.family_id for p in parts.values()}
        if len(families) != 1:
            raise ValueError(f"Conditioner: the encoders come from different families ({sorted(families)})")
        family = get_family(families.pop())
        device = resolve_device(self.context.settings.get("device.default"))
        dtype = resolve_dtype(self.context.settings.get("precision.encoders"), torch.float16)
        out = resolve_dtype(self.context.settings.get("precision.autocast"), torch.bfloat16) if self.context.settings.get("precision.autocast") != "off" else torch.float32
        self.encoder = family.load_text_encoder(parts, device=device, dtype=dtype, out_dtype=out)
        residency = getattr(self.context.runtime, "residency", None)
        if residency is not None and hasattr(self.encoder, "modules"):
            residency.register(f"encoder {self.uid}", ModulesResident(self.encoder.modules(), self.encoder.to, device), rank=0)
        cache = ConditioningCache(family.conditioning_kit(self.encoder), max_bytes=(inputs.get("cache_mb") or 4096) << 20)
        self.conditioner = Conditioner(cache, device, out)
        self.prompts, self.release = inputs.get("prompts"), inputs.get("release_encoders", True)
        return {"conditioner": self.conditioner}

    def on_prompts_changed(self, event: PromptsChanged, ctx) -> None:
        if self.prompts is None:
            return
        universe = self.prompts.universe()
        self.conditioner.on_prompts_changed(event, universe)
        if self.release and self.conditioner.cache.fits(len(universe)):
            self.encoder.to("cpu")
