"""Conditioner: text encoders plus a cache of their output, wired to the trainer."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from ...data.conditioning_cache import ConditioningCache
from ...data.conditioning_store import ConditioningStore
from ...data.ids import weights_identity
from ...data.prompts import PromptTable
from ...engine.conditioner import Conditioner
from ...engine.devices import Accelerator, resolve_device, resolve_dtype
from ...engine.runtime.encoders import OnDemandKit
from ...engine.runtime.residents import ModulesResident
from ...families.api.header import read_header
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
                                 doc="Keep the encoders on the host and bring them to the device only to encode (wire prompts so the whole set is encoded up front)."),
        "persist": Port("persist", bool, required=False, default=False, group="cache",
                        doc="Keep the encodings of stable prompts (captions of the set and the few variants of dropout) on disk next to the "
                            "dataset, so a later run neither loads the encoders nor encodes. Fleeting prompts (shuffles, pools) are never written."),
        "dataset": Port("dataset", str, required=False, picker="dataset", group="cache", visible_when=("persist", True),
                        doc="The dataset whose folder holds the cache."),
        "max_disk_mb": Port("max_disk_mb", int, required=False, default=4096, minimum=16, unit="MB", group="cache",
                            visible_when=("persist", True), doc="The store stops writing here; the run carries on from memory."),
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
        self.encoder = family.load_text_encoder(parts, device="cpu", dtype=dtype, out_dtype=out)
        accelerator = Accelerator(device)
        residency = getattr(self.context.runtime, "residency", None)
        resident = ModulesResident(self.encoder.modules(), self.encoder.to, device)
        if residency is not None:
            residency.ensure_started()
            residency.claim(f"encoders {self.uid}", resident, pinned=True, purpose="text encoders; on the host except while prompts are encoded")
        self.kit = OnDemandKit(family.conditioning_kit(self.encoder), resident, accelerator, residency)
        store = None
        if inputs.get("persist"):
            if not inputs.get("dataset"):
                raise ValueError("Conditioner: persist needs the dataset whose folder holds the cache")
            store = ConditioningStore(self.context.env.service("datasets").caches(inputs["dataset"]) / "conditioning",
                                      self._identity(parts, dtype, out), self.kit, max_bytes=(inputs.get("max_disk_mb") or 4096) << 20)
        cache = ConditioningCache(self.kit, max_bytes=(inputs.get("cache_mb") or 4096) << 20, store=store)
        self.conditioner = Conditioner(cache, device, out)
        self.prompts, self.release = inputs.get("prompts"), inputs.get("release_encoders", True)
        self.kit.hold = not self.release
        return {"conditioner": self.conditioner}

    @staticmethod
    def _identity(parts: dict[str, ModelPart], dtype: torch.dtype, out: torch.dtype) -> str:
        """What the stored encodings depend on: the encoder weights, how they are run, and the kit that arranges them."""
        spec = {"dtype": str(dtype), "out": str(out)}
        return hashlib.sha256("".join(weights_identity(read_header(p.source.path), p.source.prefix, {**spec, "role": role})
                                      for role, p in sorted(parts.items())).encode()).hexdigest()[:16]

    def on_prompts_changed(self, event: PromptsChanged, ctx) -> None:
        if self.prompts is None:
            return
        universe = self.prompts.universe()
        self.kit.hold = not self.release or not self.conditioner.cache.fits(len(universe))
        with self.kit.session(), self.context.task("Encoding prompts", total=len(universe)):
            self.conditioner.on_prompts_changed(event, universe)
        cache = self.conditioner.cache
        if cache.store is not None:
            self.context.report("conditioning_cache", {"node": self.uid, "prompts": len(universe), "from_disk": cache.loaded,
                                                       "on_disk_mb": round(cache.store.nbytes / 2**20, 1), "disk_full": cache.store.full})
