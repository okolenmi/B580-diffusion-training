"""Blend Conditioning: smooth text-conditioning variation between a caption and a partner prompt."""

from __future__ import annotations

from typing import Any

import torch

from ...engine.conditioner import BlendedConditioner, Conditioner
from ...graph import Group, Node, Port


class BlendConditioningNode(Node):
    DISPLAY_NAME = "Blend Conditioning"
    CATEGORY = "Conditioning"
    READS_SETTINGS = ("seed.base",)
    GROUPS = (Group("blend", "Blend"), Group("more", "More", collapsed=True, advanced=True))
    INPUTS = {
        "conditioner": Port("conditioner", Conditioner),
        "strength_min": Port("strength_min", float, required=False, default=0.0, minimum=0.0, maximum=1.0, step=0.05, group="blend",
                             doc="Each blended sample gets a strength drawn uniformly from min..max: the share of the partner prompt's "
                                 "conditioning (0 = the caption alone, 1 = the partner alone, 0.5 = the midpoint)."),
        "strength_max": Port("strength_max", float, required=False, default=0.0, minimum=0.0, maximum=1.0, step=0.05, group="blend",
                             doc="Upper end of the strength. Both ends at 0 (the default) change nothing at all."),
        "probability": Port("probability", float, required=False, default=1.0, minimum=0.0, maximum=1.0, step=0.05, group="blend",
                            doc="Chance that a sample is blended at all; the others keep their caption exactly."),
        "partners": Port("partners", str, required=False, default="", group="more",
                         doc="Partner prompts, one per line, one picked at random per sample. Empty means the empty prompt."),
        "segments": Port("segments", str, required=False, default="", group="more",
                         doc="Comma-separated parts of the conditioning to blend (family names, SDXL: clip_l, clip_g, pooled, size); "
                             "empty blends all of them."),
        "seed": Port("seed", int, required=False, minimum=0, group="more",
                     doc="Seed of the draws; empty derives it from the graph's base seed."),
    }
    OUTPUTS = {"conditioner": Port("conditioner", Conditioner, doc="Wire to the trainer in place of the plain conditioner.")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        inner: Conditioner = inputs["conditioner"]
        low, high, chance = inputs.get("strength_min") or 0.0, inputs.get("strength_max") or 0.0, inputs.get("probability")
        chance = 1.0 if chance is None else chance
        if high == 0.0 or chance == 0.0:
            return {"conditioner": inner}
        lines = [p for p in (inputs.get("partners") or "").splitlines() if p.strip()]
        segments = [s.strip() for s in (inputs.get("segments") or "").split(",") if s.strip()] or None
        generator = torch.Generator().manual_seed(self.derived_seed(inputs.get("seed")))
        return {"conditioner": BlendedConditioner(inner, lines or [""], (low, high), chance, segments, generator)}
