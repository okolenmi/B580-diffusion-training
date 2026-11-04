"""Addons of the Noiser and the Batcher: how the first batches of a run differ from the rest."""

from __future__ import annotations

from typing import Any

from ...data.ladder import RegionLadder
from ...data.ramp import BatchRamp
from ...graph import Addon, Port


class RegionLadderAddon(Addon):
    DISPLAY_NAME = "Region Ladder"
    SLOT = "ladder"
    INPUTS = {"start": Port("start", int, default=9, minimum=1, doc="Timesteps per image in the first batch."),
              "span": Port("span", int, required=False, default=120, minimum=1, unit="batches",
                           doc="Batches over which the count falls in equal parts to the Noiser's `regions`."),
              "skip": Port("skip", int, required=False, default=0, minimum=0, unit="batches",
                           doc="Batches already trained, when training is continued: the ladder then starts that far in (past its span: no extra timesteps).")}

    def build(self, **inputs) -> Any:
        """More timesteps per image make the first updates, which set the direction the adapter takes, less dependent on one random
        draw; with a gradient policy every extra region costs a full pass."""
        return RegionLadder(inputs["start"], inputs.get("span") or 120, inputs.get("skip") or 0)


class BatchRampAddon(Addon):
    DISPLAY_NAME = "Batch Ramp"
    SLOT = "ramp"
    INPUTS = {"size": Port("size", int, default=1, minimum=1, doc="Samples per batch in the first batches."),
              "batches": Port("batches", int, required=False, default=200, minimum=1, unit="batches",
                              doc="How many batches of the run use the smaller size; after them the Batcher's own batch size applies."),
              "skip": Port("skip", int, required=False, default=0, minimum=0, unit="batches",
                           doc="Batches already trained, when training is continued: the small batches then end that much sooner.")}

    def build(self, **inputs) -> Any:
        """A small batch at the start gives each update one picture's direction (more variety), a large one at the end averages
        several pictures (steadier, more detail)."""
        return BatchRamp(inputs["size"], inputs.get("batches") or 200, inputs.get("skip") or 0)
