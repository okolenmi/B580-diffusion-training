"""Prompt nodes: start a prompt table from the captions, then append stages to it."""

from __future__ import annotations

from typing import Any

from ...data.library import CHANNELS
from ...data.prompts import CaptionDropout, CaptionPool, PromptTable, TagShuffle, TagSwap
from ...graph import Node, Port
from .handles import ImageSet


class CaptionsNode(Node):
    """Base captions of one channel, the start of every prompt chain."""

    DISPLAY_NAME = "Captions"
    CATEGORY = "Dataset"
    READS_SETTINGS = ("seed.base",)
    INPUTS = {
        "images": Port("images", ImageSet),
        "channel": Port("channel", str, required=False, default="text", choices=CHANNELS),
        "seed": Port("seed", int, required=False, minimum=0,
                     doc="Seeds every stage after this node; the same seed gives the same prompts in every epoch. Empty derives it from the graph's base seed."),
    }
    OUTPUTS = {"prompts": Port("prompts", PromptTable)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        return {"prompts": PromptTable.from_images(inputs["images"], inputs.get("channel") or "text",
                                                   seed=self.derived_seed(inputs.get("seed")))}


class _StageNode(Node):
    """A node that returns its input table with one stage appended."""

    INPUTS: dict[str, Port] = {}
    OUTPUTS = {"prompts": Port("prompts", PromptTable)}
    CATEGORY = "Dataset"

    def stage(self, inputs: dict):
        raise NotImplementedError

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        return {"prompts": inputs["prompts"].with_stage(self.stage(inputs))}


class CaptionPoolNode(_StageNode):
    DISPLAY_NAME = "Caption Pool"
    INPUTS = {
        "prompts": Port("prompts", PromptTable),
        "mode": Port("mode", str, required=False, default="captions", choices=("captions", "parts")),
        "n_min": Port("n_min", int, required=False, default=1, minimum=1),
        "n_max": Port("n_max", int, required=False, default=3, minimum=1),
        "include_own": Port("include_own", bool, required=False, default=True),
        "dividers": Port("dividers", str, required=False, default=".,", advanced=True),
    }

    def stage(self, inputs: dict):
        return CaptionPool(inputs.get("n_min") or 1, inputs.get("n_max") or 3, inputs.get("mode") or "captions",
                           inputs.get("include_own", True), inputs.get("dividers") or ".,")


class CaptionDropoutNode(_StageNode):
    DISPLAY_NAME = "Caption Dropout"
    INPUTS = {
        "prompts": Port("prompts", PromptTable),
        "p": Port("p", float, required=False, default=0.1, minimum=0.0, maximum=1.0, step=0.01),
        "mode": Port("mode", str, required=False, default="substitute", choices=("substitute", "empty")),
    }

    def stage(self, inputs: dict):
        return CaptionDropout(inputs.get("p", 0.1), inputs.get("mode") or "substitute")


class TagSwapNode(_StageNode):
    DISPLAY_NAME = "Tag Swap"
    INPUTS = {
        "prompts": Port("prompts", PromptTable),
        "mode": Port("mode", str, required=False, default="random", choices=("random", "walk"),
                     doc="random: a new tag each epoch. walk: the tags in order, one per epoch."),
        "keep_first": Port("keep_first", int, required=False, default=0, minimum=0,
                           doc="Leading tags (a trigger word, say) that always stay."),
    }

    def stage(self, inputs: dict):
        return TagSwap(inputs.get("mode") or "random", inputs.get("keep_first") or 0)


class TagShuffleNode(_StageNode):
    DISPLAY_NAME = "Tag Shuffle"
    INPUTS = {
        "prompts": Port("prompts", PromptTable),
        "keep_first": Port("keep_first", int, required=False, default=0, minimum=0),
        "drop": Port("drop", float, required=False, default=0.0, minimum=0.0, maximum=0.99, step=0.01,
                     doc="Chance of dropping each tag (the first tags stay)."),
    }

    def stage(self, inputs: dict):
        return TagShuffle(inputs.get("keep_first") or 0, inputs.get("drop") or 0.0)
