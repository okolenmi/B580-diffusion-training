"""Learning-rate schedule nodes."""

from __future__ import annotations

from typing import Any

from ...engine.step.schedule import Constant, Cosine, LRModifier, LRSchedule, StagedModifier, WarmupModifier
from ...graph import Addon, AddonSlot, Node, Port

MODIFIERS = AddonSlot("lr_modifier", LRModifier, many=True, doc="Changes the schedule in saved order (a warmup, for example).")


def modified(schedule: LRSchedule, modifiers: list[LRModifier]) -> LRSchedule:
    for modifier in modifiers:
        schedule = modifier.wrap(schedule)
    return schedule


class ConstantLRNode(Node):
    DISPLAY_NAME = "Constant LR"
    CATEGORY = "Schedule"
    INPUTS = {"lr": Port("lr", float, default=1e-4, minimum=0.0)}
    OUTPUTS = {"schedule": Port("schedule", LRSchedule)}
    ADDON_SLOTS = (MODIFIERS,)

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        return {"schedule": modified(Constant(inputs["lr"]), self.addons.all("lr_modifier"))}


class CosineLRNode(Node):
    DISPLAY_NAME = "Cosine LR"
    CATEGORY = "Schedule"
    INPUTS = {"lr": Port("lr", float, default=1e-4, minimum=0.0),
              "total_steps": Port("total_steps", int, required=False, minimum=1,
                                doc="Optimizer steps over which the rate decays; empty follows the length of the run (steps or epochs)."),
              "min_fraction": Port("min_fraction", float, required=False, default=0.05, minimum=0.0, maximum=1.0, step=0.01)}
    OUTPUTS = {"schedule": Port("schedule", LRSchedule)}
    ADDON_SLOTS = (MODIFIERS,)

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        return {"schedule": modified(Cosine(inputs["lr"], inputs.get("total_steps"), inputs.get("min_fraction", 0.05)), self.addons.all("lr_modifier"))}


class WarmupAddon(Addon):
    DISPLAY_NAME = "Warmup"
    SLOT = "lr_modifier"
    INPUTS = {"steps": Port("steps", int, default=100, minimum=0, doc="Optimizer steps (accumulation windows) spent ramping."),
              "start": Port("start", float, required=False, default=0.0, minimum=0.0, doc="Rate at the very start.")}

    def build(self, **inputs) -> Any:
        return WarmupModifier(inputs["steps"], inputs.get("start", 0.0))


class StagedRateAddon(Addon):
    DISPLAY_NAME = "Staged rate"
    SLOT = "lr_modifier"
    INPUTS = {"boundaries": Port("boundaries", str, default="", doc="Comma-separated steps where a stage ends, for example `600,1000`."),
              "ratios": Port("ratios", str, default="", doc="Comma-separated factors of the schedule's rate, one more than there are boundaries, for example `1,0.25,0.05`."),
              "warmup": Port("warmup", int, required=False, default=20, minimum=0, doc="Steps each stage after the first spends ramping up from zero.")}

    def build(self, **inputs) -> Any:
        boundaries = tuple(int(x) for x in inputs["boundaries"].split(",") if x.strip())
        ratios = tuple(float(x) for x in inputs["ratios"].split(",") if x.strip())
        return StagedModifier(boundaries, ratios, inputs.get("warmup", 20))
