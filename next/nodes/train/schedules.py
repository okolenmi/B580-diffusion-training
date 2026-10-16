"""Learning-rate schedule nodes."""

from __future__ import annotations

from typing import Any

from ...engine.step.schedule import Constant, Cosine, LRSchedule, Warmup
from ...graph import Node, Port


class ConstantLRNode(Node):
    DISPLAY_NAME = "Constant LR"
    CATEGORY = "Schedule"
    INPUTS = {"lr": Port("lr", float, default=1e-4, minimum=0.0)}
    OUTPUTS = {"schedule": Port("schedule", LRSchedule)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        return {"schedule": Constant(inputs["lr"])}


class CosineLRNode(Node):
    DISPLAY_NAME = "Cosine LR"
    CATEGORY = "Schedule"
    INPUTS = {"lr": Port("lr", float, default=1e-4, minimum=0.0),
              "total_steps": Port("total_steps", int, minimum=1, doc="Optimizer steps over which the rate decays."),
              "min_fraction": Port("min_fraction", float, required=False, default=0.05, minimum=0.0, maximum=1.0, step=0.01)}
    OUTPUTS = {"schedule": Port("schedule", LRSchedule)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        return {"schedule": Cosine(inputs["lr"], inputs["total_steps"], inputs.get("min_fraction", 0.05))}


class WarmupLRNode(Node):
    DISPLAY_NAME = "Warmup LR"
    CATEGORY = "Schedule"
    INPUTS = {"schedule": Port("schedule", LRSchedule, doc="Ramps into this schedule's own value at each step."),
              "steps": Port("steps", int, default=100, minimum=0, doc="Optimizer steps (accumulation windows) spent ramping."),
              "start": Port("start", float, required=False, default=0.0, minimum=0.0, doc="Rate at the very start.")}
    OUTPUTS = {"schedule": Port("schedule", LRSchedule)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        return {"schedule": Warmup(inputs["schedule"], inputs["steps"], inputs.get("start", 0.0))}
