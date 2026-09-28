"""LRSchedule: pure step -> learning-rate strategies, and the nodes that build them."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import ClassVar

from ..core import Node, Port


class LRSchedule(ABC):

    @abstractmethod
    def value(self, step: int) -> float:
        ...


class ConstantLRSchedule(LRSchedule):

    def __init__(self, lr: float):
        self.lr = lr

    def value(self, step: int) -> float:
        return self.lr


class CosineLRSchedule(LRSchedule):

    def __init__(self, lr: float, total_steps: int, lr_min_frac: float = 0.05):
        self.lr = lr
        self.total_steps = max(total_steps, 1)
        self.lr_min = lr * lr_min_frac

    def value(self, step: int) -> float:
        p = min(step, self.total_steps - 1) / max(self.total_steps - 1, 1)
        return self.lr_min + 0.5 * (self.lr - self.lr_min) * (1 + math.cos(math.pi * p))


class WarmupLRSchedule(LRSchedule):
    """Linear warmup *around* any other schedule, not a schedule of its own.

    For step < warmup_steps, lerps from `warmup_start` toward the wrapped
    schedule's own value **at that same step** (so wrapping a cosine tracks
    the cosine's decay during warmup instead of freezing it); from
    warmup_steps on, returns the wrapped schedule unchanged. The ramp uses
    frac = (step + 1) / warmup_steps: step 0 starts one ramp-step *above*
    `warmup_start` (not exactly on it), and frac reaches exactly 1.0 on
    the last warmup step, so the join at step == warmup_steps is
    continuous with the wrapped schedule rather than jumping the
    remaining distance.

    Why it exists: a cold optimizer (Adam's first/second moments still at
    zero) taking full-LR normalized updates from step 0 is exactly the
    "everything breaks in the first steps and then clamps itself back
    together" shape -- the legacy loop had a 200-step warmup
    (core/config_model.py's lr_warmup_steps) that the nodes route had no
    way to express before this class."""

    def __init__(self, inner: LRSchedule, warmup_steps: int, warmup_start: float = 0.0):
        self.inner = inner
        self.warmup_steps = max(int(warmup_steps), 0)
        self.warmup_start = float(warmup_start)

    def value(self, step: int) -> float:
        target = self.inner.value(step)
        if self.warmup_steps <= 0 or step >= self.warmup_steps:
            return target
        frac = (step + 1) / self.warmup_steps
        return self.warmup_start + (target - self.warmup_start) * frac


class LRScheduleNode(Node):

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "schedule": Port(name="schedule", type=LRSchedule, required=True),
    }

    @abstractmethod
    def build(self, **inputs) -> dict[str, LRSchedule]:
        ...


class ConstantLRScheduleNode(LRScheduleNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        "lr": Port(name="lr", type=float, required=True),
    }

    def build(self, **inputs) -> dict[str, LRSchedule]:
        self.validate_inputs(inputs)
        result = {"schedule": ConstantLRSchedule(lr=inputs["lr"])}
        self.validate_outputs(result)
        return result


class CosineLRScheduleNode(LRScheduleNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        "lr": Port(name="lr", type=float, required=True),
        "total_steps": Port(name="total_steps", type=int, required=True),
        "lr_min_frac": Port(name="lr_min_frac", type=float, required=False, default=0.05),
    }

    def build(self, **inputs) -> dict[str, LRSchedule]:
        self.validate_inputs(inputs)
        result = {"schedule": CosineLRSchedule(
            lr=inputs["lr"],
            total_steps=inputs["total_steps"],
            lr_min_frac=inputs.get("lr_min_frac", self.INPUTS["lr_min_frac"].default),
        )}
        self.validate_outputs(result)
        return result


class WarmupLRScheduleNode(LRScheduleNode):
    """Wrap any other schedule node's output in a linear warmup ramp.

    Wire a ConstantLRScheduleNode/CosineLRScheduleNode's `schedule` output
    into `schedule` here, and this node's output into the trainer -- the
    warmup applies on top, route-agnostic (both trainers take an
    LRSchedule). See WarmupLRSchedule's own docstring for the exact ramp
    math and why the loop it serves needed one."""

    INPUTS: ClassVar[dict[str, Port]] = {
        "schedule": Port(
            name="schedule", type=LRSchedule, required=True,
            doc="The schedule to warm up into (Constant, Cosine, ...). The ramp targets "
                "this schedule's own value at each step, so wrapping a cosine tracks its "
                "decay during warmup instead of holding a constant target.",
        ),
        "warmup_steps": Port(
            name="warmup_steps", type=int, required=True,
            doc="Optimizer steps spent ramping. The legacy loop used 200 "
                "(core/config_model.py's lr_warmup_steps); on the managed route one "
                "optimizer step = one grad_accum window, so this counts windows, not "
                "batches.",
        ),
        "warmup_start": Port(
            name="warmup_start", type=float, required=False, default=0.0,
            doc="LR at the very start of the ramp. 0.0 default -- the first warmup step "
                "then sits at (target - 0) / warmup_steps, effectively a cold start. The "
                "legacy loop used 1e-8 (lr_warmup_start); any tiny positive value works "
                "too if a strictly-positive LR matters for a specific optimizer.",
        ),
    }

    def build(self, **inputs) -> dict[str, LRSchedule]:
        self.validate_inputs(inputs)
        warmup_steps = inputs["warmup_steps"]
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        result = {"schedule": WarmupLRSchedule(
            inner=inputs["schedule"],
            warmup_steps=warmup_steps,
            warmup_start=inputs.get("warmup_start", self.INPUTS["warmup_start"].default),
        )}
        self.validate_outputs(result)
        return result
