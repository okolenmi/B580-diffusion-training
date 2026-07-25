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
