"""Learning-rate schedules: a value per optimizer step."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class LRSchedule(ABC):
    @abstractmethod
    def value(self, step: int) -> float: ...

    def fit(self, total_steps: int) -> None:
        """Told the length of the run when it was not set explicitly (a schedule that decays over the run uses it)."""

    def summary(self, steps: int) -> dict:
        """Peak and mean over ``steps`` (sampled at up to 1001 points), to show what the run will really apply."""
        n = min(max(steps, 1), 1001)
        values = [self.value(round(i * (steps - 1) / max(n - 1, 1))) for i in range(n)]
        return {"kind": type(self).__name__, "peak": max(values), "mean": sum(values) / len(values), "steps": steps}


class Constant(LRSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def value(self, step: int) -> float:
        return self.lr


class Cosine(LRSchedule):
    def __init__(self, lr: float, total_steps: int | None, min_fraction: float = 0.05):
        self.lr, self.total, self.follows_run = lr, max(total_steps or 1, 1), total_steps is None
        self.floor = lr * min_fraction

    def fit(self, total_steps: int) -> None:
        if self.follows_run:
            self.total = max(total_steps, 1)

    def value(self, step: int) -> float:
        progress = min(step, self.total - 1) / max(self.total - 1, 1)
        return self.floor + 0.5 * (self.lr - self.floor) * (1 + math.cos(math.pi * progress))


class Warmup(LRSchedule):
    """Ramp linearly from ``start`` to the wrapped schedule's own value over ``steps``, tracking its decay meanwhile."""

    def __init__(self, inner: LRSchedule, steps: int, start: float = 0.0):
        if steps < 0:
            raise ValueError(f"warmup steps must not be negative, got {steps}")
        self.inner, self.steps, self.start = inner, steps, start

    def fit(self, total_steps: int) -> None:
        self.inner.fit(total_steps)

    def value(self, step: int) -> float:
        target = self.inner.value(step)
        if self.steps <= 0 or step >= self.steps:
            return target
        return self.start + (target - self.start) * (step + 1) / self.steps
