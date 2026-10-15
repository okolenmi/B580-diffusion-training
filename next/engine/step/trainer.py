"""The training loop: optimizer steps made of ``accumulation`` micro-steps."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

from ...graph.events import StepFinished
from ..optim.handle import FusedOptimizer, Optimizer
from .loss import LossWeighting, thirds_of_range
from .phases import (Backward, EncodeConditioning, FetchBatch, Forward, GateWindow, Loss, Pipeline, PrepareInputs, Timed)
from .schedule import LRSchedule
from .state import StepState


@dataclass
class RunResult:
    steps_done: int
    total: int
    last_loss: float | None
    stopped: bool = False


class Trainer:
    """Runs optimizer steps over a batch source. Everything it needs arrives in the constructor; it owns none of it.

    ``emit`` receives a ``StepFinished`` after every optimizer step; ``should_stop`` is polled between steps, so
    stopping always leaves the adapters at a step boundary.
    """

    def __init__(self, *, batches: Iterable[dict], denoiser, conditioner, optimizer: Optimizer, schedule: LRSchedule,
                 weighting: LossWeighting, steps: int, accumulation: int = 1, device="cpu", gate=None,
                 gate_window: GateWindow | None = None, emit: Callable[[StepFinished], None] | None = None,
                 should_stop: Callable[[], bool] | None = None, synchronize: Callable[[], None] | None = None,
                 timed: bool = False):
        if accumulation < 1 or steps < 0:
            raise ValueError("steps must not be negative and accumulation must be at least 1")
        self.optimizer, self.schedule, self.steps, self.accumulation = optimizer, schedule, steps, accumulation
        self.device, self.emit, self.should_stop = device, emit, should_stop
        self._fused = isinstance(optimizer, FusedOptimizer)
        phases = [FetchBatch(batches), PrepareInputs(gate, gate_window), EncodeConditioning(conditioner),
                  Forward(denoiser), Loss(weighting), Backward(accumulation)]
        if timed:
            phases = [Timed(p, synchronize or (lambda: None)) for p in phases]
        self.pipeline = Pipeline(phases)

    def run(self, start: int = 0) -> RunResult:
        last = None
        step = start
        for step in range(start, self.steps):
            if self.should_stop is not None and self.should_stop():
                return RunResult(step - start, self.steps, last, stopped=True)
            last = self._step(step)
        return RunResult(max(self.steps - start, 0), self.steps, last)

    def _step(self, step: int) -> float:
        started = time.perf_counter()
        lr = self.schedule.value(step)
        self.optimizer.set_lr(lr)
        if self._fused:
            self.optimizer.begin_step(self.accumulation)
        else:
            self.optimizer.zero_grad()
        losses, thirds, timing = [], [], {}
        for micro in range(self.accumulation):
            state = self.pipeline.run(StepState(step, micro, self.device))
            losses.append(state.loss.detach())
            thirds.append(thirds_of_range(state.per_sample, state.inputs.t_frac))
            for label, ms in state.timing_ms.items():
                timing[label] = timing.get(label, 0.0) + ms
            if self._fused and micro + 1 < self.accumulation:
                self.optimizer.next_pass()
        if not self._fused:
            self.optimizer.step()
        loss = float(sum(losses) / len(losses))
        metrics = {"loss": loss, "lr": lr, "step_ms": (time.perf_counter() - started) * 1000, **timing}
        for name in {n for t in thirds for n in t}:
            values = [t[name] for t in thirds if name in t]
            metrics[name] = sum(values) / len(values)
        if self.emit is not None:
            self.emit(StepFinished(step + 1, self.steps, metrics))
        return loss
