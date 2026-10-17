"""The training loop: optimizer steps made of ``accumulation`` micro-steps."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from ...graph.events import StepFinished
from ..optim.handle import FusedOptimizer, Optimizer
from ..runtime.memory import MemoryBudgetExceeded, is_out_of_memory
from .loss import LossWeighting, thirds_of_range
from .phases import (Backward, EncodeConditioning, FetchBatch, Forward, Loss, Pipeline, PrepareInputs, Timed)
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

    A step that runs out of memory is run again once on the same batches after the memory it held is given back,
    as long as no update has been applied yet: a new shape asks the driver for kernels, which can fail while the
    allocator still caches blocks of the old shapes.
    """

    OOM_RETRIES = 1

    def __init__(self, *, batches: Iterable[dict], denoiser, conditioner, optimizer: Optimizer, schedule: LRSchedule,
                 weighting: LossWeighting, steps: int, accumulation: int = 1, device="cpu", gate=None,
                 gate_window=None, emit: Callable[[StepFinished], None] | None = None,
                 should_stop: Callable[[], bool] | None = None, synchronize: Callable[[], None] | None = None,
                 timed: bool = False, residency=None):
        if accumulation < 1 or steps < 0:
            raise ValueError("steps must not be negative and accumulation must be at least 1")
        self.optimizer, self.schedule, self.steps, self.accumulation = optimizer, schedule, steps, accumulation
        self.device, self.emit, self.should_stop, self.residency = device, emit, should_stop, residency
        self._fused = isinstance(optimizer, FusedOptimizer)
        self._fetch = FetchBatch(batches)
        phases = [self._fetch, PrepareInputs(gate, gate_window), EncodeConditioning(conditioner),
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
        if self.residency is not None:
            self.residency.before_step(step)
        self.optimizer.set_lr(lr)
        self._fetch.begin_window()
        for attempt in range(self.OOM_RETRIES + 1):
            self._committed = False
            failure = None
            try:
                losses, thirds, timing = self._window(step)
                break
            except Exception as error:  # noqa: BLE001 - narrowed just below
                if self._committed or attempt == self.OOM_RETRIES or not is_out_of_memory(error):
                    explained = self._explained(step, error)
                    if explained is error:
                        raise
                    raise explained from error
                failure = error
            self._recover(step, failure)
        loss = float(sum(losses) / len(losses))
        if self.residency is not None:
            self.residency.after_step(step)
        metrics = {"loss": loss, "lr": lr, "step_ms": (time.perf_counter() - started) * 1000, **timing}
        if self.residency is not None and self.residency.last_used_mb is not None:
            stats = self.residency.last_stats
            metrics.update(device_mb=self.residency.last_used_mb, live_mb=stats.get("allocated", 0.0),
                           step_peak_mb=stats.get("peak_allocated", 0.0), reserved_mb=stats.get("reserved", 0.0))
        for name in {n for t in thirds for n in t}:
            values = [t[name] for t in thirds if name in t]
            metrics[name] = sum(values) / len(values)
        if self.emit is not None:
            self.emit(StepFinished(step + 1, self.steps, metrics))
        return loss

    def _window(self, step: int) -> tuple[list, list, dict]:
        """The ``accumulation`` micro-steps and the update. Past the point where an update may have been applied,
        ``_committed`` says a failure cannot be run again."""
        if self._fused:
            self.optimizer.begin_step(self.accumulation)
        else:
            self.optimizer.zero_grad()
        losses, thirds, timing = [], [], {}
        for micro in range(self.accumulation):
            state = StepState(step, micro, self.device)
            try:
                self.pipeline.run(state)
            except BaseException:
                self._committed = self._fused and micro + 1 == self.accumulation and state.phase == "backward"
                raise
            losses.append(state.loss.detach())
            thirds.append(thirds_of_range(state.per_sample, state.inputs.t_frac))
            for label, ms in state.timing_ms.items():
                timing[label] = timing.get(label, 0.0) + ms
            if self._fused and micro + 1 < self.accumulation:
                self.optimizer.next_pass()
        if not self._fused:
            self._committed = True
            self.optimizer.step()
        return losses, thirds, timing

    def _recover(self, step: int, error: BaseException) -> None:
        error.__traceback__ = None  # its frames hold the failed step's tensors
        self.optimizer.zero_grad()
        for p in self.optimizer.params:
            p.grad = None
        self._fetch.rewind()
        if self.residency is not None:
            self.residency.recover(step, error)
        else:
            gc.collect()

    def _explained(self, step: int, error: Exception) -> Exception:
        """Out-of-memory errors say where the card's memory was; anything else passes through unchanged."""
        if self.residency is None or not is_out_of_memory(error):
            return error
        return MemoryBudgetExceeded(f"step {step + 1} ran out of memory ({str(error).splitlines()[0] if str(error) else type(error).__name__}); "
                                    f"memory by source: {self.residency.breakdown()}")
