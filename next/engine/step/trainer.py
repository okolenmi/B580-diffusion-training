"""The training loop: optimizer steps made of ``accumulation`` micro-steps."""

from __future__ import annotations

import contextlib
import gc
import time
from dataclasses import dataclass
from contextlib import AbstractContextManager
from collections.abc import Callable, Iterable

import torch

from ...graph.events import StepFinished
from ..optim.handle import FusedOptimizer, Optimizer
from ..runtime.memory import MemoryBudgetExceeded, is_out_of_memory
from ..runtime.planner import chunk_sizes
from ..runtime.thrash import PrimitiveCacheMonitor
from .memory_probe import MemoryProbe
from .loss import LossWeighting, thirds_of_range
from .regional import RegionalUpdate, RegionGradients
from .phases import (Backward, EncodeConditioning, FetchBatch, Forward, Loss, Pipeline, PrepareInputs, StepPhase, Timed)
from .schedule import LRSchedule
from .state import StepState, split_batch


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

    With a memory controller the run starts with a probe (a sample and a few samples of the largest shape, gradients dropped)
    so the memory a batch needs is known before the first step, and every batch the controller predicts will not fit runs as
    chunks whose gradients add up to the whole batch's. A step that runs out of memory anyway is run again once, in
    twice as many chunks, after the memory it held is given back, as long as no update has been applied yet.
    """

    OOM_RETRIES = 1

    def __init__(self, *, batches: Iterable[dict], denoiser, conditioner, optimizer: Optimizer, schedule: LRSchedule,
                 weighting: LossWeighting, steps: int, accumulation: int = 1, device="cpu", gate=None,
                 gate_window=None, emit: Callable[[StepFinished], None] | None = None,
                 should_stop: Callable[[], bool] | None = None, synchronize: Callable[[], None] | None = None,
                 timed: bool = False, residency=None, density=None, placement=None, balance=None, probe=None, task: Callable[[str], AbstractContextManager] | None = None,
                 graph: Callable[[list[StepPhase]], StepPhase] | None = None,
                 warn: Callable[[str], None] | None = None, plan_chunks: bool = True, perf=None, regional: RegionalUpdate | None = None):
        if accumulation < 1 or steps < 0:
            raise ValueError("steps must not be negative and accumulation must be at least 1")
        self.optimizer, self.schedule, self.steps, self.accumulation = optimizer, schedule, steps, accumulation
        self.device, self.emit, self.should_stop, self.residency = device, emit, should_stop, residency
        self._balance, self._plan_chunks = balance, plan_chunks
        self._perf = perf  # duck-typed: step(n) context manager, observe(metrics), close(); the engine does not know the profiler
        if regional is not None and (isinstance(optimizer, FusedOptimizer) or graph is not None):
            raise ValueError("a gradient policy reads each region's gradient before the update: it needs the plain optimizer (not fused) and no graph replay")
        self._regional, self._regional_metrics = regional, {}
        self._warn, self._cache_monitor = warn, PrimitiveCacheMonitor()
        self._time_probe = probe
        self._task = task or (lambda title: contextlib.nullcontext())
        self._fused = isinstance(optimizer, FusedOptimizer)
        self._batches = batches
        self._fetch = FetchBatch(batches)
        compute: list[StepPhase] = [Forward(denoiser), Loss(weighting, balance), Backward(accumulation)]
        self.graphed = graph(compute) if graph is not None else None  # replaces the three phases above, which it falls back to
        phases = [PrepareInputs(gate, gate_window), EncodeConditioning(conditioner), *([self.graphed] if self.graphed else compute)]
        fetch = self._fetch
        if timed:
            phases = [Timed(p, synchronize or (lambda: None)) for p in phases]
            fetch = Timed(fetch, synchronize or (lambda: None))
        self._fetch_phase = fetch
        self.pipeline = Pipeline(phases)
        self._memory_probe = MemoryProbe(residency, optimizer, self.pipeline, batches, self._fetch, device, self._task, density, placement, splits=plan_chunks)

    def run(self, start: int = 0) -> RunResult:
        if self.steps:
            self._memory_probe.run()
        attach = getattr(self._perf, "attach_graph", None)
        if attach is not None and self.graphed is not None:
            attach(self.graphed)
        last = None
        step = start
        for step in range(start, self.steps):
            if self.should_stop is not None and self.should_stop():
                if self._perf is not None:
                    self._perf.close()
                return RunResult(step - start, self.steps, last, stopped=True)
            with self._perf.step(step) if self._perf is not None else contextlib.nullcontext():
                last = self._step(step)
        if self._perf is not None:
            self._perf.close()
        return RunResult(max(self.steps - start, 0), self.steps, last)

    def _step(self, step: int) -> float:
        started = time.perf_counter()
        lr = self.schedule.value(step)
        if self.residency is not None:
            self.residency.before_step(step)
        self.optimizer.set_lr(lr)
        fetched, timing = self._fetch_window(step)
        probe_metrics = self._reference_probe(fetched)
        losses, thirds, spent = self._update(step, fetched)
        for label, ms in spent.items():
            timing[label] = timing.get(label, 0.0) + ms
        loss = float(sum(losses) / len(losses))
        if self.residency is not None:
            self.residency.after_step(step)
        metrics = self._metrics(step, loss, lr, started, fetched, timing, thirds, probe_metrics)
        if self._perf is not None:
            self._perf.observe(metrics)
        if self.emit is not None:
            self.emit(StepFinished(step + 1, self.steps, metrics))
        return loss

    def _fetch_window(self, step: int) -> tuple[list[dict], dict[str, float]]:
        """The ``accumulation`` batches of one optimizer step and what fetching them cost, by phase."""
        fetched: list[dict] = []
        timing: dict[str, float] = {}
        for micro in range(self.accumulation):
            state = StepState(step, micro, self.device)
            self._fetch_phase.run(state)
            fetched.append(state.batch)
            for label, ms in state.timing_ms.items():
                timing[label] = timing.get(label, 0.0) + ms
        return fetched, timing

    def _reference_probe(self, fetched: list[dict]) -> dict[str, float]:
        """Feed the time probe; its first record is taken before the first update, so it is the reference."""
        if self._time_probe is None:
            return {}
        first = not self._time_probe.ready
        for batch in fetched:
            self._time_probe.capture(batch)
        return self._time_probe.run() if first and self._time_probe.ready else {}

    def _update(self, step: int, fetched: list[dict]) -> tuple[list, list, dict]:
        """The window's forward, backward and update; a step that runs out of memory is run again in more chunks when
        no update has been applied yet (a retry in more chunks would change the numbers, so not when chunks are off)."""
        retries = self.OOM_RETRIES if self._plan_chunks else 0
        for attempt in range(retries + 1):
            self._committed = False
            try:
                return self._window(step, fetched, attempt)
            except Exception as error:  # noqa: BLE001 - narrowed just below
                if self._committed or attempt == retries or not is_out_of_memory(error):
                    explained = self._explained(step, error)
                    if explained is error:
                        raise
                    raise explained from error
                self._recover(step, error)
        raise AssertionError("unreachable: the last attempt returns or raises")

    def _metrics(self, step: int, loss: float, lr: float, started: float, fetched: list[dict], timing: dict[str, float],
                 thirds: list, probe_metrics: dict[str, float]) -> dict[str, float]:
        height, width = fetched[-1]["x_t"].shape[-2:]
        metrics = {"loss": loss, "lr": lr, "step_ms": (time.perf_counter() - started) * 1000, "batch": len(fetched[-1]["samples"]),
                   "latent_h": height, "latent_w": width, "epoch": fetched[-1].get("epoch", 0),
                   "epoch_end": bool(fetched[-1].get("last_in_epoch", False)), **timing}
        if self.residency is not None and self.residency.last_used_mb is not None:
            stats = self.residency.last_stats
            metrics.update(device_mb=self.residency.last_used_mb, live_mb=stats.get("allocated", 0.0),
                           step_peak_mb=stats.get("peak_allocated", 0.0), reserved_mb=stats.get("reserved", 0.0))
        if self.graphed is not None:
            metrics.update(graph_replays=self.graphed.replays, graphs=self.graphed.captures)
        for name in {n for t in thirds for n in t}:
            values = [t[name] for t in thirds if name in t]
            metrics[name] = sum(values) / len(values)
        if self._time_probe is not None and not probe_metrics and self._time_probe.due(step):
            probe_metrics = self._time_probe.run()
        metrics.update(probe_metrics)
        metrics.update(self._regional_metrics)
        if self._warn is not None:
            message = self._cache_monitor.observe((metrics["batch"], height, width), metrics["step_ms"] / 1000)
            if message:
                self._warn(message)
        if self._balance is not None:
            self._balance.observe(metrics)
            metrics.update({name.replace("loss_t_", "weight_t_"): w for name, w in self._balance.weights.items()})
            shares = self._balance.third_probabilities()
            if shares is not None:
                metrics.update(zip(("draw_t_low", "draw_t_mid", "draw_t_high"), shares))
        return metrics

    def _chunks(self, batch: dict, attempt: int) -> list[int]:
        x = batch["x_t"]
        count = x.shape[0]
        sizes = self.residency.plan(count, x.shape[-2] * x.shape[-1]) if self.residency is not None and self._plan_chunks else [count]
        if attempt and len(sizes) < count:
            sizes = chunk_sizes(count, min(count, len(sizes) * 2))
        return sizes

    def _window(self, step: int, fetched: list[dict], attempt: int) -> tuple[list, list, dict]:
        """The window's micro-steps, each as one or more chunks, and the update. Past the point where an update may
        have been applied, ``_committed`` says a failure cannot be run again."""
        if self._regional is not None:
            return self._regional_window(step, fetched, attempt)
        plans = [(batch, self._chunks(batch, attempt)) for batch in fetched]
        passes = sum(len(sizes) for _, sizes in plans)
        if self._fused:
            self.optimizer.begin_step(passes)
        else:
            self.optimizer.zero_grad()
        losses, thirds, timing = [], [], {}
        done = 0
        for micro, (batch, sizes) in enumerate(plans):
            total, loss, per_sample, fractions = sum(sizes), 0.0, [], []
            for part, size in zip(split_batch(batch, sizes), sizes):
                state = StepState(step, micro, self.device, batch=part, scale=size / total)
                try:
                    self.pipeline.run(state)
                except BaseException:
                    self._committed = self._fused and done + 1 == passes and state.phase == "backward"
                    raise
                loss = loss + state.loss.detach() * state.scale
                per_sample.append(state.per_sample.detach())
                fractions.append(state.inputs.t_frac.detach().flatten().float().cpu())
                for label, ms in state.timing_ms.items():
                    timing[label] = timing.get(label, 0.0) + ms
                done += 1
                if self._fused and done < passes:
                    self.optimizer.next_pass()
            losses.append(loss)
            thirds.append(thirds_of_range(torch.cat([p.flatten() for p in per_sample]), torch.cat(fractions)))
        if not self._fused:
            self._committed = True
            self.optimizer.step()
        return losses, thirds, timing

    def _regional_window(self, step: int, fetched: list[dict], attempt: int) -> tuple[list, list, dict]:
        """Like ``_window``, but each region of a batch (its samples at one part of the timestep range) runs on its own from zero
        gradients; the policy combines the regions' gradients, the average over the window is written back and applied."""
        grads, total, used = RegionGradients(self.optimizer), None, 0
        losses, thirds, timing, per_region = [], [], {}, {}
        for micro, batch in enumerate(fetched):
            region = batch.get("region")
            sizes = [int(n) for n in torch.bincount(region).tolist() if n] if region is not None else [batch["x_t"].shape[0]]
            matrix, loss, per_sample, fractions = [], 0.0, [], []
            for index, part in enumerate(split_batch(batch, sizes)):
                self.optimizer.zero_grad()
                value, losses_part, fractions_part, spent = self._region_pass(step, micro, part, attempt)
                matrix.append(grads.read())
                per_region.setdefault(index, []).append(float(value))
                loss, per_sample, fractions = loss + value / len(sizes), per_sample + losses_part, fractions + fractions_part
                for label, ms in spent.items():
                    timing[label] = timing.get(label, 0.0) + ms
            direction = self._regional.direction(torch.stack(matrix))
            if direction is not None:
                total = direction if total is None else total + direction
                used += 1
            losses.append(loss)
            thirds.append(thirds_of_range(torch.cat([p.flatten() for p in per_sample]), torch.cat(fractions)))
        self._regional_metrics = {f"loss_region_{i}": sum(v) / len(v) for i, v in per_region.items()}
        self._regional_metrics.update(self._regional.policy.last)
        self._regional_metrics["update_skipped"] = float(total is None)
        self.optimizer.zero_grad()
        if total is not None:
            grads.write(total / used)
            self._committed = True
            self.optimizer.step()
        return losses, thirds, timing

    def _region_pass(self, step: int, micro: int, part: dict, attempt: int) -> tuple[float, list, list, dict]:
        """One region's forward and backward, as chunks when it is too big; leaves its gradient on the parameters."""
        sizes = self._chunks(part, attempt)
        total, loss, per_sample, fractions, timing = sum(sizes), 0.0, [], [], {}
        for piece, size in zip(split_batch(part, sizes), sizes):
            state = StepState(step, micro, self.device, batch=piece, scale=size / total)
            self.pipeline.run(state)
            loss = loss + state.loss.detach() * state.scale
            per_sample.append(state.per_sample.detach())
            fractions.append(state.inputs.t_frac.detach().flatten().float().cpu())
            for label, ms in state.timing_ms.items():
                timing[label] = timing.get(label, 0.0) + ms
        return loss, per_sample, fractions, timing

    def _recover(self, step: int, error: BaseException) -> None:
        error.__traceback__ = None  # its frames hold the failed step's tensors
        self.optimizer.discard_grads()
        if self.residency is not None:
            self.residency.recover(step, error)
        else:
            gc.collect()

    def _explained(self, step: int, error: Exception) -> Exception:
        """Out-of-memory errors say where the card's memory was; anything else passes through unchanged."""
        if self.residency is None or not is_out_of_memory(error):
            return error
        rows = ", ".join(f"{r['name']} {r['mb']:.0f}" for r in self.residency.ledger())
        return MemoryBudgetExceeded(f"step {step + 1} ran out of memory ({str(error).splitlines()[0] if str(error) else type(error).__name__}); "
                                    f"MB by claim: {rows}")
