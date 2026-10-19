"""The training loop: optimizer steps made of ``accumulation`` micro-steps."""

from __future__ import annotations

import contextlib
import gc
import time
from dataclasses import dataclass
from typing import Callable, ContextManager, Iterable

import torch

from ...graph.events import StepFinished
from ..optim.handle import FusedOptimizer, Optimizer
from ..runtime.memory import MemoryBudgetExceeded, is_out_of_memory
from ..runtime.planner import chunk_sizes
from .loss import LossWeighting, thirds_of_range
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

    With a memory controller the run starts with a probe (one sample of the largest shape, gradients dropped) so
    the cost of a pixel is known before the first step, and every batch the controller predicts will not fit runs as
    chunks whose gradients add up to the whole batch's. A step that runs out of memory anyway is run again once, in
    twice as many chunks, after the memory it held is given back, as long as no update has been applied yet.
    """

    OOM_RETRIES = 1

    def __init__(self, *, batches: Iterable[dict], denoiser, conditioner, optimizer: Optimizer, schedule: LRSchedule,
                 weighting: LossWeighting, steps: int, accumulation: int = 1, device="cpu", gate=None,
                 gate_window=None, emit: Callable[[StepFinished], None] | None = None,
                 should_stop: Callable[[], bool] | None = None, synchronize: Callable[[], None] | None = None,
                 timed: bool = False, residency=None, density=None, balance=None, probe=None, task: Callable[[str], ContextManager] | None = None,
                 graph: Callable[[list[StepPhase]], StepPhase] | None = None):
        if accumulation < 1 or steps < 0:
            raise ValueError("steps must not be negative and accumulation must be at least 1")
        self.optimizer, self.schedule, self.steps, self.accumulation = optimizer, schedule, steps, accumulation
        self.device, self.emit, self.should_stop, self.residency = device, emit, should_stop, residency
        self._balance = balance
        self._time_probe = probe
        self._density = density  # (CheckpointDensity, TrainableDenoiser) or None
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

    def run(self, start: int = 0) -> RunResult:
        self._probe()
        last = None
        step = start
        for step in range(start, self.steps):
            if self.should_stop is not None and self.should_stop():
                return RunResult(step - start, self.steps, last, stopped=True)
            last = self._step(step)
        return RunResult(max(self.steps - start, 0), self.steps, last)

    def _probe(self) -> None:
        """One sample of the largest shape through forward and backward, changing nothing: what a pixel costs."""
        make = getattr(self._batches, "probe_batch", None)
        if self.residency is None or make is None or self.steps == 0:
            return
        batch = make(self._fetch.peek())
        units = batch["x_t"].shape[-2] * batch["x_t"].shape[-1]

        def measure(share: float | None = None) -> float:
            if share is not None:
                self._density[1].set_checkpoint_share(share)
                self.residency.planner.reset()
            with self._task("Measuring what a pixel costs"), self.residency.measuring(units) as result, self.optimizer.suspended():
                self.pipeline.run(StepState(-1, 0, self.device, batch=batch))
            return result["cost"]

        budget = self.residency.budget_mb()
        if self._density is None or budget is None:
            measure()
            return
        tuner, model = self._density
        floor = self.residency.accelerator.memory().get("allocated", 0.0)
        chosen = tuner.choose(measure, model.checkpoint_share, floor, self._batches.largest_units(), budget)
        self.residency.note("checkpoint_share", share=chosen, of_blocks="transformer")

    def _step(self, step: int) -> float:
        started = time.perf_counter()
        lr = self.schedule.value(step)
        if self.residency is not None:
            self.residency.before_step(step)
        self.optimizer.set_lr(lr)
        fetched = []
        timing: dict[str, float] = {}
        for micro in range(self.accumulation):
            state = StepState(step, micro, self.device)
            self._fetch_phase.run(state)
            fetched.append(state.batch)
            for label, ms in state.timing_ms.items():
                timing[label] = timing.get(label, 0.0) + ms
        probe_metrics: dict[str, float] = {}
        if self._time_probe is not None:
            first = not self._time_probe.ready
            for batch in fetched:
                self._time_probe.capture(batch)
            if first and self._time_probe.ready:
                probe_metrics = self._time_probe.run()  # before the first update: the reference record
        for attempt in range(self.OOM_RETRIES + 1):
            self._committed = False
            failure = None
            try:
                losses, thirds, spent = self._window(step, fetched, attempt)
                break
            except Exception as error:  # noqa: BLE001 - narrowed just below
                if self._committed or attempt == self.OOM_RETRIES or not is_out_of_memory(error):
                    explained = self._explained(step, error)
                    if explained is error:
                        raise
                    raise explained from error
                failure = error
            self._recover(step, failure)
        for label, ms in spent.items():
            timing[label] = timing.get(label, 0.0) + ms
        loss = float(sum(losses) / len(losses))
        if self.residency is not None:
            self.residency.after_step(step)
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
        if self._balance is not None:
            self._balance.observe(metrics)
            metrics.update({name.replace("loss_t_", "weight_t_"): w for name, w in self._balance.weights.items()})
        if self.emit is not None:
            self.emit(StepFinished(step + 1, self.steps, metrics))
        return loss

    def _chunks(self, batch: dict, attempt: int) -> list[int]:
        x = batch["x_t"]
        count = x.shape[0]
        sizes = self.residency.plan(count, x.shape[-2] * x.shape[-1]) if self.residency is not None else [count]
        if attempt and len(sizes) < count:
            sizes = chunk_sizes(count, min(count, len(sizes) * 2))
        return sizes

    def _window(self, step: int, fetched: list[dict], attempt: int) -> tuple[list, list, dict]:
        """The window's micro-steps, each as one or more chunks, and the update. Past the point where an update may
        have been applied, ``_committed`` says a failure cannot be run again."""
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
