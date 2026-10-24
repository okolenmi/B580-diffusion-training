"""The profiling window of a training run: which steps are measured, what is attached to them, and the report at the end."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Callable

from torch import nn

from .budget import StepRecord
from .ceilings import load_or_measure
from .clock import DeviceClock
from .modules import ModuleTimer
from .report import ReportBuilder, ReportWriter
from .trace import KernelTrace


class PerfSession:
    """Given to the trainer as ``perf``. ``step(n)`` wraps one optimizer step; steps before ``warmup`` run untouched (kernels get
    built, shapes get seen), the next ``steps`` are timed alone (the step time), the ``steps`` after them run under the module hooks
    and the profiler (device and module times; their own wall time is inflated), each timed with the device idle at both ends. ``observe`` receives the step's metrics (replayed or eager, phase times, peak memory); ``close`` writes the report."""

    def __init__(self, model: nn.Module, device: str, folder: str | Path, *, warmup: int = 8, steps: int = 3, modules: bool = True,
                 trace: bool = False, quick_ceilings: bool = False, ceilings_path: str | Path | None = None, say: Callable[[str], None] = print, meta: dict | None = None):
        if steps < 1:
            raise ValueError("perf.steps must be at least 1")
        self.model, self.device, self.folder = model, device, Path(folder)
        self.warmup, self.steps, self.modules, self.trace = warmup, steps, modules, trace
        self._quick, self._say, self._meta = quick_ceilings, say, meta or {}
        self._ceilings_path = Path(ceilings_path) if ceilings_path else self.folder / "ceilings.json"
        self.clock = DeviceClock(device)
        self.records: list[StepRecord] = []  # profiled steps
        self.clean: list[StepRecord] = []    # timed alone
        self.seconds: dict[str, float] = {}  # what the report cost, by stage
        self._timer: ModuleTimer | None = None
        self._kernels: KernelTrace | None = None
        self._current: StepRecord | None = None
        self._open = False
        self.report: dict | None = None
        self.report_path: Path | None = None

    PHASES = ("fetch_batch", "prepare_inputs", "encode_conditioning", "forward", "loss", "backward", "graphed_compute")  # phase labels, present when the trainer is timed

    @property
    def done(self) -> bool:
        return len(self.records) >= self.steps

    @contextlib.contextmanager
    def step(self, number: int):
        if number < self.warmup or self.done:
            yield
            return
        profiled = len(self.clean) >= self.steps
        if profiled:
            self._begin()
        elif not self.clean:
            self._say(f"perf: timing steps {self.warmup + 1} to {self.warmup + self.steps}, then profiling steps {self.warmup + self.steps + 1} to {self.warmup + 2 * self.steps}")
        self.clock.synchronize()
        started = time.perf_counter()
        self._current = StepRecord(number, 0.0)
        try:
            yield
        finally:
            self.clock.synchronize()
            self._current.wall_ms = (time.perf_counter() - started) * 1000
            (self.records if profiled else self.clean).append(self._current)
            self._current = None
        if profiled and self.done:
            self._end()

    def observe(self, metrics: dict) -> None:
        record = self._current
        if record is None:
            return
        replays = metrics.get("graph_replays")
        if replays is not None:
            record.replayed = replays > getattr(self, "_replays", 0)
            self._replays = replays
        record.shape = (metrics.get("batch"), metrics.get("latent_h"), metrics.get("latent_w"))
        record.peak_mb = metrics.get("step_peak_mb")
        record.phase_ms = {k: metrics[k] for k in self.PHASES if k in metrics}

    def _begin(self) -> None:
        if self._open:
            return
        self._open = True
        if self.modules:
            self._timer = ModuleTimer(self.model, self.clock).attach()
        self._kernels = KernelTrace(self.device, trace_path=str(self.folder / "trace.json") if self.trace else None)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._kernels.__enter__()

    def _end(self) -> None:
        self.clock.synchronize()
        self._say("perf: window done, processing the profile (single-threaded, this takes a while on a real model) ...")
        stages = self.seconds
        self._lap(None)
        self._kernels.__exit__(None, None, None)
        stages.update(self._kernels.result.seconds)
        self._lap(None)
        records = []
        if self._timer is not None:
            self._timer.detach()
            records = self._timer.records()
        self._lap("module timings", stages)
        ceilings = load_or_measure(self._ceilings_path, self.device, quick=self._quick)
        self._lap("ceilings", stages)
        self.report = ReportBuilder(ceilings, self.device, self._meta).build(self.records, self._kernels.result, records, self.clean, stages)
        self._lap("report build", stages)
        self.report_path = ReportWriter(self.folder).write(self.report)
        self._lap("report write", stages)
        self._say(f"perf: report written to {self.report_path} (" + ", ".join(f"{k} {v:.1f} s" for k, v in stages.items()) + ")")
        self._open = False

    def _lap(self, label: str | None, into: dict[str, float] | None = None) -> None:
        now = time.perf_counter()
        if label and into is not None:
            into[label] = now - self._mark
        self._mark = now

    def close(self) -> None:
        """Writes the report if the run ended inside the window."""
        if self._open and self.records:
            self.steps = len(self.records)
            self._end()
        elif not self.records:
            self._say(f"perf: the run ended before the profiled window (warm-up {self.warmup} steps, then {self.steps} timed and {self.steps} profiled); nothing was profiled")
