"""The profiler's kernel table for a window of steps, grouped into categories."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from torch.profiler import ProfilerActivity, profile

from .categories import Categorizer
from .events import OpTree, read_events
from .provenance import Provenance, ProvenanceBuilder


@dataclass
class KernelRow:
    name: str
    category: str
    calls: int
    device_us: float
    host_us: float = 0.0


@dataclass
class TraceResult:
    rows: list[KernelRow] = field(default_factory=list)
    ops: int = 0               # host-side operator calls
    host_op_us: float = 0.0    # time spent in them (self time)
    device_is_host: bool = False  # no device activity was recorded: the rows are host operators
    seconds: dict[str, float] = field(default_factory=dict)  # what collecting cost, by stage
    provenance: Provenance | None = None
    error: str = ""  # why the fast path was not used

    @property
    def device_us(self) -> float:
        return sum(r.device_us for r in self.rows)

    @property
    def launches(self) -> int:
        return sum(r.calls for r in self.rows)

    def by_category(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for r in self.rows:
            c = out.setdefault(r.category, {"calls": 0, "device_us": 0.0})
            c["calls"] += r.calls
            c["device_us"] += r.device_us
        return out


class KernelTrace:
    """Context manager around ``torch.profiler``. On a device the rows are its kernels; on a CPU they are host operators."""

    def __init__(self, device: str = "cpu", categorizer: Categorizer | None = None, trace_path: str | None = None, provenance: bool = False):
        self.categorizer = categorizer or Categorizer()
        self.trace_path = trace_path
        self.provenance = provenance
        kind = str(device).split(":")[0]
        activities = [ProfilerActivity.CPU]
        if kind == "xpu" and hasattr(ProfilerActivity, "XPU"):
            activities.append(ProfilerActivity.XPU)
        self._device_kind = kind
        self._profiler = profile(activities=activities, record_shapes=provenance)
        self.result = TraceResult()

    def __enter__(self) -> KernelTrace:
        self._profiler.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        began = time.perf_counter()
        self._profiler.__exit__(*exc)
        stopped = time.perf_counter()
        if exc[0] is None:
            try:
                self.result = self._collect()
            except Exception as error:  # noqa: BLE001 - no kernel data must not cost the step times already measured
                self.result = TraceResult(error=f"{type(error).__name__}: {error}")
            self.result.seconds = {"profiler stop": stopped - began, "events to rows": time.perf_counter() - stopped}
            if self.trace_path:
                exported = time.perf_counter()
                self._profiler.export_chrome_trace(self.trace_path)
                self.result.seconds["trace export"] = time.perf_counter() - exported

    def _collect(self) -> TraceResult:
        """Reads the profiler's raw events. ``key_averages`` builds an event object and a call tree per event in Python on one thread
        (minutes for a real step); the rows need only names, durations and, for host operators, self time."""
        kineto = getattr(getattr(self._profiler, "profiler", None), "kineto_results", None)
        if kineto is None:
            return self._collect_averaged()
        try:
            table = read_events(kineto.events(), keep_raw=self.provenance)
        except (MemoryError, RuntimeError) as error:  # the profiler's C++ side failed on this window: the slow path has its own copy of the data
            result = self._collect_averaged()
            result.error = f"{type(error).__name__}: {error}"
            return result
        tree = OpTree(table.ops)
        device: dict[str, list[float]] = {}
        for name, _, us, _ in table.kernels:
            row = device.setdefault(name, [0, 0.0])
            row[0] += 1
            row[1] += us
        host: dict[str, list[float]] = {}
        host_us = 0.0
        for op, own in zip(tree.ops, tree.own_us):
            row = host.setdefault(op.name, [0, 0.0])
            own = max(own, 0.0)
            row[0] += 1
            row[1] += own
            host_us += own
        result = TraceResult(ops=len(tree.ops), host_op_us=host_us)
        source = device or host
        result.device_is_host = not device
        result.rows = sorted((KernelRow(name, self.categorizer.category(name), int(calls), us, 0.0 if device else us) for name, (calls, us) in source.items()),
                             key=lambda r: -r.device_us)
        if self.provenance:
            result.provenance = ProvenanceBuilder(table, tree, self.categorizer).build()
        return result

    def _collect_averaged(self) -> TraceResult:
        result = TraceResult()
        device_rows, host_rows = [], []
        for event in self._profiler.key_averages():
            on_device = "cpu" not in str(getattr(event, "device_type", "cpu")).lower()
            device_us = float(getattr(event, "self_device_time_total", 0.0) or 0.0)
            host_us = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
            if on_device:
                device_rows.append(KernelRow(event.key, self.categorizer.category(event.key), event.count, device_us, host_us))
            else:
                result.ops += event.count
                result.host_op_us += host_us
                host_rows.append(KernelRow(event.key, self.categorizer.category(event.key), event.count, host_us, host_us))
        if device_rows:
            result.rows = sorted(device_rows, key=lambda r: -r.device_us)
        else:
            result.device_is_host = True
            result.rows = sorted(host_rows, key=lambda r: -r.device_us)
        return result
