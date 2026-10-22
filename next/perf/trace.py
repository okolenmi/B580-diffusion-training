"""The profiler's kernel table for a window of steps, grouped into categories."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.profiler import ProfilerActivity, profile

from .categories import Categorizer


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

    def __init__(self, device: str = "cpu", categorizer: Categorizer | None = None, trace_path: str | None = None):
        self.categorizer = categorizer or Categorizer()
        self.trace_path = trace_path
        kind = str(device).split(":")[0]
        activities = [ProfilerActivity.CPU]
        if kind == "xpu" and hasattr(ProfilerActivity, "XPU"):
            activities.append(ProfilerActivity.XPU)
        self._device_kind = kind
        self._profiler = profile(activities=activities)
        self.result = TraceResult()

    def __enter__(self) -> "KernelTrace":
        self._profiler.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._profiler.__exit__(*exc)
        if exc[0] is None:
            self.result = self._collect()
            if self.trace_path:
                self._profiler.export_chrome_trace(self.trace_path)

    def _collect(self) -> TraceResult:
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
