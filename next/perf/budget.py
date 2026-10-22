"""The step budget: wall time, device busy time, utilisation and launch counts for the profiled window."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .trace import TraceResult


@dataclass
class StepRecord:
    step: int
    wall_ms: float
    replayed: bool = False
    shape: tuple | None = None
    phase_ms: dict[str, float] = field(default_factory=dict)
    peak_mb: float | None = None


def build_budget(steps: list[StepRecord], trace: TraceResult) -> dict:
    """Numbers per step. Device busy time is the sum of kernel durations (one in-order queue), so utilisation is a lower bound when
    kernels overlap; on a CPU the 'device' time is host operator time and utilisation is not reported."""
    count = max(len(steps), 1)
    wall = [s.wall_ms for s in steps]
    out: dict = {"steps": len(steps), "wall_ms": statistics.fmean(wall) if wall else 0.0,
                 "wall_ms_min": min(wall) if wall else 0.0,
                 "eager_ms": _mean([s.wall_ms for s in steps if not s.replayed]),
                 "replayed_ms": _mean([s.wall_ms for s in steps if s.replayed]),
                 "device_ms": trace.device_us / 1000 / count, "launches": trace.launches / count,
                 "ops": trace.ops / count, "host_ms": trace.host_op_us / 1000 / count,
                 "device_is_host": trace.device_is_host}
    out["host_us_per_launch"] = trace.host_op_us / trace.launches if trace.launches else 0.0
    out["utilisation"] = None if trace.device_is_host or not out["wall_ms"] else min(out["device_ms"] / out["wall_ms"], 1.0)
    phases: dict[str, float] = {}
    for s in steps:
        for label, ms in s.phase_ms.items():
            phases[label] = phases.get(label, 0.0) + ms / count
    out["phase_ms"] = phases
    shapes: dict[str, list[float]] = {}
    for s in steps:
        shapes.setdefault(str(s.shape), []).append(s.wall_ms)
    median = statistics.median(wall) if wall else 0.0
    out["outliers"] = [{"step": s.step, "shape": str(s.shape), "wall_ms": s.wall_ms} for s in steps if median and s.wall_ms > 2 * median]
    out["peak_mb"] = max((s.peak_mb for s in steps if s.peak_mb is not None), default=None)
    return out


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None
