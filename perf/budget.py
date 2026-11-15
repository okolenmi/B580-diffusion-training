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
    enqueue_ms: float | None = None  # until the host had issued the whole step, before it waited for the device
    eager_profile: bool = False  # a profiled step of a graph run, made to run the eager phases
    steady: bool = True  # False for a step that met a new shape, captured a graph, or ran eagerly in a graph run: its time is not the step's cost


def build_budget(steps: list[StepRecord], trace: TraceResult, clean: list[StepRecord] | None = None) -> dict:
    """Numbers per step. ``steps`` ran under the profiler and the module hooks (device time is exact, wall time is inflated);
    ``clean`` ran with only a device sync at both ends and gives the step time. Device busy time is the sum of kernel durations
    (one in-order queue), so utilisation is a lower bound when kernels overlap; on a CPU the 'device' time is host operator time
    and utilisation is not reported."""
    count = max(len(steps), 1)
    timed = clean or steps
    wall = [s.wall_ms for s in timed]
    out: dict = {"steps": len(steps), "timed_steps": len(timed), "wall_clean": bool(clean),
                 "wall_ms": statistics.median(wall) if wall else 0.0,
                 "wall_ms_min": min(wall) if wall else 0.0,
                 "profiled_ms": _mean([s.wall_ms for s in steps]),
                 "enqueue_ms": _median([s.enqueue_ms for s in timed if s.enqueue_ms is not None]),
                 "eager_ms": _median([s.wall_ms for s in timed if not s.replayed]),
                 "replayed_ms": _median([s.wall_ms for s in timed if s.replayed]),
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
    median = statistics.median(wall) if wall else 0.0
    out["outliers"] = [{"step": s.step, "shape": str(s.shape), "wall_ms": s.wall_ms} for s in timed if median and s.wall_ms > 2 * median]
    out["shapes_differ"] = bool(clean) and {str(s.shape) for s in clean} != {str(s.shape) for s in steps}
    out["peak_mb"] = max((s.peak_mb for s in steps if s.peak_mb is not None), default=None)
    return out


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None
