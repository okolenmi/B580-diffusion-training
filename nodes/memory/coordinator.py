"""ResourceCoordinator/OffloadOrchestrator (backlog item 12,
docs/training_pipeline_design.md sections 5.1, 5.2).

Sequenced last in the whole backlog on purpose: needs several real
DeviceResidents in place and individually exercised first, or it's
coordinating nothing concrete yet. That's now true --
OptimizerHandle (item 3), TrainableModel (item 9), and TextEncoder (this
item) are all real, tested DeviceResidents.
"""

from __future__ import annotations

from abc import ABC
from typing import Callable, Optional

from .handle import DeviceResident


class ResourceCoordinator:
    """Tracks every DeviceResident a run has constructed (explicit
    register() calls at construction time -- never reflection, never a
    global registry reached for by import). Doesn't decide *when* to
    offload anything by itself in this base form -- that's
    OffloadOrchestrator below, layered on top. This class only answers
    "what do I currently own, and what's my current total footprint,"
    and provides the one bulk operation ("offload everything except
    these") that's otherwise easy to get subtly wrong by hand (forgetting
    one resident, offloading in the wrong order and causing an
    intermediate OOM that wouldn't happen with the right order)."""

    def __init__(self):
        self._residents: dict[str, DeviceResident] = {}

    def register(self, name: str, resident: DeviceResident) -> None:
        self._residents[name] = resident

    def total_footprint_bytes(self) -> int:
        return sum(r.footprint_bytes() for r in self._residents.values())

    def offload_all_except(self, keep: set) -> None:
        for name, resident in self._residents.items():
            if name not in keep:
                resident.offload()

    def reload(self, name: str, device: Optional[str] = None) -> None:
        self._residents[name].reload(device)


class TrainingLifecycleEvent(ABC):
    """Marker base -- each concrete event is a plain, immutable value; no
    behavior of its own. The three below are illustrative, matching the
    points core/trainer.py's own hand-written offload calls already exist
    at today -- not yet published by anything in nodes/train/, since
    SupervisedLoRATrainerNode v1's own documented scope (see that
    module's docstring) excludes cyclic/teacher-rollout caching,
    checkpoint cadence, and preview generation entirely. Ready for when
    that functionality gets ported to nodes/, not simulating it now."""


class CacheRebuildStarting(TrainingLifecycleEvent):
    def __init__(self, cache_name: str):
        self.cache_name = cache_name


class PreviewGenerationStarting(TrainingLifecycleEvent):
    pass


class CheckpointSaveStarting(TrainingLifecycleEvent):
    def __init__(self, path: str):
        self.path = path


class OffloadOrchestrator:
    """Subscribes to TrainingLifecycleEvents, drives a ResourceCoordinator
    in response. This is the principled version of what core/trainer.py's
    hand-written offload calls at specific points in the training loop
    are doing today, ad hoc, per call site -- made into one reviewed
    place with an explicit, testable event -> action mapping, rather than
    N scattered `.to("cpu")` calls that each have to remember to exist.

    A genuine, non-trivial piece of design, not a small refactor -- and
    explicitly NOT a claim that it fixes the open "device lost" report on
    its own. That report needs its actual root cause found first --
    docs/suspicious_findings.md's entry on it describes a plausible
    root-cause *shape* (an async/non-blocking transfer without a matching
    explicit synchronize on the XPU offload path, per a similar symptom
    traced elsewhere to exactly that), not a confirmed diagnosis in this
    codebase. A correctness bug of that shape, if that's what it is,
    isn't fixed by this orchestrator's existence -- this fixes the
    *coordination* problem (one reviewed place offload calls happen,
    instead of scattered per-call-site `.to("cpu")`s), which is necessary
    but not sufficient on its own.

    Same explicit-injection, no-singleton shape monitor_bus.py's
    MonitorBus and nodes/monitor/handle.py's MonitorHandle already
    establish for this project's other cross-cutting concern -- not
    reusing either class directly (MonitorBus is async-event-loop-shaped,
    for SSE streaming to a dashboard; this is a synchronous, in-process
    dispatcher, a genuinely different shape), but the same architectural
    principle: constructed explicitly, handed to whatever needs it, never
    a module-level instance reached for by import."""

    def __init__(self, coordinator: ResourceCoordinator, device_ctx):
        self._coordinator = coordinator
        self._device_ctx = device_ctx
        self._handlers: dict[type, list[Callable]] = {}

    def on(self, event_type: type, handler: Callable) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: TrainingLifecycleEvent) -> None:
        for handler in self._handlers.get(type(event), []):
            handler(event, self._coordinator, self._device_ctx)
