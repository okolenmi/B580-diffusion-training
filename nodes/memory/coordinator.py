"""ResourceCoordinator/OffloadOrchestrator: tracking and event-driven
offload coordination for every DeviceResident a run constructs."""

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
    behavior of its own. The three below correspond to the points a
    training loop's own offload calls would happen at (cache rebuild,
    preview generation, checkpoint save)."""


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
    in response -- one reviewed place with an explicit, testable
    event -> action mapping, instead of scattered per-call-site
    `.to("cpu")` calls that each have to remember to exist.

    This coordinates *when* registered residents offload in response to
    a published event; it doesn't fix a correctness bug in code that
    doesn't publish events through it at all.

    Same explicit-injection, no-singleton shape as this project's other
    cross-cutting concern (MonitorBus/MonitorHandle) -- not reusing
    either class directly (MonitorBus is async-event-loop-shaped, for SSE
    streaming; this is a synchronous, in-process dispatcher), but the
    same principle: constructed explicitly, handed to whatever needs it,
    never a module-level instance reached for by import."""

    def __init__(self, coordinator: ResourceCoordinator, device_ctx):
        self._coordinator = coordinator
        self._device_ctx = device_ctx
        self._handlers: dict[type, list[Callable]] = {}

    def on(self, event_type: type, handler: Callable) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: TrainingLifecycleEvent) -> None:
        for handler in self._handlers.get(type(event), []):
            handler(event, self._coordinator, self._device_ctx)
