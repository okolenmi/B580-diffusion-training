"""ResourceControlHandle: a live, per-step budget enforcer a training
loop calls into -- the same shape MonitorHandle (nodes/monitor/handle.py)
already established for a different cross-cutting concern (reporting),
applied here to VRAM budget instead.

Why a handle, not a second graph node running "alongside" the trainer:
server/graph_executor.py runs nodes in topological order, one at a
time -- there's no mechanism for two nodes to run concurrently and
exchange live signals mid-execution. A handle sidesteps that: a node
upstream of the trainer constructs one (with just a budget, no
residents yet -- those don't exist until the trainer's own build()
constructs them), the trainer registers its own residents into it as
it builds them, then calls into it at step boundaries as part of its
own single, sequential build() call. This is exactly how
MonitorHandle/LiveMonitorHandle already work (a MonitorNode constructs
one, the trainer calls .report() on it during its own loop) -- this
follows the same shape rather than inventing a new one.

register()'s own offloadable flag, not a priority number or a separate
policy object: a first, honest version. before_step() offloads
eligible residents in registration order once actual measured usage
(DeviceContext.memory_stats()'s own "reserved_mb" -- ResourceBudget's
own docstring: measure against reserved, not allocated) exceeds the
budget; ensure_loaded() reloads one specific resident right before
whatever's about to use it needs it resident again. Whether there's
anything to offload in the current step pipeline
(nodes/train/step_pipeline.py, nodes/train/supervised.py) depends on
whether anything registered offloadable is genuinely idle for part of
a run -- text_encoder is the obvious candidate once/if a caching
TextEncoder wrapper makes live encoding skippable, but that's the
wrapper's own concern, transparent to this handle either way: it just
offloads/reloads whatever it was told is offloadable, on measured
pressure, regardless of why a given step didn't end up needing it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..components.device import DeviceContext
from ..resource_policy import ResourceBudget
from .coordinator import ResourceCoordinator
from .handle import DeviceResident


class ResourceControlHandle(ABC):

    @abstractmethod
    def register(self, name: str, resident: DeviceResident, offloadable: bool = False) -> None:
        """Called by the trainer once each of its own residents (model,
        optimizer, text_encoder, ...) is actually constructed --
        nothing exists to register before that. offloadable=True marks
        this resident as a candidate before_step() may offload under
        pressure; False (the default) means never touched here, the
        same posture a resident this handle was never told about would
        get -- explicit opt-in, not an inferred default."""

    @abstractmethod
    def before_step(self, step: int) -> None:
        """Call once per step, before that step's own compute --
        checks current measured VRAM against budget and offloads
        offloadable residents (registration order) until back under
        budget or nothing offloadable is left resident. A no-op,
        cheaply, on a backend with no reserved-memory concept
        (DeviceContext.memory_stats() -> None, e.g. CPU) or when
        nothing is currently over budget -- the common case every
        step, not the exceptional one."""

    @abstractmethod
    def ensure_loaded(self, name: str) -> None:
        """Call right before using a specific registered resident --
        reloads it if before_step() offloaded it, a no-op otherwise.
        Safe to call unconditionally before every use regardless of
        whether that resident was ever actually offloaded (the common
        case, absent real pressure) -- the check inside is cheap, and
        calling it unconditionally is what makes this safe to wire into
        a step pipeline phase once and forget, rather than something
        that has to track offload state itself."""


class BudgetedResourceControlHandle(ResourceControlHandle):
    """The one real implementation. Owns its own ResourceCoordinator
    (residents don't exist yet when this is constructed, so there's
    nothing for a caller-supplied coordinator to have registered
    already -- see this module's own top docstring) and its own
    DeviceContext (DeviceContext.for_device(), the same factory
    core/comfy_setup.py-adjacent code already uses, not shared with
    anything else since nothing else needs one before this)."""

    def __init__(self, budget: ResourceBudget, device: str):
        self._budget = budget
        self._device_ctx = DeviceContext.for_device(device)
        self._coordinator = ResourceCoordinator()
        self._offloadable: list[str] = []  # list, not set: registration order
        # is the offload order below, and dict/set iteration order isn't a
        # contract worth relying on even though real Python dicts keep insertion
        # order -- this is the one place that order actually matters, so it's
        # explicit here rather than borrowed incidentally from something else.
        self._offloaded: set[str] = set()

    def register(self, name: str, resident: DeviceResident, offloadable: bool = False) -> None:
        self._coordinator.register(name, resident)
        if offloadable:
            self._offloadable.append(name)

    def before_step(self, step: int) -> None:
        stats = self._device_ctx.memory_stats()
        if stats is None:
            return
        usable_mb = self._budget.vram_budget_mb - self._budget.vram_reserve_mb
        # Re-measures after each individual offload rather than estimating from
        # footprint_bytes() and offloading everything that adds up to enough up
        # front: one real number from the allocator beats a predicted one, and
        # stopping the moment it's enough avoids offloading (and later having to
        # reload) more than the pressure actually required.
        for name in self._offloadable:
            if name in self._offloaded:
                continue
            if stats["reserved_mb"] <= usable_mb:
                return
            self._coordinator.offload(name)
            self._offloaded.add(name)
            stats = self._device_ctx.memory_stats()

    def ensure_loaded(self, name: str) -> None:
        if name in self._offloaded:
            self._coordinator.reload(name)
            self._offloaded.discard(name)
