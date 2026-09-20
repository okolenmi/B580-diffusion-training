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
policy object: a first, honest version. before_step() and
ensure_loaded() both funnel through one shared _make_room() (below):
offload eligible, currently-loaded residents (registration order,
skipping whatever's excluded) once actual measured usage
(DeviceContext.memory_stats()'s own "reserved_mb" -- ResourceBudget's
own docstring: measure against reserved, not allocated) exceeds the
budget. before_step() calls it with nothing excluded (a general check
between steps); ensure_loaded(name) reloads name first, then calls it
excluding name (so reloading X, if that alone pushed usage over
budget, can free room by offloading something else currently loaded --
"offload everything else until X is done," direct feedback on a real
case this needs to handle: encoding a prompt on a cache miss when
model+optimizer are already near the budget shouldn't just silently
blow past it).

Whether there's anything to offload in the current step pipeline
(nodes/train/step_pipeline.py, nodes/train/loop.py) depends on
whether anything registered offloadable is genuinely idle for part of
a run and calls ensure_loaded() before it needs itself resident again
-- CachingTextEncoder (nodes/model/text_encoder_cache.py) does exactly
this on a cache miss, which is what makes text_encoder safe to mark
offloadable now. model/optimizer are NOT marked offloadable yet
(nodes/train/loop.py) even though this module's own machinery
would now handle the reverse direction (offloading them to make room
for text_encoder, then bringing them back) -- nothing yet calls
ensure_loaded("model")/ensure_loaded("optimizer") at the right points
in the step pipeline to make that safe. Real, disclosed, not yet done.

Two additions since the paragraphs above were written, both about
actually honoring the budget rather than just measuring against it --
see ResourceBudget.strict's own docstring and _make_room()'s comment
around its own synchronize() calls below for the full reasoning on
each:

1. `ResourceBudget.strict` (default False, unchanged behavior): when
   True, _make_room() raises instead of silently continuing once
   nothing registered offloadable is left to move and usage is still
   over budget.
2. An explicit DeviceContext.synchronize() after every offload/reload
   transition this class drives, before trusting the next
   memory_stats() read -- defensive, mirroring a hard-won lesson
   already paid for once in this project's own legacy core/trainer.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..components.device import DeviceContext
from ..resource_budget import ResourceBudget
from .coordinator import ResourceCoordinator
from .handle import DeviceResident


class ResourceControlHandle(ABC):

    @abstractmethod
    def register(self, name: str, resident: DeviceResident, offloadable: bool = False) -> None:
        """Called by the trainer once each of its own residents (model,
        optimizer, text_encoder, ...) is actually constructed --
        nothing exists to register before that. offloadable=True marks
        this resident as a candidate before_step()/ensure_loaded() may
        offload under pressure; False (the default) means never
        touched here, the same posture a resident this handle was
        never told about would get -- explicit opt-in, not an inferred
        default. Only mark a resident offloadable if something calls
        ensure_loaded() on it before it's actually needed again --
        otherwise it can be offloaded here and never brought back."""

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
        reloads it if it was offloaded, a no-op otherwise. If reloading
        it pushes measured usage over budget, offloads other
        offloadable, currently-loaded residents (registration order)
        to make room, the same as before_step() would between steps --
        "offload everything else until this one's done its work."
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

    def __init__(self, budget: ResourceBudget, device: str,
                 device_ctx: DeviceContext | None = None):
        self._budget = budget
        # device_ctx: real DeviceContext.for_device(device) by default (unchanged
        # behavior) -- overridable so a test can inject a fake one that reports
        # scripted memory_stats() without real XPU/CUDA hardware, the same explicit-
        # injection posture this project uses everywhere else (no singletons, per
        # docs/design/01-design-goals-and-constraints.md goal 3).
        self._device_ctx = device_ctx or DeviceContext.for_device(device)
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
        self._make_room(exclude=())

    def ensure_loaded(self, name: str) -> None:
        if name in self._offloaded:
            self._coordinator.reload(name)
            # Defensive, see _make_room()'s own comment on the matching offload-side
            # call below for the full reasoning -- same "don't trust a memory_stats()
            # snapshot taken right after a transfer without an explicit sync first"
            # posture, here for the reload direction: _make_room() below is about to
            # read reserved_mb again (exclude=(name,) still runs it), and whatever
            # calls ensure_loaded() is about to use `name` itself immediately after
            # this returns.
            self._device_ctx.synchronize()
            self._offloaded.discard(name)
        self._make_room(exclude=(name,))

    def _make_room(self, exclude: tuple[str, ...]) -> None:
        """Shared by before_step() (exclude=() -- a general check
        between steps) and ensure_loaded() (exclude=(name,) -- whatever
        was just reloaded is off-limits, it's needed right now, that's
        the whole reason ensure_loaded() was called). Re-measures after
        each individual offload rather than estimating from
        footprint_bytes() and offloading everything that adds up to
        enough up front: one real number from the allocator beats a
        predicted one, and stopping the moment it's enough avoids
        offloading (and later having to reload) more than the pressure
        actually required.

        Raises when self._budget.strict and usage is still over budget
        once every offloadable, currently-loaded resident (outside
        `exclude`) has been offloaded -- see ResourceBudget.strict's own
        docstring for why. Default strict=False keeps this method's
        previous behavior exactly (return once the offloadable list is
        exhausted, over budget or not) -- existing callers/tests see no
        behavior change unless they opt in."""
        stats = self._device_ctx.memory_stats()
        if stats is None:
            return
        usable_mb = self._budget.vram_budget_mb - self._budget.vram_reserve_mb
        for name in self._offloadable:
            if name in exclude or name in self._offloaded:
                continue
            if stats["reserved_mb"] <= usable_mb:
                return
            self._coordinator.offload(name)
            self._offloaded.add(name)
            # Defensive, not provable-necessary from this codebase's own offload()
            # implementations alone: every DeviceResident.offload() registered here
            # today already does a synchronous (non_blocking=False, torch's default)
            # .to("cpu")/.cpu() -- checked directly across nodes/model/lora_injector.py,
            # nodes/model/text_encoder.py, nodes/optimizer/composed.py, not assumed.
            # Still worth the explicit call: core/trainer.py's own offload path
            # (this project's legacy pipeline, same B580/XPU hardware) learned the
            # hard way that even a nominally-synchronous transfer is worth an
            # explicit synchronize() before trusting a memory snapshot taken right
            # after it ("should be synchronous ... an explicit sync is defensive" --
            # that file's own comment, at the exact preview-generation offload point
            # docs/known-issues/open.md's "device lost"/hang report names as a real
            # trigger). A comparable report exists for different training code on
            # this same hardware (kohya-ss/musubi-tuner, cited in that same
            # known-issues entry), tracing a matching hang to a missing/incomplete
            # synchronize on an XPU offload path. Cheap insurance against re-learning
            # that lesson a second time, here, in code this project's own known-issue
            # report hadn't reached yet (that entry explicitly scoped itself to
            # core/trainer.py, not nodes/).
            self._device_ctx.synchronize()
            stats = self._device_ctx.memory_stats()
        if self._budget.strict and stats["reserved_mb"] > usable_mb:
            raise RuntimeError(
                f"BudgetedResourceControlHandle: {stats['reserved_mb']:.0f}MB reserved "
                f"still exceeds the {usable_mb:.0f}MB usable budget "
                f"({self._budget.vram_budget_mb:.0f}MB minus "
                f"{self._budget.vram_reserve_mb:.0f}MB reserve) after offloading every "
                f"resident registered as offloadable -- nothing left this handle is "
                f"allowed to move. Raising now (strict=True) rather than silently "
                f"training on past the ceiling you asked for, which is exactly the "
                f"VRAM-pressure condition this handle exists to prevent. Either raise "
                f"vram_budget_mb, or register more residents as offloadable if that's "
                f"genuinely safe for them (see register()'s own docstring)."
            )

