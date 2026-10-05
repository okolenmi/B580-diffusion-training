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

Whether there's anything to offload depends on whether a caller either
(a) marks something offloadable and calls ensure_loaded() on it before
each use, relying on before_step()'s reactive, pressure-triggered
offloading to actually move it back off between uses (the shape
CachingTextEncoder, nodes/model/text_encoder_cache.py, uses -- offload
only happens if and when measured usage actually exceeds budget), or
(b) additionally calls the newer release() below right after each use,
for a resident whose idle windows are known upfront rather than
discovered reactively -- deterministic, not pressure-triggered: gone
from GPU the moment its own phase ends, every step, budget exceeded or
not. Which residents get which treatment (or neither -- some are
resident for a run's entire duration, e.g. a large frozen base model
that's too expensive to move every step) is each trainer's own,
disclosed choice; this module provides both mechanisms, not a
one-size-fits-all policy.

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

A third addition, `release()`, is the deterministic counterpart to
`ensure_loaded()` described in (b) above -- see its own docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..components.device import DeviceContext
from ..resource_budget import ResourceBudget
from .graph_memory import GraphMemory
from .handle import DeviceResident


#: Priority bands for the two movable registration states, used when a
#: resident is registered with the run's ``GraphMemory``. The numbers
#: themselves are arbitrary; what matters is that an offloadable
#: resident sorts ahead of a sacrificable one, so a demand-driven pass
#: only reaches for the expensive last resort when nothing cheaper can
#: make room. "never" is not a band at all -- it is pinned, and pinned
#: is never a candidate.
_OFFLOADABLE_PRIORITY = 0
_SACRIFICABLE_PRIORITY = 10


class ResourceControlHandle(ABC):

    @abstractmethod
    def register(self, name: str, resident: DeviceResident, offloadable: bool = False,
                 sacrificable: bool = False) -> None:
        """Called by the trainer once each of its own residents (model,
        optimizer, text_encoder, ...) is actually constructed --
        nothing exists to register before that. offloadable=True marks
        this resident as a candidate before_step()/ensure_loaded() may
        offload under pressure; False (the default) means never
        touched here, the same posture a resident this handle was
        never told about would get -- explicit opt-in, not an inferred
        default. Only mark a resident offloadable if something calls
        ensure_loaded() on it before it's actually needed again --
        otherwise it can be offloaded here and never brought back.

        **`sacrificable` is a third state, and a real one.** `offloadable`
        answers "may this be released *between* uses", which is the
        optimizer's and the text encoder's situation: their idle windows are
        known, so dropping them the moment their phase ends is free.
        `sacrificable` answers a different question -- "may this be moved out
        of the way because something *else* needs the room right now", which
        is the large base model's situation. It is used every step, so
        releasing it proactively buys nothing and costs a great deal:
        measured on the B580 a UNet round trip is 1,449 ms down plus 1,145 ms
        up = 2,594 ms against a 2,430 ms step. Doing that from
        `before_step()` every step is a 2.1x slowdown bought with nothing.

        So these are not a weaker and a stronger `offloadable`:

          * never          -- never moves
          * offloadable    -- may be released between its own uses
          * sacrificable   -- moves only to satisfy an explicit
                              `ensure_loaded()`, and only once no
                              offloadable resident can make room

        `before_step()` honours the middle one only. `ensure_loaded()`
        honours both, and a trainer that sacrifices something owns calling
        `ensure_loaded()` on it again before its next use -- the obligation
        `offloadable` already carries, and the reason the model is
        registered non-offloadable by default rather than sacrificable.
        """

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
        reloads it if it was offloaded, a no-op *state-wise* otherwise.
        If reloading it pushes measured usage over budget, offloads other
        offloadable, currently-loaded residents (registration order)
        to make room, the same as before_step() would between steps --
        "offload everything else until this one's done its work."
        Safe to call unconditionally before every use regardless of
        whether that resident was ever actually offloaded (the common
        case, absent real pressure) -- correct either way, but not free
        when already resident: the reload branch is skipped, yet
        _make_room() still runs, and its first act is a memory_stats()
        read -- a real device query, not a cache lookup. Callers that
        can batch several imminent uses behind one call should (see
        CachingTextEncoder.encode()'s both-keys-first check -- the one
        place in this repo where the difference was worth fixing).
        Calling it unconditionally is what makes this safe to wire into
        a step pipeline phase once and forget, rather than something
        that has to track offload state itself."""

    @abstractmethod
    def release(self, name: str) -> None:
        """ensure_loaded()'s deterministic opposite: offload `name` now,
        unconditionally -- regardless of whether measured usage is
        currently over budget. For a caller that knows precisely when a
        resident's idle window starts (not just reactively, once
        something else needs the room) -- e.g. right after a text
        encoder's own conditioning-encode phase ends for this step, or
        right after an optimizer's own step() call updates its
        momentum buffers -- and wants it off GPU for the rest of that
        window on principle, not only if pressure happens to demand it.
        Safe to call unconditionally, the same as ensure_loaded() -- a
        no-op if `name` is already offloaded. Raises if `name` was
        registered with offloadable=False: release() is an explicit
        request to move something, not a hint, so silently ignoring one
        for a resident register() was told is unsafe to move would hide
        a real caller bug instead of surfacing it."""

    @abstractmethod
    def usable_budget_mb(self) -> Optional[float]:
        """The ceiling a caller can plan against directly, without
        needing to know how this handle represents its own budget
        internally (ResourceBudget is this module's own concern, not
        every caller's) -- None if this implementation has no fixed
        ceiling concept at all. For nodes/train/managed.py's own
        AdaptiveResidencyController: what "does everything fit without
        offloading" actually means here."""


class BudgetedResourceControlHandle(ResourceControlHandle):
    """A thin adapter over ``GraphMemory`` (MEM-05 #5).

    Owns almost nothing now: the eviction **order** and the resident
    bookkeeping live in the graph's ``GraphMemory`` (which the whole
    memory rework converges on), and this class supplies the two things
    that are genuinely its own -- a budget *stated per node* rather than
    by admission, and a device reading taken between steps rather than
    on demand.

    What that buys, concretely: one place decides which resident goes
    first, and it is the same place the lease policy (MEM-06) asks, so a
    trainer's reactive between-steps offloading and a downstream node's
    ``memory.request()`` cannot disagree about who is cheapest to lose.
    The measured reload cost that ordering uses is the same measurement
    in both.

    What it deliberately keeps, because merging them would change
    behaviour for existing trainers:

    * **the ceiling.** ``ResourceBudget``'s ``vram_budget_mb -
      vram_reserve_mb`` is what this handle enforces, and it is the
      *node's* number. The run's own ``GraphMemory.budget_mb`` is
      admission's, a different quantity that exists only when the run
      stated one; a handle is constructed by a node, long before or
      outside any particular run, so the usable ceiling is what this
      handle hands to the ``GraphMemory`` it owns.
    * **the measurement.** Enforcement reads
      ``DeviceContext.memory_stats()['reserved_mb']`` -- the allocator's
      real reserved bytes, which cover everything this process holds,
      not just this handle's declared footprints. Declared footprints
      are what the lease API accounts with; they are not a substitute
      for the real reading, and swapping one for the other here would
      stop enforcing anything (every resident's ``footprint_bytes()`` is
      best-effort by contract, and real ones report 0 while resident).

    The three registration states map onto ``GraphMemory``'s resident
    record exactly: ``never`` is pinned, ``offloadable`` and
    ``sacrificable`` are both evictable and differ only in priority
    band, which is what makes ``ensure_loaded``'s demand-driven pass
    prefer an offloadable resident and fall back to a sacrificable one
    only when nothing cheaper can make room. Their semantics are
    unchanged -- ``register()``'s own docstring on the ABC above is
    still the authoritative statement of them.
    """

    def __init__(self, budget: ResourceBudget, device: str,
                 device_ctx: DeviceContext | None = None):
        self._budget = budget
        # device_ctx: real DeviceContext.for_device(device) by default (unchanged
        # behavior) -- overridable so a test can inject a fake one that reports
        # scripted memory_stats() without real XPU/CUDA hardware, the same explicit-
        # injection posture this project uses everywhere else (no singletons, per
        # the root README's Goals section, goal 3).
        self._device_ctx = device_ctx or DeviceContext.for_device(device)
        usable_mb = budget.vram_budget_mb - budget.vram_reserve_mb
        # This handle's own memory authority, whose budget is the usable
        # ceiling above -- the same number _make_room() enforces against,
        # so the accounting and the ordering share one bound.
        self._memory = GraphMemory(budget_mb=usable_mb)
        # The resident lifecycle and the offloaded set are the
        # GraphMemory's, aliased rather than copied: two owners of "is
        # this resident offloaded" is exactly the desync that would let
        # the accounting and the eviction disagree. These two names are
        # kept because they are what the trainer-facing surface (and the
        # existing tests) reach for.
        self._coordinator = self._memory._coordinator
        self._offloaded = self._memory._offloaded
        # The candidate sets by state. A list, in registration order,
        # because that order is the final tiebreak and is not something
        # to borrow incidentally from a hash table.
        self._offloadable: list[str] = []
        self._sacrificable: list[str] = []

    def register(self, name: str, resident: DeviceResident, offloadable: bool = False,
                 sacrificable: bool = False) -> None:
        # ``never`` (neither flag) is pinned; the two movable states are
        # both evictable and differ only in priority band, so a
        # demand-driven pass tries an offloadable resident before a
        # sacrificable one. See this class's own docstring.
        pinned = not (offloadable or sacrificable)
        priority = _SACRIFICABLE_PRIORITY if sacrificable else _OFFLOADABLE_PRIORITY
        self._memory.register_resident(
            name, resident, pinned=pinned, priority=priority)
        if offloadable:
            self._offloadable.append(name)
        if sacrificable:
            self._sacrificable.append(name)

    def before_step(self, step: int) -> None:
        self._make_room(exclude=(), demand_driven=False)

    def ensure_loaded(self, name: str) -> None:
        if name in self._offloaded:
            self._memory.move(name, offload=False)
            # Defensive, see _make_room()'s own comment on the matching offload-side
            # call below for the full reasoning -- same "don't trust a memory_stats()
            # snapshot taken right after a transfer without an explicit sync first"
            # posture, here for the reload direction: _make_room() below is about to
            # read reserved_mb again (exclude=(name,) still runs it), and whatever
            # calls ensure_loaded() is about to use `name` itself immediately after
            # this returns.
            self._device_ctx.synchronize()
        self._make_room(exclude=(name,), demand_driven=True)

    def release(self, name: str) -> None:
        if name not in self._offloadable:
            raise ValueError(
                f"release({name!r}): not registered offloadable -- register() with "
                f"offloadable=True first if this resident is actually safe to move "
                f"between uses. Refusing rather than silently no-op-ing: a caller "
                f"asking to release something it was told is unsafe to move is a real "
                f"bug worth surfacing, not hiding."
            )
        if name in self._offloaded:
            return
        self._memory.move(name, offload=True)
        # Same defensive reasoning as _make_room()'s own offload-side call below --
        # this is the same operation (move it off, then synchronize()), just
        # triggered deterministically by a caller instead of reactively by measured
        # pressure.
        self._device_ctx.synchronize()

    def usable_budget_mb(self) -> Optional[float]:
        return self._memory.budget_mb

    def _make_room(self, exclude: tuple[str, ...],
                   demand_driven: bool = False) -> None:
        """Relieve measured pressure by moving residents, in policy order.

        Kept on this handle rather than delegated wholesale to
        ``GraphMemory`` because it measures the *device* between steps --
        see this class's docstring for why that is not the lease API's
        declared-footprint accounting. The part that is purely a policy
        decision, which resident goes next, is ``GraphMemory``'s and is
        shared with every other user of it.

        ``demand_driven`` separates the two callers, and the candidate
        set carries it. ``before_step()`` offers only the *offloadable*
        residents, because releasing something between steps is free
        precisely when its idle window is known, and a resident about
        to be used -- a base model -- is not between uses at all.
        ``ensure_loaded()`` offers every evictable resident, because
        something needed the room *now*. Measured cost of getting that
        wrong: a UNet round trip is 2,594 ms against a 2,430 ms step, so
        ``before_step()`` doing it every step is a 2.1x slowdown.

        Re-measures after each individual move rather than estimating
        from footprint_bytes() and moving everything that adds up to
        enough up front: one real number from the allocator beats a
        predicted one, and stopping the moment it's enough avoids moving
        (and later having to reload) more than the pressure actually
        required.

        Raises when self._budget.strict and usage is still over budget
        once every candidate, currently-loaded resident (outside
        ``exclude``) has been moved -- see ResourceBudget.strict's own
        docstring for why. Default strict=False keeps this method's
        previous behavior exactly (return once the candidates are
        exhausted, over budget or not) -- existing callers/tests see no
        behavior change unless they opt in."""
        stats = self._device_ctx.memory_stats()
        if stats is None:
            return
        usable_mb = self._memory.budget_mb or 0.0
        candidates = None if demand_driven else tuple(self._offloadable)
        while stats["reserved_mb"] > usable_mb:
            victim = self._memory.next_to_evict(
                candidates=candidates, exclude=exclude)
            if victim is None:
                break
            self._memory.move(victim.name, offload=True)
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