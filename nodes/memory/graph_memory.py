"""GraphMemory -- what one graph run may use on the device (MEM-05).

Built by the worker (and the in-process gateway) from the spawn
arguments *before anything loads* -- before the graph JSON is read,
before discovery, before torch -- so a bad number is refused while
there is nothing loaded yet, and every later consumer asks this object
instead of re-deriving the answer. Nodes reach it through
``ExecutionContext.memory``.

Two numbers, each optional but never silently:

* ``grant_mb`` -- the device-MB claim admission held for this run. The
  ledger reserved it, but the ledger does not know foreign users
  (ComfyUI, the desktop), which is what the physical check compares it
  against: the card's real free memory.
* ``budget_mb`` -- the allocator MB this process may use: the run's
  stated/observed demand, the input to the allocator's per-process
  fraction backstop.

``None`` on either means *never supplied* -- an explicit UNKNOWN the
consumer names, never a zero to guess with (a zero claim would read as
"fits everything", which is the opposite of "nobody told me", task
rule 2).

The number checks mirror ``backend/domain/memory_settings.py``'s
``_budget`` on purpose: nodes/ never imports backend/ (task rule 9), so
the domain's validation is reimplemented at this layer instead of
shared. It matters here more than at the edge: a NaN grant would make
every ``free < grant`` comparison False, so the physical check would
pass on garbage -- the check itself is only as honest as this
constructor.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from .handle import DeviceResident

logger = logging.getLogger(__name__)

#: Physical-check outcomes (MEM-05 #2). Three named states, not a
#: bool: "unknown" is a thing that happened (and gets said), not the
#: absence of an answer.
CHECK_OK = "ok"
CHECK_SHORTFALL = "shortfall"
CHECK_UNKNOWN = "unknown"

#: Lease/eviction decision names (MEM-06). Every one of these is both a
#: log line and a telemetry record (``report_memory_event``), so a run's
#: memory decisions are readable after the fact rather than inferred.
DECISION_GRANT = "lease_granted"
DECISION_EVICTED = "lease_evicted"
DECISION_RESTORED = "lease_restored"
DECISION_DENIED = "lease_denied"


class MemoryRequestDenied(RuntimeError):
    """A lease request could not be satisfied and nothing was changed.

    Carries the three numbers a caller needs to act on it (task rule 6:
    a refusal that cannot be quoted at is just a "no"):

    * ``needed_mb`` -- what the caller asked for,
    * ``available_mb`` -- what the budget could actually give after
      every evictable resident was moved,
    * ``evictable`` -- the names, by policy order, of the residents
      that *were* candidates. Empty means nothing was movable, which is
      a different answer from "everything movable was not enough", and
      the caller may want to register more or raise its own budget.

    Raised only after every evicted resident has been reloaded, so the
    graph is exactly as it was before the request.
    """

    def __init__(self, needed_mb: float, available_mb: float,
                 evictable: list[str]) -> None:
        super().__init__(
            f"memory: cannot grant {needed_mb:.0f} MB -- only "
            f"{available_mb:.0f} MB is available inside this run's budget, "
            f"and evicting every candidate "
            f"({', '.join(evictable) if evictable else 'none registered'}) "
            f"was not enough"
        )
        self.needed_mb = needed_mb
        self.available_mb = available_mb
        self.evictable = list(evictable)


@dataclass
class Resident:
    """One device-resident this graph may evict to satisfy a lease.

    The eviction-relevant facts, nothing else. Kept here rather than
    inferred from the ``DeviceResident`` because three of the four
    (``priority``, the measured reload cost, the last-used stamp) are
    things only the memory authority knows as it moves things around.

    ``reload_cost_ms`` starts unknown and is measured the first time
    this resident is actually reloaded. Unknown sorts *last*: evicting
    something whose restore cost has never been measured would be
    guessing, and the whole ordering exists to avoid guessing.
    """

    name: str
    resident: DeviceResident
    pinned: bool = False
    priority: int = 0
    #: The device footprint this resident holds right now, MB. Zero
    #: while it is offloaded (its host-RAM copy does not count -- see
    #: DeviceResident.footprint_bytes' own docstring).
    footprint_mb: float = 0.0
    offloaded: bool = False
    reload_cost_ms: float | None = None
    last_used: float = field(default_factory=time.monotonic)

    @property
    def evictable(self) -> bool:
        """Pinned residents never move, whatever else is true of them."""
        return not self.pinned

    def _eviction_key(self) -> tuple[int, float, float]:
        """Sort key: the order this resident is given up in.

        Exactly the policy order -- never ``pinned``; then ``priority``
        (least important first); then lowest measured reload cost per
        MB freed (cheapest to get back, so least disruptive to move);
        then least recently used (coldest first). Unmeasured reload
        cost is ``inf`` and therefore sorts last.
        """
        per_mb = (
            self.reload_cost_ms / self.footprint_mb
            if self.reload_cost_ms is not None and self.footprint_mb > 0
            else float("inf")
        )
        return (self.priority, per_mb, self.last_used)


def _size_or_raise(name: str, value) -> float:
    """``_size`` for a size that must exist: a request cannot be unknown.

    ``_size`` passes ``None`` through (the grant/budget case, where
    unknown is a legitimate state the consumer names). A lease request
    is not: ``request(None)`` has no meaning to guess at, so it is
    refused by name here rather than silently treated as zero.
    """
    if value is None:
        raise ValueError(f"{name}: a request must state a size, not None")
    resolved = _size(name, value)
    assert resolved is not None  # None cannot reach here; _size passed it
    return resolved


@dataclass(frozen=True)
class PhysicalCheck:
    """What the pre-load comparison against the real card found.

    Carries both numbers whenever either exists, because a refusal the
    caller cannot quote numbers at (task rule 6) is just a "no".
    """

    #: One of the ``CHECK_*`` constants.
    status: str
    #: The claim admission held, or None when this run's numbers never
    #: travelled to the child.
    grant_mb: float | None
    #: What the driver said was free, or None when the backend has no
    #: such notion (CPU) or the query failed.
    free_mb: float | None
    #: Why the status is ``CHECK_UNKNOWN``; "" otherwise.
    reason: str = ""

    @property
    def refused(self) -> bool:
        return self.status == CHECK_SHORTFALL

    def refusal_message(self) -> str:
        """The outcome text for a shortfall -- only meaningful when
        ``refused``, which by construction means both numbers exist.

        A refusal the caller cannot quote numbers at (task rule 6) is
        just a "no", so the message always carries the grant, what was
        actually free, and the party admission's ledger structurally
        cannot see."""
        return (
            f"memory: the card cannot give this run its granted "
            f"{self.grant_mb} MB -- only {self.free_mb} MB is free "
            f"(a foreign process such as ComfyUI or the desktop is not "
            f"in the ledger)"
        )


def _size(name: str, value) -> float | None:
    """Validate one memory size; ``None`` passes through as unknown.

    Rejects what a device size cannot be: a boolean (``True`` is an
    ``int`` in Python), a non-number, a non-finite number (NaN and
    infinity poison every comparison they join), a number past the
    float range (``10**400`` overflows into an ``OverflowError`` that
    would escape as a crash instead of a refusal), and zero or below --
    no claim is ``None``, not ``0``.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: {value!r} is not a number")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(
            f"{name}: {value!r} is too large to be a device size"
        ) from None
    if not math.isfinite(number):
        raise ValueError(f"{name}: {value!r} is not a finite number")
    if number <= 0:
        raise ValueError(f"{name}: {number} MB is not above zero")
    return number


class Lease:
    """What ``GraphMemory.request()`` hands the block it guards.

    The lease is the record of one granted request: how much it holds
    (so ``in_use`` counts it while the block runs) and which residents
    were moved to make room (so they can be put back on the way out).
    A caller mostly ignores it -- it exists so ``with ... as lease:``
    has something truthful to bind.
    """

    def __init__(self, memory: GraphMemory, mb: float, reason: str,
                 moved: list[Resident]) -> None:
        self._memory = memory
        self.mb = mb
        self.why = reason
        self._moved = moved

    def _restore(self) -> None:
        """Put back whatever was moved for this lease, and say so.

        Reloads in reverse of the order they were given up, so the
        resident evicted first (the cheapest to lose) comes back first
        -- and each reload's measured cost feeds the next request's
        eviction order, which is how the ordering gets better at its
        own job the longer a run goes.

        A reload that raises is logged and the rest still happen: one
        resident failing to come back must not leave the others
        stranded off the device.
        """
        if not self._moved:
            return
        total_ms = 0.0
        restored: list[str] = []
        for resident in reversed(self._moved):
            try:
                total_ms += self._memory._move(resident, offload=False)
                restored.append(resident.name)
            except Exception:  # noqa: BLE001 -- one resident must not
                # strand the others; the run continues and the reason
                # is on the record.
                logger.exception(
                    "reloading %r after a %s lease failed; it stays "
                    "offloaded and its owner will have to rebuild it",
                    resident.name, self.why,
                )
        logger.info(
            "memory request (%s) released %.0f MB; restored %s in %.0f ms",
            self.why, self.mb, ", ".join(restored) or "nothing", total_ms,
        )
        self._memory.report_memory_event(
            DECISION_RESTORED, self.mb, total_ms)

    def __repr__(self) -> str:
        return (f"Lease({self.mb:.0f} MB, why={self.why!r}, "
                f"moved={[r.name for r in self._moved]})")


class GraphMemory:
    """What one graph run may use on the device, and the authority for it.

    Two numbers from admission (``grant_mb``, ``budget_mb``), the
    pre-load physical check against the real card, and -- since MEM-06
    -- the lease policy: the residents this run may be asked to give up
    when something else needs the room, and the accounting that decides
    whether it can be.

    Thread-safe. Every lease decision takes ``_lock``, so two nodes
    asking at once are serialised rather than both independently
    deciding there was room and together overrunning the budget.
    """

    def __init__(
        self, *, grant_mb: float | None = None, budget_mb: float | None = None
    ) -> None:
        self._grant_mb = _size("grant_mb", grant_mb)
        self._budget_mb = _size("budget_mb", budget_mb)
        #: MEM-05 #4: where lease/eviction decisions are reported, as
        #: ``on_memory_event(event, mb, cost_ms)``. Set by the worker,
        #: which owns the event writer; None until wired. A graph that
        #: never runs under a worker (a unit test, a node building
        #: alone) simply has no listener -- see ``report_memory_event``.
        self.on_memory_event = None
        # MEM-06. Insertion-ordered because registration order is the
        # final tiebreak in the eviction policy, so it has to be a
        # property of the container and not left to a hash table.
        self._residents: dict[str, Resident] = {}
        self._active_leases: list[Lease] = []
        self._lock = threading.RLock()

    # -- the two numbers ----------------------------------------------------

    @property
    def grant_mb(self) -> float | None:
        """The device-MB claim admission held, or ``None`` (unknown)."""
        return self._grant_mb

    @property
    def budget_mb(self) -> float | None:
        """The allocator MB this process may use, or ``None`` (unknown)."""
        return self._budget_mb

    def report_memory_event(self, event: str, mb: float, cost_ms: float) -> None:
        """MEM-05 #4: report one lease/eviction decision to the worker.

        Called by the lease policy (MEM-06) on every decision, with
        the MB the decision granted, freed, or declined, and the
        measured cost in ms of getting there -- for a grant, the
        eviction cost paid to make room. The worker's listener writes
        it as a memory record; with no listener this is a no-op, and
        that is not an error: the decision's own log line is the
        policy's, and the record is for the UI.
        """
        if self.on_memory_event is not None:
            self.on_memory_event(event, mb, cost_ms)

    # -- MEM-06: the lease policy ------------------------------------------

    def register_resident(self, name: str, resident: DeviceResident, *,
                          pinned: bool = False, priority: int = 0) -> None:
        """Tell this graph about a device resident it may be asked to move.

        A trainer registers what it built (model, optimizer, text
        encoder) so a later node's ``request()`` has something it is
        allowed to take the room from. Registering loads and unloads
        nothing: the footprint is read once, here, because a resident
        that is offloaded holds none.

        ``pinned`` residents are never evicted, whatever else is true of
        them -- that is the "never" state, and it is opt-in precisely
        because a resident nobody can move is a resident that can make
        every later request fail. ``priority`` orders the rest: lower
        goes first.
        """
        with self._lock:
            if name in self._residents:
                raise ValueError(
                    f"register_resident({name!r}): already registered -- a "
                    f"second registration would orphan the first "
                    f"resident's footprint in the eviction accounting"
                )
            self._residents[name] = Resident(
                name=name,
                resident=resident,
                pinned=pinned,
                priority=priority,
                footprint_mb=resident.footprint_bytes() / (1024 ** 2),
                offloaded=False,
                last_used=time.monotonic(),
            )

    def touch(self, name: str) -> None:
        """Mark a resident as just used, so LRU eviction keeps its order.

        The alternative -- treating "loaded" as "in use" -- makes the
        last-used key meaningless for any resident that stays loaded
        across a whole run, which is most of them. Cheap, and only the
        callers that know they just used something need to call it.
        """
        with self._lock:
            resident = self._residents.get(name)
            if resident is not None:
                resident.last_used = time.monotonic()

    def in_use_mb(self) -> float:
        """What this graph currently holds: residents plus live leases.

        Not the device's own reading, and deliberately so: the question
        a lease asks is about *this run's* allocations, which is what
        the budget is a budget for. The device reading is the physical
        check's job, and it answers a different question.
        """
        with self._lock:
            return self._in_use_mb()

    def _in_use_mb(self) -> float:
        return (
            sum(r.footprint_mb for r in self._residents.values() if not r.offloaded)
            + sum(lease.mb for lease in self._active_leases)
        )

    @contextmanager
    def request(self, mb: float, *, why: str = ""):
        """Hold ``mb`` of this run's budget for the body of the block.

        ```
        with memory.request(2048.0, why="vae decode"):
            vae = load_vae()      # the room is guaranteed here
            ...
        ```

        Granted outright when ``budget - in_use >= mb``. Otherwise this
        run's evictable residents are moved, in policy order, until it
        is -- and if every candidate is moved and it still is not, the
        request is *denied*: ``MemoryRequestDenied``, carrying what was
        needed, what was available and who was movable, with every
        evicted resident reloaded first, so a refusal changes nothing.

        Two properties, both of which the tests pin because each is the
        whole point of the thing: a grant never leaves in-use above the
        budget, and the block's exit restores what was moved --
        including when the body raises, since the eviction is undone on
        the way out either way.
        """
        wanted = _size_or_raise("mb", mb)
        reason = why or "unspecified"
        with self._lock:
            moved = self._release_room(wanted, reason)
            lease = Lease(self, wanted, reason, moved)
            self._active_leases.append(lease)
        try:
            yield lease
        finally:
            with self._lock:
                if lease in self._active_leases:
                    self._active_leases.remove(lease)
                lease._restore()

    def _release_room(self, wanted_mb: float, reason: str) -> list[Resident]:
        """Make room for ``wanted_mb``; deny the request if it cannot be.

        Returns the residents it moved, so the lease can put them back.
        On failure it reloads everything it moved *before* raising --
        which is what makes "leaving state unchanged" true rather than
        aspirational.
        """
        budget = self._budget_mb
        if budget is None:
            # Unknown budget is not a zero budget (task rule 2). With no
            # ceiling stated there is nothing to enforce here, so the
            # request is granted and said once -- the physical check and
            # the allocator backstop are the enforcement on a run that
            # stated no budget.
            logger.warning(
                "memory request (%s, %.0f MB): no budget was stated for this "
                "run, so nothing can be checked and nothing is evicted",
                reason, wanted_mb,
            )
            self.report_memory_event(DECISION_GRANT, wanted_mb, 0.0)
            return []

        moved: list[Resident] = []
        cost_ms = 0.0
        if budget - self._in_use_mb() < wanted_mb:
            for resident in sorted(self._residents.values(),
                                   key=Resident._eviction_key):
                if not resident.evictable or resident.offloaded:
                    continue
                moved.append(resident)
                cost_ms += self._move(resident, offload=True)
                if budget - self._in_use_mb() >= wanted_mb:
                    break

        if budget - self._in_use_mb() < wanted_mb:
            for resident in reversed(moved):
                self._move(resident, offload=False)
            available = budget - self._in_use_mb()
            logger.warning(
                "memory request (%s, %.0f MB) denied: %.0f MB available "
                "after evicting every candidate",
                reason, wanted_mb, available,
            )
            self.report_memory_event(DECISION_DENIED, wanted_mb, 0.0)
            raise MemoryRequestDenied(
                needed_mb=wanted_mb,
                available_mb=available,
                evictable=sorted(
                    (r.name for r in self._residents.values() if r.evictable),
                    key=lambda name: self._residents[name]._eviction_key(),
                ),
            )

        if moved:
            logger.info(
                "memory request (%s, %.0f MB): evicted %s (%.0f MB) in "
                "%.0f ms to make room",
                reason, wanted_mb,
                ", ".join(r.name for r in moved),
                sum(r.footprint_mb for r in moved) or wanted_mb,
                cost_ms,
            )
            self.report_memory_event(DECISION_EVICTED, wanted_mb, cost_ms)
        self.report_memory_event(DECISION_GRANT, wanted_mb, cost_ms)
        return moved

    def _move(self, resident: Resident, *, offload: bool) -> float:
        """Offload or reload one resident; returns the elapsed ms.

        The reload is timed around the real call and the result stored
        on the resident, because the eviction policy orders by that
        measured cost and a guessed one would be exactly the guess the
        ordering exists to avoid. An offload returns 0.0: "reload cost"
        is not what an offload costs.
        """
        if offload:
            resident.resident.offload()
            resident.offloaded = True
            # The footprint is gone now; keeping it would leave the
            # accounting claiming memory the device no longer holds.
            resident.footprint_mb = 0.0
            resident.last_used = time.monotonic()
            return 0.0
        started = time.monotonic()
        resident.resident.reload()
        elapsed_ms = (time.monotonic() - started) * 1000.0
        resident.offloaded = False
        resident.footprint_mb = resident.resident.footprint_bytes() / (1024 ** 2)
        resident.reload_cost_ms = elapsed_ms
        resident.last_used = time.monotonic()
        return elapsed_ms

    def physical_check(self, device) -> PhysicalCheck:
        """MEM-05 #2: can the card actually give what admission granted?

        Run after the device context exists and before anything loads.
        ``device`` is anything with ``free_memory_mb()`` -- the real
        ``DeviceContext`` or a fake -- because admission's ledger knows
        this project's own claims and nothing at all about foreign
        users (ComfyUI, the desktop), and this comparison is the only
        place that gap is closed.

        Unknown, not guessed, on either side: no grant (this run's
        numbers never travelled) or no answer from the backend (CPU, a
        failed query) means no comparison exists -- the status says so
        and the caller continues with a warning, rather than trusting a
        zero that would either refuse everything or pass everything.
        A non-finite reading is the same unknown: NaN would make
        ``free < grant`` False, i.e. pass on garbage.
        """
        raw_free = device.free_memory_mb()
        if self._grant_mb is None:
            return PhysicalCheck(
                CHECK_UNKNOWN, None,
                None if raw_free is None else float(raw_free),
                reason="this run's grant was never supplied",
            )
        if raw_free is None:
            return PhysicalCheck(
                CHECK_UNKNOWN, self._grant_mb, None,
                reason="this device cannot report free memory",
            )
        free = float(raw_free)
        if not math.isfinite(free):
            return PhysicalCheck(
                CHECK_UNKNOWN, self._grant_mb, None,
                reason=f"the device reported a non-finite free reading ({raw_free!r})",
            )
        if free < self._grant_mb:
            return PhysicalCheck(CHECK_SHORTFALL, self._grant_mb, free)
        return PhysicalCheck(CHECK_OK, self._grant_mb, free)
