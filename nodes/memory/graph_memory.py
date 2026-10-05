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

import math
from dataclasses import dataclass

#: Physical-check outcomes (MEM-05 #2). Three named states, not a
#: bool: "unknown" is a thing that happened (and gets said), not the
#: absence of an answer.
CHECK_OK = "ok"
CHECK_SHORTFALL = "shortfall"
CHECK_UNKNOWN = "unknown"


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


class GraphMemory:
    """The two numbers one run is allowed, with their unknowns named.

    See the module docstring. Deliberately small: the resident registry
    and the lease policy are added here as the rework lands, and the
    physical check and allocator backstop hang off these two properties.
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
