"""GraphMemory -- what one graph run may use on the device (MEM-05).

Built by the worker (and the in-process gateway) from the spawn
arguments *before anything loads* -- before the graph JSON is read,
before discovery, before torch -- so a bad number is refused while
there is nothing loaded yet, and every later consumer asks this object
instead of re-deriving the answer.

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
``_budget`` on purpose: the child imports this module before anything
else, so it must not depend on the domain layer. It matters here more
than at the edge: a NaN grant would make
every ``free < grant`` comparison False, so the physical check would
pass on garbage -- the check itself is only as honest as this
constructor.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

#: Physical-check outcomes (MEM-05 #2). Three named states, not a
#: bool: "unknown" is a thing that happened (and gets said), not the
#: absence of an answer.
CHECK_OK = "ok"
CHECK_SHORTFALL = "shortfall"
CHECK_UNKNOWN = "unknown"


def _size_or_raise(name: str, value) -> float:
    """``_size`` for a size that must exist: shrinking to "unknown" has no meaning, so it is refused by name."""
    if value is None:
        raise ValueError(f"{name}: a size is required, not None")
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




class GraphMemory:
    """What one graph run may use on the device: the two numbers from admission (``grant_mb``, ``budget_mb``) and the
    pre-load physical check against the real card.

    Thread-safe: the worker may shrink the grant while another thread reads it.
    """

    def __init__(
        self, *, grant_mb: float | None = None, budget_mb: float | None = None
    ) -> None:
        self._grant_mb = _size("grant_mb", grant_mb)
        self._budget_mb = _size("budget_mb", budget_mb)
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

    def shrink_to(self, grant_mb: float) -> None:
        """Lower this run's grant (and the budget with it) to ``grant_mb``.

        Only ever lowers: a run that asked for "auto" was handed a
        ceiling, and the card can turn out to have less free than that
        ceiling (the desktop, ComfyUI). Shrinking to what is really free
        lets it run inside its real share rather than refusing a run that
        would have fit. Raising would be inventing memory, so a larger
        number is ignored. Called once, by the worker, before anything
        loads; the budget never exceeds the grant afterwards.
        """
        grant = _size_or_raise("grant_mb", grant_mb)
        with self._lock:
            if self._grant_mb is None or grant < self._grant_mb:
                self._grant_mb = grant
            if self._budget_mb is None or self._budget_mb > grant:
                self._budget_mb = grant













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
