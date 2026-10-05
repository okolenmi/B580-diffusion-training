"""MemoryLedger -- the server-side admission controller.

Level 1 of the two-level design: admission between processes. The server
owns it. A `MemoryLedger` in `backend/application` decides, once, at
start, inside the existing start lock, for every GPU-using child: graph
executions, dataset tasks, and the installer's device probe. No
preemption, no mid-run negotiation.

The ledger holds no state of its own beyond the lock; on startup it is
rebuilt from rows of adopted/running children. Capacity is
`total_mb - foreign_reserve_mb` (the desktop and other applications; the
measured gap was ~950 MB desktop + ~600 MB overhead on a 12,216 MB card).

A refusal explains itself: capacity, foreign reserve, every holder and
its size, what is free, what was asked, and what would fit.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


#: Extra VRAM held above an observed peak, in MB. Same as
#: `device_reservations.OBSERVED_PILLOW_MB` and `peak_record.DEFAULT_PILLOW_MB`.
OBSERVED_PILLOW_MB = 150.0

#: Default foreign reserve: the desktop and other applications. Measured
#: on the B580: ~950 MB desktop + ~600 MB overhead on a 12,216 MB card.
DEFAULT_FOREIGN_RESERVE_MB = 1024.0

#: Default per-process overhead: the driver reading minus allocator
#: reserved for an idle context. Measured on the B580: ~600 MB.
DEFAULT_PROCESS_OVERHEAD_MB = 600.0


def _bad_demand_reason(demand_mb) -> str | None:
    """Why this demand cannot be claimed, or None when it can.

    MEM-03H-01, ledger layer -- the third of three independent checks
    (API 422, domain ValueError, refusal here): ``nan`` used to be
    granted, which made ``held_mb()`` and ``free_mb()`` NaN, after which
    ``demand > NaN`` was False and *every* later claim was granted too;
    ``-9000`` was granted as if it freed space; zero was accepted as a
    claim. A demand is a size: a finite number, not a boolean, above
    zero. The reason always names the value (task rule 6).
    """
    if isinstance(demand_mb, bool) or not isinstance(demand_mb, (int, float)):
        return f"asked for {demand_mb!r}, which is not a number"
    try:
        number = float(demand_mb)
    except (OverflowError, ValueError):  # an int too big to be a size
        return f"asked for {demand_mb!r}, which is too large to be a device size"
    if not math.isfinite(number) or number <= 0:
        return (
            f"asked for {number} MB, which is not a finite number above zero"
        )
    return None


def _asked(demand_mb) -> float | str:
    """The ask, in the form a refusal is allowed to carry.

    The number when it is one, its repr when it is not: a refusal's
    breakdown goes into a JSON response, and Starlette serializes with
    ``allow_nan=False`` -- a breakdown carrying NaN would raise where a
    409 was owed (the same 500 ``presentation/responses.py`` documents
    for a frame that carries one).
    """
    if isinstance(demand_mb, bool) or not isinstance(demand_mb, (int, float)):
        return repr(demand_mb)
    try:
        number = float(demand_mb)
    except (OverflowError, ValueError):
        return repr(demand_mb)
    return number if math.isfinite(number) else repr(demand_mb)


@dataclass(frozen=True, slots=True)
class Grant:
    """A successful reservation. The child may now use up to `mb` device MB."""

    owner: str
    mb: float
    #: Whether this is an exploratory exclusive run (unknown demand).
    exploratory: bool = False


@dataclass(frozen=True, slots=True)
class Refusal:
    """A refusal. Explains itself: capacity, foreign reserve, every holder
    and its size, what is free, what was asked, and what would fit."""

    owner: str
    #: What was asked. A number when the ask was one; its repr when it
    #: was not (``"nan"``, ``"'lots'"``) -- see ``_asked``, which keeps
    #: the breakdown JSON-serializable for the response it lands in.
    requested_mb: float | str
    capacity_mb: float
    foreign_reserve_mb: float
    free_mb: float
    holders: dict[str, float]
    reason: str

    def breakdown(self) -> dict:
        """The full breakdown for the API response."""
        return {
            "owner": self.owner,
            "requested_mb": self.requested_mb,
            "capacity_mb": self.capacity_mb,
            "foreign_reserve_mb": self.foreign_reserve_mb,
            "free_mb": self.free_mb,
            "holders": dict(self.holders),
            "reason": self.reason,
            "what_would_fit": self.free_mb,
        }


@dataclass
class _Holder:
    """One holder's claim, internal to the ledger."""

    owner: str
    mb: float
    exploratory: bool = False


class MemoryLedger:
    """Every claim on the device, and what is left.

    Not a singleton: the device is the scope, so a caller constructs one
    per device and passes it where it is needed. The ledger holds no state
    of its own beyond the lock; on startup it is rebuilt from rows of
    adopted/running children.

    Thread-safe within a process (a lock around read-modify-write). The
    cross-process guarantee comes from the server being the only writer:
    children ask the server to reserve, they do not write the ledger
    themselves.
    """

    def __init__(
        self,
        *,
        total_mb: float,
        foreign_reserve_mb: float = DEFAULT_FOREIGN_RESERVE_MB,
        process_overhead_mb: float = DEFAULT_PROCESS_OVERHEAD_MB,
    ) -> None:
        self._total_mb = total_mb
        self._foreign_reserve_mb = foreign_reserve_mb
        self._process_overhead_mb = process_overhead_mb
        self._holders: dict[str, _Holder] = {}
        # RLock, not Lock: reserve() calls held_mb() while holding the lock,
        # and a non-reentrant Lock would deadlock on that nested acquire.
        self._lock = threading.RLock()

    @property
    def capacity_mb(self) -> float:
        """What is available for GPU-using children."""
        return self._total_mb - self._foreign_reserve_mb

    @property
    def total_mb(self) -> float:
        return self._total_mb

    @property
    def foreign_reserve_mb(self) -> float:
        return self._foreign_reserve_mb

    @property
    def process_overhead_mb(self) -> float:
        return self._process_overhead_mb

    def reserve(
        self,
        owner: str,
        demand_mb: float,
        *,
        exploratory: bool = False,
    ) -> Grant | Refusal:
        """Reserve for `owner`, or refuse.

        `demand_mb` is in **device MB** (allocator budget + process
        overhead). The ledger sums device-visible claims against the
        device-visible capacity.

        `exploratory` marks an unknown-demand run that claims all free
        capacity. It is admitted only if nothing else holds the card.

        A demand that is not a finite number above zero is refused
        whatever the mode (MEM-03H-01), before any state is touched:
        the reason names the value.
        """
        with self._lock:
            free = self.capacity_mb - self.held_mb()

            bad = _bad_demand_reason(demand_mb)
            if bad is not None:
                return Refusal(
                    owner=owner,
                    requested_mb=_asked(demand_mb),
                    capacity_mb=self.capacity_mb,
                    foreign_reserve_mb=self._foreign_reserve_mb,
                    free_mb=free,
                    holders={h.owner: h.mb for h in self._holders.values()},
                    reason=bad,
                )

            if exploratory:
                # An exploratory run claims all free capacity, but only if
                # nothing else holds the card.
                if self._holders:
                    return Refusal(
                        owner=owner,
                        requested_mb=demand_mb,
                        capacity_mb=self.capacity_mb,
                        foreign_reserve_mb=self._foreign_reserve_mb,
                        free_mb=free,
                        holders={h.owner: h.mb for h in self._holders.values()},
                        reason=(
                            "exploratory run refused: other holders occupy the card"
                        ),
                    )
                # Claim all free capacity
                self._holders[owner] = _Holder(
                    owner=owner, mb=free, exploratory=True
                )
                return Grant(owner=owner, mb=free, exploratory=True)

            if demand_mb > free:
                return Refusal(
                    owner=owner,
                    requested_mb=demand_mb,
                    capacity_mb=self.capacity_mb,
                    foreign_reserve_mb=self._foreign_reserve_mb,
                    free_mb=free,
                    holders={h.owner: h.mb for h in self._holders.values()},
                    reason=(
                        f"asked for {demand_mb:.0f} MB but only {free:.0f} MB "
                        f"is free on the device"
                    ),
                )

            self._holders[owner] = _Holder(owner=owner, mb=demand_mb)
            return Grant(owner=owner, mb=demand_mb)

    def rename(self, old_owner: str, new_owner: str) -> None:
        """Re-key a claim in place, atomically.

        A claim has to be taken *before* the row it belongs to exists
        (a refusal must not leave a row behind), but the row's id is
        only known after the insert -- so the claim starts under a
        provisional owner and is renamed to the row-derived one while
        still inside the start lock. Release paths derive owners from
        rows, so the rename is what makes ``graph:<id>`` / ``task:<id>``
        the one naming convention.

        Idempotent: renaming an owner that holds nothing does nothing
        (the same posture as `release`). If `new_owner` already holds a
        claim that is a bug in the caller -- ids are unique -- and the
        newer claim wins, so the ledger never over-counts; it may
        momentarily under-count a claim that should not have collided.
        """
        with self._lock:
            holder = self._holders.pop(old_owner, None)
            if holder is not None:
                holder.owner = new_owner
                self._holders[new_owner] = holder

    def release(self, owner: str) -> None:
        """Drop a holder's claim. Idempotent, so a failed start can clean up."""
        with self._lock:
            self._holders.pop(owner, None)

    def held_mb(self) -> float:
        """Total claimed by all holders."""
        with self._lock:
            return sum(h.mb for h in self._holders.values())

    def held_by(self, owner: str) -> float:
        """How much one holder claims. 0 for a holder that does not exist."""
        with self._lock:
            holder = self._holders.get(owner)
            return holder.mb if holder else 0.0

    def free_mb(self) -> float:
        """What is left for a new holder."""
        with self._lock:
            return self.capacity_mb - self.held_mb()

    def snapshot(self) -> dict:
        """The current state, for /health and the UI."""
        with self._lock:
            return {
                "total_mb": self._total_mb,
                "foreign_reserve_mb": self._foreign_reserve_mb,
                "process_overhead_mb": self._process_overhead_mb,
                "capacity_mb": self.capacity_mb,
                "held_mb": self.held_mb(),
                "free_mb": self.free_mb(),
                "holders": {
                    h.owner: {"mb": h.mb, "exploratory": h.exploratory}
                    for h in self._holders.values()
                },
            }

    def rebuild_from_rows(self, rows: list[dict]) -> None:
        """Rebuild the ledger from persisted rows on startup.

        Each row is a dict with `owner` and `mb` keys. The ledger holds no
        state of its own beyond the lock; on startup it is rebuilt from
        rows of adopted/running children.

        A row whose claim is not a finite number above zero is skipped
        and logged (MEM-03H-01): such a claim never held anything worth
        counting, and rebuilding it would poison the very totals this
        rebuild exists to reproduce -- one NaN row and every later
        admission decision is made against a NaN ledger.
        """
        with self._lock:
            self._holders.clear()
            for row in rows:
                owner = row["owner"]
                mb = row["mb"]
                bad = _bad_demand_reason(mb)
                if bad is not None:
                    logger.warning(
                        "rebuild: skipping the claim of %r -- %s "
                        "(a row that states no usable claim holds nothing)",
                        owner, bad,
                    )
                    continue
                self._holders[owner] = _Holder(owner=owner, mb=float(mb))
