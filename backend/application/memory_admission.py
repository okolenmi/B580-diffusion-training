"""Memory admission -- how start paths and release paths use the ledger.

Level 1 wiring for MEM-03. Three things live here so every caller says
them the same way:

* **Owner naming.** A claim is durable only through its row, so claims
  are keyed ``graph:<id>`` / ``task:<id>`` -- what every release path
  (finish, reconcile, sweeper, wipe) can derive from the row it holds.
  A claim must be taken *before* the row exists (a refusal writes no
  row), so it starts under ``<kind>:pending:<uuid>`` and the start path
  renames it once the id binds. One naming convention, one source: the
  helpers below, used by reserve *and* rebuild *and* release.
* **`LedgerProvider`** -- the container's one ledger, built on first
  use from the cached device probe and rebuilt from the unfinished rows
  that carry a claim, so a restart reproduces the same held total.
  First use, not boot: the device total may only become known after the
  installer has run, and a provider answering ``None`` refuses every
  start explicitly -- while it answers ``None`` no claim can exist, so
  building the ledger later starts from an empty, correct state (rule 2:
  a claim is never recorded unchecked).
* **`admit()`** -- reserve-or-raise 409 ``memory_unavailable`` with the
  refusal's full breakdown (rule 6: capacity, foreign reserve, every
  holder and its size, free, asked), plus the explicit
  device-total-unknown refusal for a container with no ledger, plus
  (MEM-03H-02) the refusal for a graph whose declared ``vram_min_mb``
  floor the device cannot honor.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import NoReturn
from uuid import uuid4

from .errors import MemoryUnavailableError
from .memory_ledger import Grant, MemoryLedger, Refusal
from .ports.dataset_tasks import DatasetTasks
from .ports.environment import DeviceProbe
from .ports.graph_execution_repository import GraphExecutionRepository

logger = logging.getLogger(__name__)

#: What every caller holds: the container's ledger, or None while the
#: device total is unknown. A plain callable so tests can pass a lambda.
LedgerSource = Callable[[], MemoryLedger | None]


def graph_owner(execution_id: int) -> str:
    """The row-derived claim key for a graph execution."""
    return f"graph:{execution_id}"


def task_owner(task_id: int) -> str:
    """The row-derived claim key for a dataset task."""
    return f"task:{task_id}"


def pending_owner(kind: str) -> str:
    """A provisional claim key, taken before the row (and its id) exists.

    Unique per reservation, so two racing starts never share a
    provisional; renamed to the row-derived owner under the start lock
    once the id binds. Nothing derives owners from these -- they live
    only between reserve and rename.
    """
    return f"{kind}:pending:{uuid4().hex}"


def release(source: LedgerSource | None, owner: str) -> None:
    """Hand one claim back, if this container has a ledger to hand it to.

    The single shape every release path uses (watcher finish, reconcile,
    sweeper, history wipe, the task exit-waiter): a missing source or a
    missing ledger means no claim could exist -- a start without a
    ledger refuses before it reserves -- and the ledger's own
    ``release`` is idempotent, so double release is safe.
    """
    if source is None:
        return
    ledger = source()
    if ledger is not None:
        ledger.release(owner)


def admit(
    ledger: MemoryLedger | None,
    owner: str,
    demand_mb: float,
    *,
    exploratory: bool,
    what: str,
    vram_min_mb: float = 0.0,
) -> Grant:
    """Reserve for `owner`, or refuse with the whole breakdown.

    The refusals are 409 ``memory_unavailable``:

    * no ledger -- the device total is unknown, so nothing can be
      sized. An explicit UNKNOWN refusal, never an unchecked claim
      (task rule 2). Checked first: while the total is unknown no
      floor question can be answered either.
    * a `Refusal` from the ledger -- the breakdown in ``details`` names
      capacity, foreign reserve, every holder and its size, what is
      free, what was asked (rule 6).
    * the graph's declared floor (MEM-03H-02, ``vram_min_mb`` in
      allocator MB; 0 means no floor, which is every caller but the
      graph start path): an exploratory claim would be all free space,
      so refuse when free cannot cover ``vram_min_mb + process
      overhead``; a stated or observed claim must itself reach the
      floor, or the run would start below the minimum it declared.
      Both carry the same breakdown, and the reason names the floor
      and the number it was measured against.
    """
    if ledger is None:
        raise MemoryUnavailableError(
            f"{what} cannot be admitted: the device total is unknown, so no "
            "claim can be sized (the probe has not reported a card)",
            details={
                "reason": "device_total_unknown",
                "requested_mb": demand_mb,
                "holders": {},
            },
        )

    if vram_min_mb > 0:
        # One snapshot: the numbers the refusal quotes must describe
        # the same moment (the free that failed and the holders that
        # left it free come from one read).
        snapshot = ledger.snapshot()
        floor_mb = vram_min_mb + snapshot["process_overhead_mb"]
        free = snapshot["free_mb"]
        below = free < floor_mb if exploratory else demand_mb < floor_mb
        if below:
            _refuse_below_floor(
                snapshot=snapshot,
                owner=owner,
                what=what,
                vram_min_mb=vram_min_mb,
                floor_mb=floor_mb,
                demand_mb=demand_mb,
                free_mb=free,
                exploratory=exploratory,
            )

    outcome = ledger.reserve(owner, demand_mb, exploratory=exploratory)
    if isinstance(outcome, Refusal):
        names = ", ".join(sorted(outcome.holders)) or "nobody"
        raise MemoryUnavailableError(
            f"{what} cannot be admitted: {outcome.reason} "
            f"(capacity {outcome.capacity_mb:.0f} MB, free "
            f"{outcome.free_mb:.0f} MB, held by {names})",
            details=outcome.breakdown(),
        )
    return outcome


def _refuse_below_floor(
    *,
    snapshot: dict,
    owner: str,
    what: str,
    vram_min_mb: float,
    floor_mb: float,
    demand_mb: float,
    free_mb: float,
    exploratory: bool,
) -> NoReturn:
    """Raise the floor refusal, carrying the ledger's own breakdown.

    Built through the ``Refusal`` dataclass on purpose: a floor refusal
    then has exactly the keys any other refusal has (rule 6), instead of
    a hand-assembled dict that could drift from them.
    """
    holders = {
        name: claim["mb"] for name, claim in snapshot["holders"].items()
    }
    names = ", ".join(sorted(holders)) or "nobody"
    needs = (
        f"needs at least {vram_min_mb:g} MB (vram_min_mb) plus "
        f"{snapshot['process_overhead_mb']:g} MB of process overhead "
        f"= {floor_mb:g} MB"
    )
    reason = (
        f"{needs}, but only {free_mb:g} MB is free"
        if exploratory
        else f"{needs}, but its demand is only {demand_mb:g} MB"
    )
    refusal = Refusal(
        owner=owner,
        requested_mb=floor_mb,
        capacity_mb=snapshot["capacity_mb"],
        foreign_reserve_mb=snapshot["foreign_reserve_mb"],
        free_mb=free_mb,
        holders=holders,
        reason=reason,
    )
    raise MemoryUnavailableError(
        f"{what} cannot be admitted: {reason} "
        f"(capacity {snapshot['capacity_mb']:.0f} MB, free {free_mb:.0f} MB, "
        f"held by {names})",
        details=refusal.breakdown(),
    )


class LedgerProvider:
    """Builds the container's one `MemoryLedger` on first use.

    Single-flight: concurrent first calls construct once. Construction
    immediately rebuilds the ledger from the unfinished rows that carry
    a ``reserved_mb`` claim, so the held total after a restart equals
    what the rows say -- reconcile then releases the rows that turn out
    to be debris, and the supervisor releases the adopted ones when
    their children finish.

    A probe reading with no total (no card, stack not installed yet)
    leaves the provider answering ``None``; it re-probes on the next
    call, so the first start after a successful install builds the
    ledger without a server restart.
    """

    def __init__(
        self,
        *,
        probe: DeviceProbe,
        graph_executions: GraphExecutionRepository,
        dataset_tasks: DatasetTasks,
        foreign_reserve_mb: float,
        process_overhead_mb: float,
    ) -> None:
        self._probe = probe
        self._graph_executions = graph_executions
        self._dataset_tasks = dataset_tasks
        self._foreign_reserve_mb = foreign_reserve_mb
        self._process_overhead_mb = process_overhead_mb
        self._ledger: MemoryLedger | None = None
        self._building = False
        self._lock = threading.Lock()

    def __call__(self) -> MemoryLedger | None:
        if self._ledger is not None:
            return self._ledger
        with self._lock:
            if self._ledger is not None:
                return self._ledger
            if self._building:
                # A probe report can re-enter (health asking while a
                # start builds): answer None rather than recurse. No
                # claim exists yet either way; the next call builds.
                return None
            self._building = True
            try:
                report = self._probe.report()
            except Exception:  # noqa: BLE001 -- a failed probe is an
                # admission answer, not a crash: unknown total, refuse.
                logger.exception("device probe failed while sizing the ledger")
                return None
            finally:
                self._building = False
            total = report.total_memory_mb
            if total is None or total <= 0:
                return None
            ledger = MemoryLedger(
                total_mb=total,
                foreign_reserve_mb=self._foreign_reserve_mb,
                process_overhead_mb=self._process_overhead_mb,
            )
            ledger.rebuild_from_rows(self._claim_rows())
            self._ledger = ledger
            return ledger

    def _claim_rows(self) -> list[dict]:
        """Unfinished rows that carry a claim, in ledger owner form."""
        rows: list[dict] = []
        for execution in self._graph_executions.list_unfinished():
            if execution.reserved_mb and execution.id is not None:
                rows.append({
                    "owner": graph_owner(execution.id),
                    "mb": execution.reserved_mb,
                })
        for task in self._dataset_tasks.list_unfinished():
            if task.reserved_mb:
                rows.append({"owner": task_owner(task.id), "mb": task.reserved_mb})
        return rows
