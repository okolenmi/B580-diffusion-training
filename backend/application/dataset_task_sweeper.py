"""DatasetTaskSweeper -- one definition of "this task row is dead".

Two situations make a dead task knowable, and both used to have their
own copy of the rule (``ListDatasetTasks`` swept on every read --
including rows of unrelated datasets -- and ``ReconcileDatasetTasks``
swept at startup with slightly different wording). The rule lives here
once, and the two callers say *when* rather than *how* (docs 08 S-03):

* a ``running`` row whose child is gone (OOM kill, a write that never
  landed): the gateway's liveness guard disproves the pid;
* a ``pending`` row with no pid older than ``STUCK_PENDING_SECONDS``
  that failed before its first progress write -- normally impossible,
  since ``StartDatasetTask`` records the pid inside its lock, so age is
  what makes sweeping it safe.

Rows that are genuinely finished are never touched: a task keeps its
counts, error text and terminal status forever (task history).
"""

from __future__ import annotations

import logging

from .ports.clock import Clock
from .ports.dataset_task_gateway import DatasetTaskGateway
from .ports.dataset_tasks import DatasetTasks, DatasetTask, TaskStatus

logger = logging.getLogger(__name__)

# A pending row with no pid older than this is debris from a crash
# between the insert and the pid bookkeeping.
STUCK_PENDING_SECONDS = 60.0

DEAD_PROCESS_NOTE = "task process is not running"
STUCK_PENDING_NOTE = "task never reported progress"


class DatasetTaskSweeper:
    def __init__(
        self,
        *,
        tasks: DatasetTasks,
        gateway: DatasetTaskGateway,
        clock: Clock,
    ) -> None:
        self._tasks = tasks
        self._gateway = gateway
        self._clock = clock

    def sweep(
        self,
        *,
        pidless_pending_is_debris: bool = False,
        note_for_dead: str = DEAD_PROCESS_NOTE,
    ) -> int:
        """Fail every unfinished row whose process is provably gone.

        ``pidless_pending_is_debris`` is the one difference between the
        two callers. A ``pending`` row with no pid is *normal* for the
        seconds between ``add`` and the pid bookkeeping, so a sweep that
        can run while a start is in flight (``StartDatasetTask``) waits
        for the age rule. At startup nothing is mid-spawn, so every
        pidless pending row is debris from a crash
        (``ReconcileDatasetTasks``).

        ``note_for_dead`` lets a caller stamp its own wording on the rows
        it is responsible for, so the caller stays the authority on *why*
        it swept.
        """
        now = self._clock.now()
        swept = 0
        for task in self._tasks.list_unfinished():
            reason = self._why_dead(task, now, pidless_pending_is_debris)
            if reason is None:
                continue
            note = note_for_dead if reason is DEAD_PROCESS_NOTE else reason
            if self._tasks.finalize_if_active(
                task.id, TaskStatus.FAILED, error=note
            ):
                swept += 1
        return swept

    def _why_dead(
        self, task: DatasetTask, now, pidless_pending_is_debris: bool
    ) -> str | None:
        if task.pid is not None:
            return None if self._gateway.is_alive(task.pid) else DEAD_PROCESS_NOTE
        if pidless_pending_is_debris:
            return STUCK_PENDING_NOTE
        if (now - task.created_at).total_seconds() > STUCK_PENDING_SECONDS:
            return STUCK_PENDING_NOTE
        return None