"""ReconcileDatasetTasks -- startup sweep of leftover task rows.

Runs once at boot before any request (the mirror of ``ReconcileRuns``):
any pending/running row whose process is gone -- including a pid that
now belongs to something else, which the gateway's liveness guard
rejects -- is CASed to ``failed`` with a reconcile note. Rows whose
child is genuinely still alive (a task outliving a server restart)
are left running: the reporter keeps writing and the UI keeps working.

The "is it dead" rule itself belongs to ``DatasetTaskSweeper``, which
``StartDatasetTask`` uses for the same judgement (docs 08 S-03).
"""

from __future__ import annotations

import logging

from ..dataset_task_sweeper import DatasetTaskSweeper
from ..dto import ReconcileResult

logger = logging.getLogger(__name__)


class ReconcileDatasetTasks:
    def __init__(self, *, sweeper: DatasetTaskSweeper) -> None:
        self._sweeper = sweeper

    def execute(self) -> ReconcileResult:
        # At startup no start is in flight, so a pidless pending row is
        # debris rather than a launch that has not recorded its pid yet.
        cleaned = self._sweeper.sweep(
            pidless_pending_is_debris=True,
            note_for_dead="reconciled: task process is not running",
        )
        if cleaned:
            logger.info("reconciled %d unfinished dataset task(s)", cleaned)
        return ReconcileResult(cleaned=cleaned)