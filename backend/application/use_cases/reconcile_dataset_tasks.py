"""ReconcileDatasetTasks -- startup sweep of leftover task rows.

Runs once at boot before any request (the mirror of ``ReconcileRuns``):
any pending/running row whose process is gone -- including a pid that
now belongs to something else, which the gateway's liveness guard
rejects -- is CASed to ``failed`` with a reconcile note. Rows whose
child is genuinely still alive (a task outliving a server restart)
are left running: the reporter keeps writing and the UI keeps working.
"""

from __future__ import annotations

import logging

from ..dto import ReconcileResult
from ..ports.dataset_task_gateway import DatasetTaskGateway
from ..ports.dataset_tasks import DatasetTasks

logger = logging.getLogger(__name__)


class ReconcileDatasetTasks:
    def __init__(self, *, tasks: DatasetTasks, gateway: DatasetTaskGateway) -> None:
        self._tasks = tasks
        self._gateway = gateway

    def execute(self) -> ReconcileResult:
        cleaned = 0
        for task in self._tasks.list_unfinished():
            if task.pid is not None and self._gateway.is_alive(task.pid):
                logger.info(
                    "dataset task %s (%s) still alive after restart; keeping",
                    task.id, task.dataset,
                )
                continue
            if self._tasks.fail_if_active(
                task.id, "reconciled: task process is not running"
            ):
                cleaned += 1
        if cleaned:
            logger.info("reconciled %d unfinished dataset task(s)", cleaned)
        return ReconcileResult(cleaned=cleaned)
