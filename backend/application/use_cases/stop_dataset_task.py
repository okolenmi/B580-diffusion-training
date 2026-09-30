"""StopDatasetTask -- force-kill a pending/running ingestion child.

SIGKILL to the process group (legacy parity: ingestion has no
graceful-stop story -- SIGINT would race the builder's own ``except``
bookkeeping and blur "killed" into "failed"). The status flip is CASed
so a child that finished in the same instant keeps its own outcome.
"""

from __future__ import annotations

from ..errors import DatasetTaskNotFoundError, DatasetTaskNotActiveError
from ..ports.dataset_task_gateway import DatasetTaskGateway
from ..ports.dataset_tasks import (
    ACTIVE_TASK_STATUSES,
    DatasetTask,
    DatasetTasks,
)


class StopDatasetTask:
    def __init__(self, *, tasks: DatasetTasks, gateway: DatasetTaskGateway) -> None:
        self._tasks = tasks
        self._gateway = gateway

    def execute(self, task_id: int) -> DatasetTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise DatasetTaskNotFoundError(f"no dataset task {task_id}")
        if task.status not in ACTIVE_TASK_STATUSES:
            raise DatasetTaskNotActiveError(
                f"task {task_id} already {task.status}"
            )
        if task.pid is not None:
            self._gateway.kill(task.pid)
        self._tasks.kill_if_active(task_id)
        return self._tasks.get(task_id) or task
