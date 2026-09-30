"""ListDatasetTasks -- task rows, with lazy cleanup of dead rows.

Two sweeps before the list is returned:

* ``running`` rows whose child died without reporting (OOM kill, a
  write that never landed) are detected via the gateway's liveness
  guard and CASed to ``failed`` -- the UI never shows a zombie progress
  bar;
* ``pending`` rows with no pid older than ``_STUCK_PENDING_SECONDS``
  failed before their first progress write (crash between ``add`` and
  pid bookkeeping; normally impossible -- the start use case records
  the pid inside its lock -- so age is what makes sweeping it safe).

Rows are otherwise untouched: a task whose child genuinely died keeps
its counts, error text, and terminal status forever (task history).
"""

from __future__ import annotations

from ..dto import DatasetTaskListResult
from ..ports.clock import Clock
from ..ports.dataset_library import DatasetLibrary
from ..ports.dataset_task_gateway import DatasetTaskGateway
from ..ports.dataset_tasks import DatasetTasks

_STUCK_PENDING_SECONDS = 60.0


class ListDatasetTasks:
    def __init__(
        self,
        *,
        library: DatasetLibrary,
        tasks: DatasetTasks,
        gateway: DatasetTaskGateway,
        clock: Clock,
    ) -> None:
        self._library = library
        self._tasks = tasks
        self._gateway = gateway
        self._clock = clock

    def execute(
        self, name: str, *, active_only: bool = True
    ) -> DatasetTaskListResult:
        self._library.root(name)  # validates name + existence
        self._sweep_dead()
        rows = self._tasks.list_for(name, active_only=active_only)
        return DatasetTaskListResult(tasks=rows, count=len(rows))

    def _sweep_dead(self) -> None:
        now = self._clock.now()
        for task in self._tasks.list_unfinished():
            if task.status == "running" and task.pid is not None:
                if not self._gateway.is_alive(task.pid):
                    self._tasks.fail_if_active(
                        task.id, "task process is not running"
                    )
            elif task.status == "pending" and task.pid is None:
                age = (now - task.created_at).total_seconds()
                if age > _STUCK_PENDING_SECONDS:
                    self._tasks.fail_if_active(
                        task.id, "task never reported progress"
                    )
