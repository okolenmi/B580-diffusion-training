"""ListDatasetTasks -- task rows for one dataset.

A pure query: it validates the name, reads the rows the caller asked
for, and touches nothing else. Liveness sweeping used to happen here,
which meant a *read* of dataset A CAS-failed rows belonging to dataset
B -- and the sweep itself duplicated ``ReconcileDatasetTasks`` (docs 08
S-03). Sweeping now happens where liveness is actually owned: at
startup (``ReconcileDatasetTasks``) and when a task is started
(``StartDatasetTask``), which is also when a dead child becomes
knowable.
"""

from __future__ import annotations

from ..dto import DatasetTaskListResult
from ..ports.dataset_library import DatasetLibrary
from ..ports.dataset_tasks import DatasetTasks


class ListDatasetTasks:
    def __init__(
        self,
        *,
        library: DatasetLibrary,
        tasks: DatasetTasks,
    ) -> None:
        self._library = library
        self._tasks = tasks

    def execute(
        self, name: str, *, active_only: bool = True
    ) -> DatasetTaskListResult:
        self._library.root(name)  # validates name + existence
        rows = self._tasks.list_for(name, active_only=active_only)
        return DatasetTaskListResult(tasks=rows, count=len(rows))