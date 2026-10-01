"""DeleteDataset -- remove a dataset directory (allowed for any format).

Refuses while a task is pending/running: killing a half-written shard
mid-ingest would leave the row/file bookkeeping in a state nobody can
reason about. The caller stops the task first. The card preview
override is dropped with the directory so a dataset created later
under the same name starts from its own items, not a stranger's row.
"""

from __future__ import annotations

from ..errors import DatasetTaskActiveError
from ..ports.dataset_library import DatasetLibrary
from ..ports.dataset_previews import DatasetPreviews
from ..ports.dataset_tasks import DatasetTasks


class DeleteDataset:
    def __init__(
        self,
        *,
        library: DatasetLibrary,
        tasks: DatasetTasks,
        previews: DatasetPreviews,
    ) -> None:
        self._library = library
        self._tasks = tasks
        self._previews = previews

    def execute(self, name: str) -> bool:
        # Raises DatasetNotFoundError when it does not exist -- a DELETE
        # of nothing is a client mistake here, not a no-op.
        self._library.root(name)
        active = self._tasks.find_active(name)
        if active is not None:
            raise DatasetTaskActiveError(
                f"dataset '{name}' has task {active.id} still "
                f"{active.status}; stop it before deleting",
                details={"task_id": active.id, "status": active.status},
            )
        deleted = self._library.delete(name)
        if deleted:
            self._previews.remove(name)
        return deleted
