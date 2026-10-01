"""GetDataset -- one round-trip for a dataset detail page.

Legacy (v1) datasets fail here on purpose with migration guidance:
their stats/sets cannot be computed against v2 columns, and pretending
otherwise would serve a half-truth. The *list* endpoint still shows
them, flagged by ``format_version``.
"""

from __future__ import annotations

from ..dto import DatasetDetail
from ..errors import DatasetNotMigratedError
from ..ports.dataset_library import DatasetLibrary
from ..ports.dataset_previews import DatasetPreviews
from ..ports.dataset_tasks import DatasetTasks


class GetDataset:
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

    def execute(self, name: str) -> DatasetDetail:
        info = self._library.get(name)
        if info.format_version != 2:
            raise DatasetNotMigratedError(
                f"dataset '{name}' is in legacy format "
                f"(version {info.format_version}); run "
                f"scripts/migrate_dataset_format.py on it first",
                details={"dataset": name, "format_version": info.format_version},
            )
        return DatasetDetail(
            info=info,
            stats=self._library.stats(name),
            sets=self._library.list_sets(name),
            active_tasks=tuple(
                t for t in self._tasks.list_for(name, active_only=True)
            ),
            preview_path=self._previews.resolve(name),
        )
