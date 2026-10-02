"""ListDatasets -- every dataset directory as one summary round-trip.

Each entry's ``preview_path`` is resolved through the
``DatasetPreviews`` port (stored override or first-item fallback);
legacy datasets honestly resolve to null.
"""

from __future__ import annotations

from dataclasses import replace

from ..dto import DatasetListResult
from ..ports.dataset_library import DatasetLibrary
from ..ports.dataset_previews import DatasetPreviews


class ListDatasets:
    def __init__(self, *, library: DatasetLibrary, previews: DatasetPreviews) -> None:
        self._library = library
        self._previews = previews

    def execute(self) -> DatasetListResult:
        summaries = tuple(
            replace(s, preview_path=self._previews.resolve(s.info.name))
            for s in self._library.list_datasets()
        )
        return DatasetListResult(datasets=summaries, count=len(summaries))
