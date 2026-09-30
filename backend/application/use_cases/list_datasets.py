"""ListDatasets -- every dataset directory as one summary round-trip."""

from __future__ import annotations

from ..dto import DatasetListResult
from ..ports.dataset_library import DatasetLibrary


class ListDatasets:
    def __init__(self, *, library: DatasetLibrary) -> None:
        self._library = library

    def execute(self) -> DatasetListResult:
        summaries = self._library.list()
        return DatasetListResult(datasets=summaries, count=len(summaries))
