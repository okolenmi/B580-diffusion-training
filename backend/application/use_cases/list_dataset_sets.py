"""ListDatasetSets -- named training sets with member counts."""

from __future__ import annotations

from ..ports.dataset_library import DatasetLibrary, TrainingSetInfo


class ListDatasetSets:
    def __init__(self, *, library: DatasetLibrary) -> None:
        self._library = library

    def execute(self, name: str) -> tuple[TrainingSetInfo, ...]:
        return self._library.list_sets(name)
