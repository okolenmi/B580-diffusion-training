"""CreateDataset -- a fresh, empty v2 dataset directory."""

from __future__ import annotations

from ..ports.dataset_library import DatasetInfo, DatasetLibrary


class CreateDataset:
    def __init__(self, *, library: DatasetLibrary) -> None:
        self._library = library

    def execute(self, name: str, description: str | None = None) -> DatasetInfo:
        # Name validation, ghost-directory cleanup, and schema creation
        # live in the adapter (the schema comes from the manager bridge
        # so it stays byte-identical to what the trainer writes).
        return self._library.create(name, description)
