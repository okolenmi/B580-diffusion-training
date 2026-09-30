"""ListDatasetItems -- trajectory rows for the curation UI.

``committed`` is the v2 membership view (pending vs. committed); the
adapter refuses legacy datasets.
"""

from __future__ import annotations

from ..dto import DatasetItemsResult
from ..ports.dataset_library import DatasetLibrary


class ListDatasetItems:
    def __init__(self, *, library: DatasetLibrary) -> None:
        self._library = library

    def execute(self, name: str, *, committed: bool | None = None) -> DatasetItemsResult:
        items = self._library.list_items(name, committed=committed)
        return DatasetItemsResult(items=items, count=len(items))
