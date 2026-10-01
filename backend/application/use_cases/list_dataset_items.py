"""ListDatasetItems -- trajectory rows for the curation UI.

``committed`` is the v2 membership view (pending vs. committed); the
adapter refuses legacy datasets. ``limit``/``offset`` are opt-in paging:
without a limit the adapter still returns every row (what the UI asks
for), but a caller that knows the dataset is huge can page through it
instead of materialising all of it in one request (docs 07 F-14).
"""

from __future__ import annotations

from ..dto import DatasetItemsResult
from ..limits import MAX_PAGE_SIZE
from ..errors import InvalidQueryError
from ..ports.dataset_library import DatasetLibrary


class ListDatasetItems:
    MAX_LIMIT = MAX_PAGE_SIZE

    def __init__(self, *, library: DatasetLibrary) -> None:
        self._library = library

    def execute(
        self,
        name: str,
        *,
        committed: bool | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetItemsResult:
        if limit is not None and not 1 <= limit <= self.MAX_LIMIT:
            raise InvalidQueryError(
                f"limit must be between 1 and {self.MAX_LIMIT}, got {limit}"
            )
        if offset < 0:
            raise InvalidQueryError(f"offset cannot be negative, got {offset}")
        items = self._library.list_items(
            name, committed=committed, limit=limit, offset=offset
        )
        return DatasetItemsResult(
            items=items, count=len(items), limit=limit, offset=offset
        )