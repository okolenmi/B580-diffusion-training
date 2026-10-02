"""ListDatasetItems -- a page of trajectory rows for the curation UI.

``committed`` is the v2 membership view (pending vs. committed); the
adapter refuses legacy datasets.

**Paging is no longer opt-in.** Without a ``limit`` this used to return
every row, because that is what the curation UI asked for. A dataset is
the one collection in this system whose size is genuinely unbounded --
it grows by ingestion, not by user action -- so that default meant a
response that grew until the browser stopped rendering it, and no way
for the UI to know it had only seen part of the data (docs 07 F-14,
docs 08 Q10). The default page is deliberately large enough that an
ordinary dataset is still served whole.

The result carries ``total`` and ``next_offset`` so a client can say
"showing 500 of 12,000" and offer to continue, rather than presenting
a truncated list as if it were the whole set. Those two fields are why
``count`` alone was never enough: it is the size of the page, not of
the result.
"""

from __future__ import annotations

from ..dto import DatasetItemsResult
from ..errors import InvalidQueryError
from ..limits import DEFAULT_DATASET_ITEM_PAGE_SIZE, MAX_PAGE_SIZE
from ..ports.dataset_library import DatasetLibrary


class ListDatasetItems:
    MAX_LIMIT = MAX_PAGE_SIZE
    DEFAULT_LIMIT = DEFAULT_DATASET_ITEM_PAGE_SIZE

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
        effective = self.DEFAULT_LIMIT if limit is None else limit
        if not 1 <= effective <= self.MAX_LIMIT:
            raise InvalidQueryError(
                f"limit must be between 1 and {self.MAX_LIMIT}, got {limit}"
            )
        if offset < 0:
            raise InvalidQueryError(f"offset cannot be negative, got {offset}")

        items = self._library.list_items(
            name, committed=committed, limit=effective, offset=offset
        )
        # Counted only for the *filters*, never for the whole table, and
        # only when it can change the answer: an unfiltered count of every
        # row is the one query a client does not need, because the page
        # already told it how many rows it got.
        total: int | None = None
        if committed is not None:
            total = self._library.count_items(name, committed=committed)
        else:
            stats = self._library.stats(name)
            total = stats.items

        next_offset = offset + len(items)
        return DatasetItemsResult(
            items=items,
            count=len(items),
            limit=effective,
            offset=offset,
            total=total,
            # None rather than a sentinel the client has to compare:
            # "no more rows" is a fact, and next_offset == offset is not
            # a way to say it (an empty page mid-dataset would look like
            # the end).
            next_offset=(
                next_offset
                if total is None or next_offset < total
                else None
            ),
        )