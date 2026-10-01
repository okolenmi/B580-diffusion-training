"""DiscardDatasetItems -- delete rows, previews, and empty shards.

Set membership cascades with the row (a committed item can be
discarded; its set simply loses a member). Shard files are only
untouched or fully removed -- a shard with any surviving row keeps its
file, per the format contract.
"""

from __future__ import annotations

from ..dto import DiscardItemsResult
from ..errors import InvalidQueryError
from ..requests import ItemSelection
from ..ports.dataset_library import DatasetLibrary


class DiscardDatasetItems:
    def __init__(self, *, library: DatasetLibrary) -> None:
        self._library = library

    def execute(self, name: str, item_ids: list[int]) -> DiscardItemsResult:
        selection = ItemSelection.of(item_ids)
        return DiscardItemsResult(
            deleted=self._library.discard(name, selection.as_list())
        )
