"""CommitDatasetItems -- membership-only training-set commit.

Files never move (format v2 invariant); committing to an existing set
name adds members to that set. The membership rules live in the
manager package and are reached through its lazy bridge in the
adapter -- one implementation of "what commit means".
"""

from __future__ import annotations

from ..dto import CommitResult
from ..errors import InvalidQueryError
from ..ports.dataset_library import DatasetLibrary


class CommitDatasetItems:
    def __init__(self, *, library: DatasetLibrary) -> None:
        self._library = library

    def execute(
        self, name: str, item_ids: list[int], *, set_name: str
    ) -> CommitResult:
        if not item_ids:
            raise InvalidQueryError("item_ids must not be empty")
        if not set_name or not set_name.strip():
            raise InvalidQueryError("set name must not be empty")
        set_id = self._library.commit(name, list(item_ids), set_name.strip())
        return CommitResult(
            set_id=set_id, set_name=set_name.strip(), added=len(item_ids)
        )
