"""UpdateDatasetItem -- partial edit of one trajectory's curation fields.

``None`` means "untouched", so a cleared caption is sent as ``""`` and
arrives as a real value (the legacy API's ``is not None`` semantics,
kept deliberately).
"""

from __future__ import annotations

from ..errors import InvalidQueryError
from ..ports.dataset_library import DatasetItem, DatasetLibrary, ItemChanges

VALID_TYPES: tuple[str, ...] = ("good", "bad")


class UpdateDatasetItem:
    def __init__(self, *, library: DatasetLibrary) -> None:
        self._library = library

    def execute(
        self,
        name: str,
        item_id: int,
        *,
        prompt: str | None = None,
        neg_prompt: str | None = None,
        cfg: float | None = None,
        type: str | None = None,
    ) -> DatasetItem:
        if type is not None and type not in VALID_TYPES:
            raise InvalidQueryError(
                f"unknown type {type!r}; expected one of {list(VALID_TYPES)}"
            )
        changes = ItemChanges(
            prompt=prompt, neg_prompt=neg_prompt, cfg=cfg, type=type
        )
        if all(v is None for v in (prompt, neg_prompt, cfg, type)):
            raise InvalidQueryError("no changes provided")
        return self._library.update_item(name, item_id, changes)
