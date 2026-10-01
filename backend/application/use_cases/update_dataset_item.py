"""UpdateDatasetItem -- partial edit of one trajectory's curation fields.

``None`` means "untouched", so a cleared caption is sent as ``""`` and
arrives as a real value (the legacy API's ``is not None`` semantics,
kept deliberately). The "at least one field must change" and "the
verdict must be a known one" rules live in ``ItemChangesRequest``
(docs 08 S-09).
"""

from __future__ import annotations

from ..ports.dataset_library import DatasetItem, DatasetLibrary
from ..requests import ItemChangesRequest


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
        changes = ItemChangesRequest.single(
            prompt=prompt, neg_prompt=neg_prompt, cfg=cfg, type=type
        )
        return self._library.update_item(name, item_id, changes)