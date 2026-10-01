"""BulkUpdateDatasetItems -- caption/prompt/CFG/verdict edits across many rows.

Truthiness gates on ``prompt``/``neg_prompt`` are legacy-compatible on
purpose (the bulk endpoint never used an empty string as "clear" --
use the single-item PATCH for that); text modes are ``set`` (replace),
``prepend`` (legacy trigger-word flow, applied before the existing
caption and idempotent) and ``append`` (M8e, after the existing
caption, idempotent). ``type`` flips the review verdict for the whole
selection (M8e multi-edit); at least one change must be provided, so a
PATCH that would only rewrite identical values is refused like the
single-item endpoint refuses an empty one.

The mode/verdict vocabulary and the "no changes" rule live in
``ItemChangesRequest``, shared with the single-item edit (docs 08 S-09).
"""

from __future__ import annotations

from ..dto import BulkUpdateResult
from ..ports.dataset_library import DatasetLibrary
from ..requests import ItemChangesRequest, ItemSelection


class BulkUpdateDatasetItems:
    def __init__(self, *, library: DatasetLibrary) -> None:
        self._library = library

    def execute(
        self,
        name: str,
        item_ids: list[int],
        *,
        prompt: str | None = None,
        prompt_mode: str = "set",
        neg_prompt: str | None = None,
        neg_prompt_mode: str = "set",
        cfg: float | None = None,
        type: str | None = None,
    ) -> BulkUpdateResult:
        selection = ItemSelection.of(item_ids)
        changes = ItemChangesRequest.bulk(
            prompt=prompt,
            prompt_mode=prompt_mode,
            neg_prompt=neg_prompt,
            neg_prompt_mode=neg_prompt_mode,
            cfg=cfg,
            type=type,
        )
        return BulkUpdateResult(
            updated=self._library.bulk_update(
                name, selection.as_list(), changes
            )
        )