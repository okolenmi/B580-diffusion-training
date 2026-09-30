"""BulkUpdateDatasetItems -- caption/prompt/CFG edits across many rows.

Truthiness gates on ``prompt``/``neg_prompt`` are legacy-compatible on
purpose (the bulk endpoint never used an empty string as "clear" --
use the single-item PATCH for that), and ``prompt_mode='prepend'``
keeps the trigger-word workflow the UI offers.
"""

from __future__ import annotations

from ..dto import BulkUpdateResult
from ..errors import InvalidQueryError
from ..ports.dataset_library import BulkItemChanges, DatasetLibrary

PROMPT_MODES: tuple[str, ...] = ("set", "prepend")


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
        cfg: float | None = None,
    ) -> BulkUpdateResult:
        if not item_ids:
            raise InvalidQueryError("item_ids must not be empty")
        if prompt_mode not in PROMPT_MODES:
            raise InvalidQueryError(
                f"unknown prompt_mode {prompt_mode!r}; "
                f"expected one of {list(PROMPT_MODES)}"
            )
        changes = BulkItemChanges(
            prompt=prompt,
            prompt_mode=prompt_mode,
            neg_prompt=neg_prompt,
            cfg=cfg,
        )
        return BulkUpdateResult(
            updated=self._library.bulk_update(name, list(item_ids), changes)
        )
