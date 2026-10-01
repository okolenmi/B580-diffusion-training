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
"""

from __future__ import annotations

from ..dto import BulkUpdateResult
from ..errors import InvalidQueryError
from ..ports.dataset_library import BulkItemChanges, DatasetLibrary
from .update_dataset_item import VALID_TYPES

TEXT_MODES: tuple[str, ...] = ("set", "prepend", "append")


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
        if not item_ids:
            raise InvalidQueryError("item_ids must not be empty")
        if prompt_mode not in TEXT_MODES:
            raise InvalidQueryError(
                f"unknown prompt_mode {prompt_mode!r}; "
                f"expected one of {list(TEXT_MODES)}"
            )
        if neg_prompt_mode not in TEXT_MODES:
            raise InvalidQueryError(
                f"unknown neg_prompt_mode {neg_prompt_mode!r}; "
                f"expected one of {list(TEXT_MODES)}"
            )
        if type is not None and type not in VALID_TYPES:
            raise InvalidQueryError(
                f"unknown type {type!r}; expected one of {list(VALID_TYPES)}"
            )
        if all(v is None for v in (prompt, neg_prompt, cfg, type)):
            raise InvalidQueryError("no changes provided")
        changes = BulkItemChanges(
            prompt=prompt,
            prompt_mode=prompt_mode,
            neg_prompt=neg_prompt,
            neg_prompt_mode=neg_prompt_mode,
            cfg=cfg,
            type=type,
        )
        return BulkUpdateResult(
            updated=self._library.bulk_update(name, list(item_ids), changes)
        )
