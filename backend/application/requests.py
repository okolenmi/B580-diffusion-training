"""Request value objects -- "the client named something", validated once.

Four guards were written five, three, two and two times respectively::

    if not kind:
        raise InvalidQueryError("asset kind is required")
    if not item_ids:
        raise InvalidQueryError("item_ids must not be empty")
    if all(v is None for v in (prompt, neg_prompt, cfg, type)):
        raise InvalidQueryError("no changes provided")
    if type is not None and type not in VALID_TYPES: ...

Copying a guard is how the wording drifts (``path is required`` versus
``relative_path is required`` for the same argument) and how one of the
copies ends up missing. These types raise once and hand the use cases a
value they can use directly (docs 08 S-09).

The port dataclasses (``ItemChanges``, ``BulkItemChanges``) stay pure
data; the *rules* about them live here, beside the other request rules.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import InvalidQueryError
from .ports.dataset_library import BulkItemChanges, ItemChanges

#: The item verdict vocabulary, shared by the single and bulk edits.
ITEM_TYPES: tuple[str, ...] = ("good", "bad")

#: How a bulk text edit combines with the existing caption.
TEXT_MODES: tuple[str, ...] = ("set", "prepend", "append")


def _one_of(value: str, allowed: tuple[str, ...], field: str) -> None:
    if value not in allowed:
        raise InvalidQueryError(
            f"unknown {field} {value!r}; expected one of {list(allowed)}"
        )


@dataclass(frozen=True, slots=True)
class AssetRequest:
    """An asset operation addressed by kind and (optionally) a path."""

    kind: str
    relative_path: str = ""

    @classmethod
    def of(
        cls,
        kind: str,
        relative_path: str = "",
        *,
        path_required: bool = False,
    ) -> "AssetRequest":
        if not kind:
            raise InvalidQueryError("asset kind is required")
        if path_required and not relative_path:
            raise InvalidQueryError("relative_path is required")
        return cls(kind=kind, relative_path=relative_path)


@dataclass(frozen=True, slots=True)
class ItemSelection:
    """A non-empty list of trajectory ids."""

    ids: tuple[int, ...]

    @classmethod
    def of(cls, item_ids: list[int]) -> "ItemSelection":
        if not item_ids:
            raise InvalidQueryError("item_ids must not be empty")
        return cls(ids=tuple(item_ids))

    def as_list(self) -> list[int]:
        return list(self.ids)

    def __iter__(self):
        return iter(self.ids)

    def __len__(self) -> int:
        return len(self.ids)


class ItemChangesRequest:
    """Factory for the two curation-patch shapes.

    Both raise the same two rules -- at least one field must actually
    change, and the verdict must be a known one -- so the single and
    bulk edit paths cannot drift apart.
    """

    @staticmethod
    def single(
        *,
        prompt: str | None = None,
        neg_prompt: str | None = None,
        cfg: float | None = None,
        type: str | None = None,
    ) -> ItemChanges:
        if type is not None:
            _one_of(type, ITEM_TYPES, "type")
        if all(v is None for v in (prompt, neg_prompt, cfg, type)):
            raise InvalidQueryError("no changes provided")
        return ItemChanges(
            prompt=prompt, neg_prompt=neg_prompt, cfg=cfg, type=type
        )

    @staticmethod
    def bulk(
        *,
        prompt: str | None = None,
        prompt_mode: str = "set",
        neg_prompt: str | None = None,
        neg_prompt_mode: str = "set",
        cfg: float | None = None,
        type: str | None = None,
    ) -> BulkItemChanges:
        _one_of(prompt_mode, TEXT_MODES, "prompt_mode")
        _one_of(neg_prompt_mode, TEXT_MODES, "neg_prompt_mode")
        if type is not None:
            _one_of(type, ITEM_TYPES, "type")
        if all(v is None for v in (prompt, neg_prompt, cfg, type)):
            raise InvalidQueryError("no changes provided")
        return BulkItemChanges(
            prompt=prompt,
            prompt_mode=prompt_mode,
            neg_prompt=neg_prompt,
            neg_prompt_mode=neg_prompt_mode,
            cfg=cfg,
            type=type,
        )