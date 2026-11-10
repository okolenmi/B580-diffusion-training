"""Request value objects -- "the client named something", validated once.

A guard copied into each use case drifts in wording (``path is required``
versus ``relative_path is required`` for the same argument) and ends up
missing from one of the copies. These types raise once and hand the use
cases a value they can use directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import InvalidQueryError


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
    ) -> AssetRequest:
        if not kind:
            raise InvalidQueryError("asset kind is required")
        if path_required and not relative_path:
            raise InvalidQueryError("relative_path is required")
        return cls(kind=kind, relative_path=relative_path)
