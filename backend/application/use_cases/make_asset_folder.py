"""MakeAssetFolder -- create a subfolder inside a kind (idempotent)."""

from __future__ import annotations

from ..errors import InvalidQueryError
from ..ports.asset_store import AssetStore


class MakeAssetFolder:
    def __init__(self, *, assets: AssetStore) -> None:
        self._assets = assets

    def execute(self, kind: str, relative_path: str) -> str:
        if not kind:
            raise InvalidQueryError("asset kind is required")
        if not relative_path:
            raise InvalidQueryError("relative_path is required")
        return self._assets.make_folder(kind, relative_path)
