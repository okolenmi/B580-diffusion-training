"""BrowseAssets -- immediate children of one directory in a kind."""

from __future__ import annotations

from ..errors import InvalidQueryError
from ..ports.asset_store import AssetBrowse, AssetStore


class BrowseAssets:
    def __init__(self, *, assets: AssetStore) -> None:
        self._assets = assets

    def execute(self, kind: str, path: str = "") -> AssetBrowse:
        if not kind:
            raise InvalidQueryError("asset kind is required")
        return self._assets.browse(kind, path)
