"""BrowseAssets -- immediate children of one directory in a kind."""

from __future__ import annotations

from ..ports.asset_store import AssetBrowse, AssetStore
from ..requests import AssetRequest


class BrowseAssets:
    def __init__(self, *, assets: AssetStore) -> None:
        self._assets = assets

    def execute(self, kind: str, path: str = "") -> AssetBrowse:
        request = AssetRequest.of(kind, path)
        return self._assets.browse(request.kind, request.relative_path)
