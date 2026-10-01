"""ListAssets -- one round-trip for a file picker's catalog."""

from __future__ import annotations

from ..ports.asset_store import AssetCatalog, AssetStore
from ..requests import AssetRequest


class ListAssets:
    def __init__(self, *, assets: AssetStore) -> None:
        self._assets = assets

    def execute(self, kind: str) -> AssetCatalog:
        return self._assets.catalog(AssetRequest.of(kind).kind)
