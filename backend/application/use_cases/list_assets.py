"""ListAssets -- one round-trip for a file picker's catalog."""

from __future__ import annotations

from ..errors import InvalidQueryError
from ..ports.asset_store import AssetCatalog, AssetStore


class ListAssets:
    def __init__(self, *, assets: AssetStore) -> None:
        self._assets = assets

    def execute(self, kind: str) -> AssetCatalog:
        if not kind:
            raise InvalidQueryError("asset kind is required")
        return self._assets.catalog(kind)
