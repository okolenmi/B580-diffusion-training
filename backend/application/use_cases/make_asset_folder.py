"""MakeAssetFolder -- create a subfolder inside a kind (idempotent)."""

from __future__ import annotations

from ..ports.asset_store import AssetStore
from ..requests import AssetRequest


class MakeAssetFolder:
    def __init__(self, *, assets: AssetStore) -> None:
        self._assets = assets

    def execute(self, kind: str, relative_path: str) -> str:
        request = AssetRequest.of(kind, relative_path, path_required=True)
        return self._assets.make_folder(request.kind, request.relative_path)
