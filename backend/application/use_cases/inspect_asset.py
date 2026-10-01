"""InspectAsset -- header-only safetensors metadata for one file.

The returned dict is a deliberate fixed contract per kind (never a
raw header dump): checkpoints yield ``{kind, path, components}``,
LoRAs yield ``{kind, path, dtype, rank, key_count}``.
"""

from __future__ import annotations

from typing import Any

from ..ports.asset_store import AssetStore
from ..requests import AssetRequest


class InspectAsset:
    def __init__(self, *, assets: AssetStore) -> None:
        self._assets = assets

    def execute(self, kind: str, relative_path: str) -> dict[str, Any]:
        request = AssetRequest.of(kind, relative_path, path_required=True)
        return self._assets.inspect(request.kind, request.relative_path)
