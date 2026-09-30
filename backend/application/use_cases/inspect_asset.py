"""InspectAsset -- header-only safetensors metadata for one file.

The returned dict is a deliberate fixed contract per kind (never a
raw header dump): checkpoints yield ``{kind, path, components}``,
LoRAs yield ``{kind, path, dtype, rank, key_count}``.
"""

from __future__ import annotations

from typing import Any

from ..errors import InvalidQueryError
from ..ports.asset_store import AssetStore


class InspectAsset:
    def __init__(self, *, assets: AssetStore) -> None:
        self._assets = assets

    def execute(self, kind: str, relative_path: str) -> dict[str, Any]:
        if not kind:
            raise InvalidQueryError("asset kind is required")
        if not relative_path:
            raise InvalidQueryError("path is required")
        return self._assets.inspect(kind, relative_path)
