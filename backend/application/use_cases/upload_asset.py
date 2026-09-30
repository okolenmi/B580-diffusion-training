"""UploadAsset -- write raw bytes to a path inside a kind."""

from __future__ import annotations

from ..errors import InvalidQueryError
from ..ports.asset_store import AssetStore


class UploadAsset:
    def __init__(self, *, assets: AssetStore) -> None:
        self._assets = assets

    def execute(self, kind: str, relative_path: str, content: bytes) -> str:
        if not kind:
            raise InvalidQueryError("asset kind is required")
        if not relative_path:
            raise InvalidQueryError("relative_path is required")
        if not isinstance(content, (bytes, bytearray)):
            raise InvalidQueryError("content must be raw bytes")
        return self._assets.save_upload(kind, relative_path, bytes(content))
