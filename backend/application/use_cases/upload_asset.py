"""UploadAsset -- write raw bytes to a path inside a kind."""

from __future__ import annotations

from ..errors import InvalidQueryError
from ..ports.asset_store import AssetStore
from ..requests import AssetRequest


class UploadAsset:
    def __init__(self, *, assets: AssetStore) -> None:
        self._assets = assets

    def execute(self, kind: str, relative_path: str, content: bytes) -> str:
        request = AssetRequest.of(kind, relative_path, path_required=True)
        if not isinstance(content, (bytes, bytearray)):
            raise InvalidQueryError("content must be raw bytes")
        return self._assets.save_upload(
            request.kind, request.relative_path, bytes(content)
        )
