"""FileArtifactStore -- ArtifactStore over one folder; only the picture files nodes write are served."""

from __future__ import annotations

from pathlib import Path

from ..application.ports.artifact_store import ArtifactStore

_SERVED = {".png"}


class FileArtifactStore(ArtifactStore):
    def __init__(self, root: Path) -> None:
        self._root = root

    def resolve(self, path: str) -> Path | None:
        if not path or "\x00" in path:
            return None
        root = self._root.resolve()
        target = (root / path).resolve()
        if root not in target.parents or target.suffix.lower() not in _SERVED or not target.is_file():
            return None
        return target
