"""FsDatasetFiles -- DatasetFiles over the datasets directory.

Containment is decided on *resolved* paths: both the dataset root and
the target are resolved first, so ``..`` segments and symlinks alike
must land inside the dataset directory or the read is refused with a
plain not-found (an escape is never reported as a different error --
that would confirm the path exists).
"""

from __future__ import annotations

from pathlib import Path

from ..application.errors import DatasetFileNotFoundError, DatasetNotFoundError
from ..application.ports.dataset_files import DatasetFile, DatasetFiles

_MEDIA_TYPES = {
    ".png": "image/png",
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
}


class FsDatasetFiles(DatasetFiles):
    def __init__(self, datasets_dir: Path) -> None:
        self._datasets_dir = datasets_dir

    def read(self, dataset: str, rel_path: str) -> DatasetFile:
        root = (self._datasets_dir / dataset).resolve()
        if not root.is_relative_to(self._datasets_dir.resolve()):
            # name itself escapes (e.g. ".."): not a dataset of ours
            raise DatasetNotFoundError(f"dataset '{dataset}' not found")
        if not root.is_dir():
            raise DatasetNotFoundError(f"dataset '{dataset}' not found")

        target = (root / rel_path).resolve()
        if not target.is_relative_to(root):
            raise DatasetFileNotFoundError(
                f"file '{rel_path}' is outside dataset '{dataset}'"
            )
        if not target.is_file():
            raise DatasetFileNotFoundError(
                f"file '{rel_path}' not found in dataset '{dataset}'"
            )
        return DatasetFile(
            content=target.read_bytes(),
            media_type=_MEDIA_TYPES.get(
                target.suffix.lower(), "application/octet-stream"
            ),
        )
