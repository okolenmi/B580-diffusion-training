"""DatasetFiles port -- byte reads inside a dataset directory.

Serving preview images is a *scoped* read: the only files a caller may
ever reach are the ones under ``datasets/{name}/``. An image URL must
never be steerable at the rest of the filesystem, so containment (and
its refusal) lives behind this port -- presentation receives bytes and
a media type, never a filesystem path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetFile:
    """File bytes plus the media type derived from the suffix."""

    content: bytes
    media_type: str


class DatasetFiles(ABC):
    @abstractmethod
    def read(self, dataset: str, rel_path: str) -> DatasetFile:
        """Bytes of ``datasets/{dataset}/{rel_path}``.

        Raises ``DatasetNotFoundError`` (no such dataset directory) or
        ``DatasetFileNotFoundError`` (missing, not a regular file, or
        escaping the dataset root -- the escape case is reported as
        not-found, never resolved).
        """
        raise NotImplementedError
