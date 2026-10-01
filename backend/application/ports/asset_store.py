"""AssetStore port -- model files on disk (checkpoints, LoRAs).

A "kind" names a base directory (per the resolved settings) plus a
capability set: both kinds currently supported are listable and
browsable; ``dataset`` joins later as a read-only kind.

Every relative path arriving from a client is untrusted: the
adapter sandboxes it against the kind's base directory and the use
cases validate presence -- this API is expected to be reachable from
another machine, and a path string is never treated as trusted
because it "came from the UI".

``inspect`` returns the fixed per-kind contract documented on the
adapter: header-only safetensors metadata, nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

#: Upload policy (the contract both presentation and the adapter honour):
#: model folders hold safetensors only -- the pickers list ``*.safetensors``
#: and the inspectors read nothing else, so any other name is rejected
#: before a byte is written.
UPLOAD_SUFFIXES = (".safetensors",)

#: Hard cap for one upload body, bytes (8 GiB -- larger than any single
#: checkpoint this app manages). Presentation refuses a larger declared
#: Content-Length before buffering and stops reading at the cap, so a
#: chunked request cannot buffer the server into swap; the adapter
#: enforces the same cap while writing (defence in depth).
MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AssetOption:
    """One picker entry: ``value`` is also the relative path."""

    value: str
    label: str


@dataclass(frozen=True, slots=True)
class AssetCatalog:
    """Everything a file picker needs in one round-trip."""

    kind: str
    base_dir: str
    options: tuple[AssetOption, ...]
    upload_supported: bool
    browse_supported: bool


@dataclass(frozen=True, slots=True)
class AssetBrowse:
    """Immediate children of one directory inside a kind."""

    kind: str
    path: str
    folders: tuple[str, ...]
    files: tuple[str, ...]


class AssetStore(ABC):
    @abstractmethod
    def catalog(self, kind: str) -> AssetCatalog:
        raise NotImplementedError

    @abstractmethod
    def browse(self, kind: str, path: str = "") -> AssetBrowse:
        raise NotImplementedError

    @abstractmethod
    def make_folder(self, kind: str, relative_path: str) -> str:
        """Create (idempotently) and return the absolute path."""
        raise NotImplementedError

    @abstractmethod
    def save_upload(self, kind: str, relative_path: str, content: bytes) -> str:
        """Write the bytes and return the absolute path."""
        raise NotImplementedError

    @abstractmethod
    def inspect(self, kind: str, relative_path: str) -> dict:
        raise NotImplementedError
