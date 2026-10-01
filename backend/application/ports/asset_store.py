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
#: Content-Length before reading the body and stops reading at the cap;
#: the adapter re-checks every chunk *while writing*, so the cap holds
#: even if the body lies about its length.
#:
#: The cap being 8 GiB is exactly why the write is streamed: buffering a
#: body of that size, then joining the chunks, holds about twice the
#: file in RAM. It used to (docs 08 N-02, measured at 971 ms of event-loop
#: stall for 600 MB, and ~2x RAM).
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


class AssetUploadWriter(ABC):
    """An open, in-progress upload.

    Streaming in three calls so the caller can hand over one chunk at a
    time and nothing ever holds the whole body: ``write`` appends,
    ``finish`` makes the result visible at the final path, ``abort``
    removes the partial. Exactly one of the last two must be called --
    :meth:`abort` is idempotent, and the writer is also usable as a
    context manager, which is how a caller guarantees the abort on an
    exception without a try/finally of its own.

    All three raise on failure and leave no partial behind.
    """

    @abstractmethod
    def write(self, chunk: bytes) -> None:
        """Append one chunk. Raises ``AssetTooLargeError`` past the cap."""
        raise NotImplementedError

    @abstractmethod
    def finish(self) -> str:
        """Commit and return the absolute path of the final file."""
        raise NotImplementedError

    @abstractmethod
    def abort(self) -> None:
        """Remove the partial. Never raises, never leaves a ``.part``."""
        raise NotImplementedError


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
    def begin_upload(self, kind: str, relative_path: str, *,
                     overwrite: bool = False) -> AssetUploadWriter:
        """Validate everything, then open a streaming write.

        `overwrite=False` (the default) refuses an existing target with
        ``AssetExistsError`` -- a ``PUT`` that silently replaces a real
        checkpoint is data loss with no way to tell it happened
        (docs 08 N-14). Every check runs before any byte is written, so a
        refused upload leaves no directory, no file and no partial.

        The caller must ``finish()`` or ``abort()`` the writer.
        """
        raise NotImplementedError

    @abstractmethod
    def inspect(self, kind: str, relative_path: str) -> dict:
        raise NotImplementedError
