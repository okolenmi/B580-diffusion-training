"""ImageLibraries port -- the image libraries (images + captions + sets) that training nodes open.

One library per dataset name. Images stay where they are on disk; a library keeps their identity (content hash),
size and captions. Answers are plain documents shaped for the Datasets page (rework/contracts/datasets.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ImageLibraries(ABC):
    @abstractmethod
    def list(self) -> list[dict]:
        """``{name, images, bytes, sets: {name: count}, captioned: {channel: count}, orientation: {portrait, square, landscape}, covers: [image id], caches: [{vae, files, bytes}]}`` per library."""

    @abstractmethod
    def create(self, name: str) -> dict:
        """A new empty library. Raises ``DatasetAlreadyExistsError`` / ``InvalidQueryError`` (unusable name)."""

    @abstractmethod
    def delete(self, name: str) -> None:
        """Remove the library and its caches (never the image files). Raises ``DatasetNotFoundError``."""

    @abstractmethod
    def page(self, name: str, *, set_name: str, query: str, missing: str, offset: int, limit: int) -> dict:
        """``{total, items: [{id, width, height, bytes, path, captions, sets}]}`` of one set, newest first; ``missing`` is a caption channel the images must lack."""

    @abstractmethod
    def ids(self, name: str, *, set_name: str, query: str, missing: str) -> list[str]:
        """Every image id the same filters select, newest first."""

    @abstractmethod
    def report(self, name: str, channel: str) -> dict:
        """Problems worth seeing before a run (size spread, empty and repeated captions, tag counts) for one caption channel."""

    @abstractmethod
    def image(self, name: str, image_id: str, *, thumb: int | None) -> tuple[bytes, str]:
        """The picture bytes and media type, or a thumbnail no larger than ``thumb`` pixels. Raises ``DatasetFileNotFoundError``."""

    @abstractmethod
    def edit_captions(self, name: str, *, ids: list[str] | None, set_name: str, query: str, channel: str, mode: str, text: str, find: str) -> int:
        """Change one caption channel of ``ids``, or of everything the set and query select when ``ids`` is None; returns how many changed."""

    @abstractmethod
    def remove(self, name: str, ids: list[str], set_name: str | None) -> int:
        """Take images out of one set, or out of the library when ``set_name`` is None."""

    @abstractmethod
    def add_to_set(self, name: str, set_name: str, ids: list[str]) -> int:
        """Add images to a set (created when new); returns its size."""

    @abstractmethod
    def delete_set(self, name: str, set_name: str) -> None: ...

    @abstractmethod
    def delete_cache(self, name: str, vae: str) -> None: ...

    @abstractmethod
    def start_import(self, name: str, folder: Path, *, recursive: bool, set_name: str) -> dict:
        """Start reading a folder in the background; one at a time per library. Returns the job document."""

    @abstractmethod
    def imports(self, name: str) -> list[dict]:
        """Import jobs of this library, newest first: ``{id, state, done, total, added, known, skipped, error}``."""
