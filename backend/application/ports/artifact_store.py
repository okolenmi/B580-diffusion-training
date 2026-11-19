"""ArtifactStore -- the pictures nodes leave for the page while a graph runs (``Preview Image``)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ArtifactStore(ABC):
    @abstractmethod
    def resolve(self, path: str) -> Path | None:
        """The file a client path names, or None when it is not there or leaves the store. Never raises for a bad path."""
