"""Where datasets live on disk: one folder per dataset, holding its library and its per-VAE caches."""

from __future__ import annotations

import re
from pathlib import Path

from .library import ImageLibrary

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$")


class DatasetStore:
    """``root/<name>/library.sqlite`` and ``root/<name>/caches/<vae id>/``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _folder(self, name: str) -> Path:
        if not _NAME.match(name or ""):
            raise ValueError(f"{name!r} is not a usable dataset name (letters, digits, space, _ . -; at most 64)")
        return self.root / name

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "library.sqlite").is_file())

    def library(self, name: str, *, create: bool = False) -> ImageLibrary:
        folder = self._folder(name)
        if not create and not (folder / "library.sqlite").is_file():
            raise LookupError(f"dataset {name!r} does not exist; known: {self.names() or 'none'}")
        return ImageLibrary(folder / "library.sqlite")

    def caches(self, name: str) -> Path:
        return self._folder(name) / "caches"
