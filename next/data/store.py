"""Where datasets live on disk: one folder per dataset, holding its library and its per-VAE caches."""

from __future__ import annotations

import re
import shutil
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

    def create(self, name: str) -> None:
        if (self._folder(name) / "library.sqlite").is_file():
            raise FileExistsError(f"dataset {name!r} already exists")
        self.library(name, create=True).close()

    def delete(self, name: str) -> None:
        """Remove the library and its caches; the image files it points to are not touched."""
        folder = self._folder(name)
        if not (folder / "library.sqlite").is_file():
            raise LookupError(f"dataset {name!r} does not exist; known: {self.names() or 'none'}")
        shutil.rmtree(folder)

    def cache_report(self, name: str) -> list[dict]:
        """One entry per VAE with a latent cache: its id, files and bytes."""
        root = self.caches(name)
        if not root.is_dir():
            return []
        out = []
        for vae in sorted(p for p in root.iterdir() if p.is_dir()):
            files = [f for f in vae.rglob("*") if f.is_file()]
            out.append({"vae": vae.name, "files": len(files), "bytes": sum(f.stat().st_size for f in files)})
        return out

    def delete_cache(self, name: str, vae: str) -> None:
        target = (self.caches(name) / vae).resolve()
        if not target.is_dir() or target.parent != self.caches(name).resolve():
            raise LookupError(f"dataset {name!r} has no cache for {vae!r}")
        shutil.rmtree(target)

    def caches(self, name: str) -> Path:
        return self._folder(name) / "caches"
