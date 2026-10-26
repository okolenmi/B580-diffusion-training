"""Where model files live: scopes are folder lists, nothing more.

A scope never says what a file contains (that is detection's job). Names are always relative to a
scope's folder; absolute paths, '..' and symlinks that leave the folder are refused.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from collections.abc import Mapping, Sequence

from .header import HeaderCache, HeaderIndex, ModelFileError

SCOPES = ("checkpoints", "diffusion_models", "text_encoders", "vae", "loras")
SUPPORTED_SUFFIX = ".safetensors"


class ModelLibrary:
    def __init__(self, folders: Mapping[str, Sequence[str | os.PathLike]], cache: HeaderCache | None = None):
        unknown = set(folders) - set(SCOPES)
        if unknown:
            raise ValueError(f"unknown scope(s) {sorted(unknown)}; scopes are {SCOPES}")
        self._folders = {scope: tuple(Path(p) for p in paths) for scope, paths in folders.items()}
        self.cache = cache if cache is not None else HeaderCache()

    def folders(self, scope: str) -> tuple[Path, ...]:
        return self._folders.get(scope, ())

    def list_files(self, scope: str) -> list[str]:
        """Relative names of every .safetensors file under the scope's folders, sorted."""
        found: set[str] = set()
        for root in self.folders(scope):
            if not root.is_dir():
                continue
            real_root = root.resolve()
            for dirpath, _dirs, files in os.walk(root, followlinks=True):
                for name in files:
                    if name.lower().endswith(SUPPORTED_SUFFIX):
                        full = Path(dirpath, name)
                        try:
                            full.resolve().relative_to(real_root)
                        except ValueError:
                            continue
                        found.add(full.relative_to(root).as_posix())
        return sorted(found, key=str.lower)

    def resolve(self, scope: str, relative: str) -> Path:
        """The real path of a scope file, or ModelFileError."""
        if scope not in SCOPES:
            raise ModelFileError("bad_scope", f"unknown scope {scope!r}")
        if not isinstance(relative, str) or not relative.strip():
            raise ModelFileError("no_file", "no file chosen")
        pure = PurePosixPath(relative.replace("\\", "/"))
        if pure.is_absolute() or any(part in ("..", "") for part in pure.parts) or ":" in pure.parts[0]:
            raise ModelFileError("outside_scope", f"{relative!r} is not a name inside the {scope} folder")
        roots = self.folders(scope)
        if not roots:
            raise ModelFileError("no_folder", f"no folder is configured for {scope}")
        for root in roots:
            candidate = root / Path(*pure.parts)
            if not candidate.exists():
                continue
            real = candidate.resolve()
            try:
                real.relative_to(root.resolve())
            except ValueError:
                raise ModelFileError("outside_scope", f"{relative!r} leaves the {scope} folder") from None
            return real
        raise ModelFileError("not_found", f"{relative}: not found in the {scope} folder")

    def header(self, scope: str, relative: str) -> HeaderIndex:
        return self.cache.get(self.resolve(scope, relative))
