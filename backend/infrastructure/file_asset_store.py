"""FileSystemAssetStore -- model files under the resolved base dirs.

Security posture: every relative path a client sends is untrusted.
The sandboxed resolver below (ported semantics: no absolute paths,
no ``..``, no backslash tricks, final target must stay inside the
kind's base directory) is the *only* way a client path becomes a
filesystem path -- this module never string-joins its way out of the
base dir. The API is meant to be reachable from another machine; a
path gets the same treatment a public web app gives an upload
filename.

``inspect`` imports the safetensors header reader lazily (it pulls in
torch, which has no business being loaded by catalog/browse calls)
and returns the fixed per-kind contract, never a raw header dump.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from ..application.errors import InvalidQueryError
from ..application.ports.asset_store import (
    AssetBrowse,
    AssetCatalog,
    AssetOption,
    AssetStore,
)
from . import path_tiers
from .workspace import WorkspaceLayout

KINDS = ("checkpoint", "lora")


class FileSystemAssetStore(AssetStore):
    def __init__(self, layout: WorkspaceLayout) -> None:
        self._layout = layout

    # -- capabilities ---------------------------------------------------

    def catalog(self, kind: str) -> AssetCatalog:
        base = self._base(kind)
        files = self._list_model_files(base)
        return AssetCatalog(
            kind=kind,
            base_dir=str(base),
            options=tuple(AssetOption(value=name, label=name) for name in files),
            upload_supported=True,
            browse_supported=True,
        )

    def browse(self, kind: str, path: str = "") -> AssetBrowse:
        base = self._base(kind).resolve()
        target = self._safe_resolve(kind, path) if path else base
        if not target.exists():
            return AssetBrowse(kind=kind, path=path, folders=(), files=())
        if not target.is_dir():
            raise InvalidQueryError(f"not a directory: {path!r}")

        folders: list[str] = []
        files: list[str] = []
        for child in sorted(target.iterdir()):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                if child.name == "resume" and target == base:
                    continue  # auto-managed working files, same exclusion as listing
                folders.append(child.name)
            elif child.suffix == ".safetensors":
                files.append(child.name)
        return AssetBrowse(
            kind=kind, path=path, folders=tuple(folders), files=tuple(files)
        )

    def make_folder(self, kind: str, relative_path: str) -> str:
        resolved = self._safe_resolve(kind, relative_path)
        resolved.mkdir(parents=True, exist_ok=True)
        return str(resolved)

    def save_upload(self, kind: str, relative_path: str, content: bytes) -> str:
        resolved = self._safe_resolve(kind, relative_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(content)
        return str(resolved)

    def inspect(self, kind: str, relative_path: str) -> dict[str, Any]:
        self._base(kind)  # kind whitelist (checkpoint/lora only)
        resolved = self._safe_resolve(kind, relative_path)
        if not resolved.exists():
            raise InvalidQueryError(f"no such file: {relative_path!r}")
        if not resolved.is_file():
            raise InvalidQueryError(f"not a file: {relative_path!r}")

        path_tiers.import_paths(self._layout.project_root)
        try:
            from nodes.model.resource_inspection import (  # repo bridge
                dtype_to_str,
                inspect_checkpoint_dtypes,
                inspect_lora,
            )
        except ImportError as exc:  # pragma: no cover - env breakage
            raise InvalidQueryError(f"resource inspection unavailable: {exc}") from exc

        if kind == "checkpoint":
            try:
                per_component = inspect_checkpoint_dtypes(resolved)
            except Exception as exc:
                raise InvalidQueryError(
                    f"{relative_path!r} doesn't look like a valid safetensors "
                    f"file ({exc})"
                ) from exc
            return {
                "kind": kind,
                "path": relative_path,
                "components": {
                    name: {
                        "dtype": dtype_to_str(info.dtype),
                        "key_count": info.key_count,
                    }
                    for name, info in per_component.items()
                },
            }

        try:
            info = inspect_lora(resolved)
        except Exception as exc:
            raise InvalidQueryError(
                f"{relative_path!r} doesn't look like a valid LoRA safetensors "
                f"file ({exc})"
            ) from exc
        return {
            "kind": kind,
            "path": relative_path,
            "dtype": dtype_to_str(info.dtype),
            "rank": info.rank,
            "key_count": info.key_count,
        }

    # -- internals ------------------------------------------------------

    def _base(self, kind: str) -> Path:
        if kind == "checkpoint":
            return self._layout.checkpoints_dir
        if kind == "lora":
            return self._layout.loras_dir
        raise InvalidQueryError(
            f"unknown asset kind {kind!r}; expected one of {list(KINDS)}"
        )

    def _safe_resolve(self, kind: str, relative: str) -> Path:
        """Untrusted relative path -> absolute target inside the base dir."""
        base = self._base(kind)
        if not relative or relative != relative.strip():
            raise InvalidQueryError("path must not be empty or have surrounding whitespace")
        normalized = relative.replace("\\", "/")
        parsed = PurePosixPath(normalized)
        if parsed.is_absolute() or ".." in parsed.parts or "" in parsed.parts:
            raise InvalidQueryError(f"invalid relative path: {relative!r}")
        base_resolved = base.resolve()
        resolved = (base_resolved / Path(*parsed.parts)).resolve()
        if resolved != base_resolved and not resolved.is_relative_to(base_resolved):
            raise InvalidQueryError(f"path escapes the {kind} directory: {relative!r}")
        return resolved

    @staticmethod
    def _list_model_files(base: Path) -> list[str]:
        """All .safetensors under ``base`` as relative paths, excluding the
        resume/ working subfolder and anything dot-prefixed (the same
        visibility rule ``browse`` applies). Empty list (not an error)
        when the directory doesn't exist yet."""
        if not base.is_dir():
            return []
        results: list[str] = []
        for path in sorted(base.rglob("*.safetensors")):
            try:
                rel = path.relative_to(base)
            except ValueError:
                continue
            if rel.parts and rel.parts[0] == "resume":
                continue
            if any(part.startswith(".") for part in rel.parts):
                continue  # same dotfile rule as browse: hidden stays hidden
            results.append(str(rel))
        return results
