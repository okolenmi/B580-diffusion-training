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

Uploads are policy-checked before any write: ``.safetensors`` names
only (``UPLOAD_SUFFIXES``), an existing target is refused unless the
caller explicitly asked to overwrite, capped at ``max_upload_bytes``
*while writing*, and written to a ``.part`` sibling that is renamed on
commit -- so a rejected, oversized, overwritten-by-accident or
interrupted upload leaves nothing behind.

The write is streamed chunk by chunk (``begin_upload`` returning a
writer) rather than handed a finished ``bytes``. At this app's 8 GiB
cap, buffering the body and then joining the chunks held about twice the
file in RAM and stalled the event loop for the duration (docs 08 N-02,
measured).
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

from ..application.errors import (
    AssetExistsError,
    AssetTooLargeError,
    InvalidQueryError,
)
from ..application.ports.asset_store import (
    MAX_UPLOAD_BYTES,
    UPLOAD_SUFFIXES,
    AssetBrowse,
    AssetCatalog,
    AssetOption,
    AssetStore,
    AssetUploadWriter,
)
from ..application.ports.dataset_library import DatasetLibrary
from . import path_tiers
from .workspace import WorkspaceLayout

KINDS = ("checkpoint", "lora", "dataset")

# Kinds that are catalog-only: names appear in pickers, but browsing,
# uploading, folder-making, and inspection are rejected with guidance
# toward the datasets API (which owns the real semantics).
CATALOG_ONLY_KINDS = ("dataset",)


class FileSystemAssetStore(AssetStore):
    def __init__(
        self, layout: WorkspaceLayout, *, datasets: DatasetLibrary | None = None
    ) -> None:
        self._layout = layout
        self._datasets = datasets
        # contract default; overridable (tests, future settings)
        self.max_upload_bytes = MAX_UPLOAD_BYTES

    # -- capabilities ---------------------------------------------------

    def catalog(self, kind: str) -> AssetCatalog:
        if kind == "dataset":
            return self._dataset_catalog()
        base = self._base(kind)
        files = self._list_model_files(base)
        return AssetCatalog(
            kind=kind,
            base_dir=str(base),
            options=tuple(AssetOption(value=name, label=name) for name in files),
            upload_supported=True,
            browse_supported=True,
        )

    def _dataset_catalog(self) -> AssetCatalog:
        """Dataset names as picker options (M3b: the catalog-only kind).

        Read from the library so the list applies the same visibility
        rules as the datasets API (dot-prefixed dirs skipped, no
        metadata.db skipped); no files are ever surfaced here."""
        if self._datasets is None:  # pragma: no cover - composition bug
            raise InvalidQueryError("dataset catalog is not wired")
        names = [summary.info.name for summary in self._datasets.list_datasets()]
        return AssetCatalog(
            kind="dataset",
            base_dir=str(self._layout.datasets_dir),
            options=tuple(AssetOption(value=name, label=name) for name in names),
            upload_supported=False,
            browse_supported=False,
        )

    def browse(self, kind: str, path: str = "") -> AssetBrowse:
        if kind in CATALOG_ONLY_KINDS:
            raise InvalidQueryError(
                f"asset kind {kind!r} is catalog-only; "
                f"use the datasets API instead"
            )
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
        self._reject_catalog_only(kind, "create folders in")
        resolved = self._safe_resolve(kind, relative_path)
        resolved.mkdir(parents=True, exist_ok=True)
        return str(resolved)

    def begin_upload(self, kind: str, relative_path: str, *,
                     overwrite: bool = False) -> AssetUploadWriter:
        # validate everything BEFORE touching the filesystem: a rejected
        # upload must leave no directory, no file, no partial (docs 07,
        # quality rule 3)
        self._reject_catalog_only(kind, "upload into")
        resolved = self._safe_resolve(kind, relative_path)
        if resolved.suffix.lower() not in UPLOAD_SUFFIXES:
            raise InvalidQueryError(
                f"uploads must end in {', '.join(UPLOAD_SUFFIXES)}: {relative_path!r}"
            )
        # Refuse to clobber an existing checkpoint unless asked (N-14).
        # Checked here, before the .part is created, so a refusal leaves
        # nothing at all behind -- including no directory.
        if resolved.exists() and not overwrite:
            raise AssetExistsError(
                f"{relative_path!r} already exists; pass overwrite=true to "
                f"replace it"
            )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        # write beside the target, then rename: an interrupted or failed
        # write never leaves a partial file at the final path (.part is
        # invisible to pickers -- they list *.safetensors only)
        partial = resolved.with_name(resolved.name + ".part")
        # A stale .part from a killed process would be silently
        # truncated-and-appended by a plain "ab"; start from nothing so a
        # resumed write can never splice two uploads together.
        try:
            # No context manager: this handle is the writer's, and it
            # outlives this function -- write()/finish()/abort() all use
            # it, and finish() is what closes it (or abort() does, on the
            # failure path). A `with` here would close it before the
            # first chunk.
            handle = open(partial, "wb")  # noqa: SIM115 -- see above
        except OSError as exc:
            raise InvalidQueryError(f"cannot write {relative_path!r}: {exc}") from exc
        return _PartialUpload(
            writer=handle,
            partial=partial,
            final=resolved,
            max_bytes=self.max_upload_bytes,
        )

    def inspect(self, kind: str, relative_path: str) -> dict[str, Any]:
        self._reject_catalog_only(kind, "inspect files in")
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

    @staticmethod
    def _reject_catalog_only(kind: str, action: str) -> None:
        if kind in CATALOG_ONLY_KINDS:
            raise InvalidQueryError(
                f"asset kind {kind!r} is catalog-only; use the datasets API "
                f"to {action} datasets"
            )

    def _base(self, kind: str) -> Path:
        if kind == "checkpoint":
            return self._layout.checkpoints_dir
        if kind == "lora":
            return self._layout.loras_dir
        if kind == "dataset":
            return self._layout.datasets_dir
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


class _PartialUpload(AssetUploadWriter):
    """One in-progress upload: chunks into ``partial``, then renamed.

    The cap is enforced *here*, on the running total, not from a
    declared length: a chunked request has no content-length to trust,
    and the whole point of this class is that a body cannot decide how
    much it costs.
    """

    def __init__(self, writer, partial: Path, final: Path, max_bytes: int) -> None:
        self._writer = writer
        self._partial = partial
        self._final = final
        self._max_bytes = max_bytes
        self._written = 0
        self._settled = False  # finish() or abort() has run

    def write(self, chunk: bytes) -> None:
        if self._settled:
            raise RuntimeError("write() after finish()/abort()")
        # Check before appending, so an over-cap upload never puts the
        # offending bytes on disk either.
        if self._written + len(chunk) > self._max_bytes:
            self.abort()
            raise AssetTooLargeError(
                f"{self._final.name!r} exceeds the "
                f"{self._max_bytes}-byte cap"
            )
        self._writer.write(chunk)
        self._written += len(chunk)

    def finish(self) -> str:
        if self._settled:
            raise RuntimeError("finish() called twice")
        self._settled = True
        try:
            self._writer.flush()
            os.fsync(self._writer.fileno())
            self._writer.close()
            self._partial.replace(self._final)
        except BaseException:
            # Includes KeyboardInterrupt/CancelledError: a torn upload
            # must not leave a partial the next run would append to.
            self._cleanup()
            raise
        return str(self._final)

    def abort(self) -> None:
        if self._settled:
            return
        self._settled = True
        self._cleanup()

    def _cleanup(self) -> None:
        try:
            self._writer.close()
        except OSError:
            pass  # already closed, or never opened: nothing to undo
        try:
            self._partial.unlink()
        except OSError:
            pass  # the point is only "no .part remains"

    def __enter__(self) -> _PartialUpload:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Never suppresses -- see UploadSession.__exit__ for why the
        # return type is None rather than bool (mypy exit-return).
        # Only the unhappy path: on success finish() has already run and
        # this is a no-op, so `with` cannot double-commit.
        self.abort()
