"""Image libraries on disk, through ``next.data`` (the one code that reads and writes them).

Every call opens the library it needs and closes it again, so request threads and the import thread never
share a SQLite connection. Thumbnails are made on first request and kept beside the library.
"""

from __future__ import annotations

import io
import threading
import time
import uuid
from pathlib import Path

from ..application.errors import DatasetAlreadyExistsError, DatasetFileNotFoundError, DatasetNotFoundError, InvalidQueryError
from ..application.ports.image_libraries import ImageLibraries
from .graph import tree
from .workspace import WorkspaceLayout

_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".bmp": "image/bmp"}
_KEPT_JOBS = 10


class _Jobs:
    """Import jobs of the running server; they are not kept across a restart."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_library: dict[str, list[dict]] = {}

    def start(self, name: str) -> dict:
        with self._lock:
            jobs = self._by_library.setdefault(name, [])
            if any(j["state"] == "running" for j in jobs):
                raise InvalidQueryError(f"an import into {name!r} is already running")
            job = {"id": uuid.uuid4().hex[:8], "state": "running", "done": 0, "total": 0, "added": 0, "known": 0,
                   "skipped": [], "error": None, "started": time.time()}
            jobs.insert(0, job)
            del jobs[_KEPT_JOBS:]
            return job

    def list(self, name: str) -> list[dict]:
        with self._lock:
            return [dict(j) for j in self._by_library.get(name, [])]


class FsImageLibraries(ImageLibraries):
    def __init__(self, layout: WorkspaceLayout) -> None:
        self._layout = layout
        self._jobs = _Jobs()

    @property
    def _store(self):
        return tree.DatasetStore(self._layout.datasets_dir)

    def _open(self, name: str):
        try:
            return self._store.library(name)
        except LookupError as exc:
            raise DatasetNotFoundError(str(exc)) from exc
        except ValueError as exc:
            raise InvalidQueryError(str(exc)) from exc

    def _summary(self, name: str) -> dict:
        lib = self._open(name)
        try:
            return {"name": name, **lib.stats(), "caches": self._store.cache_report(name)}
        finally:
            lib.close()

    def list(self) -> list[dict]:
        return [self._summary(n) for n in self._store.names()]

    def create(self, name: str) -> dict:
        try:
            self._store.create(name)
        except FileExistsError as exc:
            raise DatasetAlreadyExistsError(str(exc)) from exc
        except ValueError as exc:
            raise InvalidQueryError(str(exc)) from exc
        return self._summary(name)

    def delete(self, name: str) -> None:
        try:
            self._store.delete(name)
        except LookupError as exc:
            raise DatasetNotFoundError(str(exc)) from exc
        except ValueError as exc:
            raise InvalidQueryError(str(exc)) from exc

    def page(self, name: str, *, set_name: str, query: str, missing: str, offset: int, limit: int) -> dict:
        lib = self._open(name)
        try:
            return {"total": lib.count(set_name, query=query, missing=missing),
                    "items": lib.page(set_name, query=query, missing=missing, offset=offset, limit=limit)}
        except ValueError as exc:
            raise InvalidQueryError(str(exc)) from exc
        finally:
            lib.close()

    def ids(self, name: str, *, set_name: str, query: str, missing: str) -> list[str]:
        lib = self._open(name)
        try:
            return lib.ids(set_name, query=query, missing=missing)
        except ValueError as exc:
            raise InvalidQueryError(str(exc)) from exc
        finally:
            lib.close()

    def report(self, name: str, channel: str) -> dict:
        lib = self._open(name)
        try:
            return tree.library_report(lib.snapshot(), channel)
        except ValueError as exc:
            raise InvalidQueryError(str(exc)) from exc
        finally:
            lib.close()

    def image(self, name: str, image_id: str, *, thumb: int | None) -> tuple[bytes, str]:
        lib = self._open(name)
        try:
            path = lib.path_of(image_id)
        finally:
            lib.close()
        source = Path(path) if path else None
        if source is None or not source.is_file() or source.suffix.lower() not in _MEDIA:
            raise DatasetFileNotFoundError(f"image {image_id!r} is not readable (moved or deleted?)")
        if not thumb:
            return source.read_bytes(), _MEDIA[source.suffix.lower()]
        cached = self._store.caches(name).parent / "thumbs" / f"{image_id}_{thumb}.webp"
        if not cached.is_file():
            from PIL import Image

            with Image.open(source) as picture:
                picture.thumbnail((thumb, thumb))
                buffer = io.BytesIO()
                picture.convert("RGB").save(buffer, "WEBP", quality=80)
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(buffer.getvalue())
        return cached.read_bytes(), "image/webp"

    def edit_captions(self, name: str, *, ids: list[str] | None, set_name: str, query: str, channel: str, mode: str, text: str, find: str) -> int:
        lib = self._open(name)
        try:
            chosen = ids if ids is not None else lib.ids(set_name, query=query)
            return lib.edit_captions(chosen, channel, mode, text, find)
        except ValueError as exc:
            raise InvalidQueryError(str(exc)) from exc
        finally:
            lib.close()

    def remove(self, name: str, ids: list[str], set_name: str | None) -> int:
        lib = self._open(name)
        try:
            return lib.remove(ids, set_name)
        finally:
            lib.close()

    def add_to_set(self, name: str, set_name: str, ids: list[str]) -> int:
        lib = self._open(name)
        try:
            return lib.add_to_set(set_name, ids)
        finally:
            lib.close()

    def delete_set(self, name: str, set_name: str) -> None:
        lib = self._open(name)
        try:
            lib.delete_set(set_name)
        except ValueError as exc:
            raise InvalidQueryError(str(exc)) from exc
        finally:
            lib.close()

    def delete_cache(self, name: str, vae: str) -> None:
        self._open(name).close()
        try:
            self._store.delete_cache(name, vae)
        except LookupError as exc:
            raise DatasetNotFoundError(str(exc)) from exc

    def start_import(self, name: str, folder: Path, *, recursive: bool, set_name: str) -> dict:
        self._open(name).close()
        if not folder.is_dir():
            raise InvalidQueryError(f"{str(folder)!r} is not a folder on this machine")
        job = self._jobs.start(name)
        threading.Thread(target=self._run_import, args=(name, folder, recursive, set_name, job), name=f"import-{name}", daemon=True).start()
        return dict(job)

    def _run_import(self, name: str, folder: Path, recursive: bool, set_name: str, job: dict) -> None:
        def progress(done: int, total: int) -> None:
            job["done"], job["total"] = done, total

        try:
            lib = self._open(name)
            try:
                result = lib.ingest(folder, set_name=set_name, recursive=recursive, progress=progress)
            finally:
                lib.close()
            job.update(added=result.added, known=result.known, skipped=result.skipped[:50], state="done")
        except Exception as exc:  # the job reports it; the server goes on
            job.update(state="failed", error=f"{type(exc).__name__}: {exc}")

    def imports(self, name: str) -> list[dict]:
        self._open(name).close()
        return self._jobs.list(name)
