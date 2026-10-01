"""SqliteDatasetPreviews -- stored override + first-item fallback.

Two stores compose here on purpose: the override row lives in
backend.db (server state, migration 006), the fallback lives in the
dataset's own ``trajectories`` rows read through the library port.
Re-checking the override file at resolve time is what makes a stale
pointer (item discarded, shard cleaned up) degrade to the fallback
instead of serving a dead URL.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..application.errors import DatasetNotFoundError
from ..application.ports.dataset_library import DatasetLibrary
from ..application.ports.dataset_previews import DatasetPreviews
from .persistence.sqlite import SqliteDatabase


class SqliteDatasetPreviews(DatasetPreviews):
    def __init__(self, database: SqliteDatabase, library: DatasetLibrary) -> None:
        self._db = database
        self._library = library

    def resolve(self, name: str) -> str | None:
        stored: str | None = None
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT preview_path FROM dataset_previews WHERE dataset = ?",
                (name,),
            ).fetchone()
        if row is not None:
            stored = str(row[0])
        try:
            root = self._library.root(name)
        except DatasetNotFoundError:
            return None  # dataset gone between list and resolve: honest null
        # Both candidates are paths into the dataset dir -- containment
        # and existence are re-checked on every read, so a stale
        # pointer (item discarded, file removed) degrades to the next
        # candidate instead of serving a dead URL.
        for candidate in (stored, self._library.first_preview(name)):
            if not candidate:
                continue
            target = (root / candidate).resolve()
            if target.is_relative_to(root.resolve()) and target.is_file():
                return candidate
        return None

    def set(self, name: str, preview_path: str) -> None:
        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO dataset_previews (dataset, preview_path, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(dataset) DO UPDATE SET "
                "preview_path = excluded.preview_path, "
                "updated_at = excluded.updated_at",
                (name, preview_path, datetime.now(timezone.utc).isoformat()),
            )

    def remove(self, name: str) -> None:
        with self._db.connection() as conn:
            conn.execute(
                "DELETE FROM dataset_previews WHERE dataset = ?", (name,)
            )
