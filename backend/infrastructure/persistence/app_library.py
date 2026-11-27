"""SqliteAppLibrary -- the AppLibrary port over ``apps``.

Same upsert shape as ``SqliteGraphLibrary`` (``INSERT OR IGNORE`` then ``UPDATE``, so two tabs saving one name cannot race and
``created`` falls out of the insert's rowcount). The version moves inside the UPDATE statement, so it is decided against the row as it
is at that moment, not as a tab last read it. The stored document is returned exactly as written.
"""

from __future__ import annotations

import json
from datetime import datetime

from ...application.ports.app_library import AppLibrary, SavedApp
from .sqlite import SqliteDatabase

_COLUMNS = "name, description, version, graph_hash, doc, created_at, updated_at, values_json"


def _row_to_saved(row) -> SavedApp:
    return SavedApp(
        name=row["name"],
        description=row["description"],
        version=row["version"],
        graph_hash=row["graph_hash"],
        doc=json.loads(row["doc"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        values=json.loads(row["values_json"]) if row["values_json"] else {},
    )


class SqliteAppLibrary(AppLibrary):
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def save(self, name: str, doc: dict, *, graph_hash: str, description: str = "") -> tuple[SavedApp, bool]:
        now = datetime.now().isoformat()
        payload = json.dumps(doc)
        with self._db.connection() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO apps (name, description, version, graph_hash, doc, created_at, updated_at) "
                "VALUES (?, ?, 1, ?, ?, ?, ?)",
                (name, description, graph_hash, payload, now, now),
            )
            created = cursor.rowcount > 0
            if not created:
                conn.execute(
                    "UPDATE apps SET description = ?, doc = ?, updated_at = ?, values_json = NULL, "
                    "version = CASE WHEN graph_hash != ? THEN version + 1 ELSE version END, graph_hash = ? "
                    "WHERE name = ?",
                    (description, payload, now, graph_hash, graph_hash, name),
                )
            row = conn.execute(f"SELECT {_COLUMNS} FROM apps WHERE name = ?", (name,)).fetchone()
        return _row_to_saved(row), created

    def put_values(self, name: str, values: dict) -> bool:
        with self._db.connection() as conn:
            cursor = conn.execute("UPDATE apps SET values_json = ? WHERE name = ?", (json.dumps(values), name))
        return cursor.rowcount > 0

    def get(self, name: str) -> SavedApp | None:
        with self._db.connection() as conn:
            row = conn.execute(f"SELECT {_COLUMNS} FROM apps WHERE name = ?", (name,)).fetchone()
        return _row_to_saved(row) if row else None

    def list_apps(self) -> tuple[SavedApp, ...]:
        with self._db.connection() as conn:
            rows = conn.execute(f"SELECT {_COLUMNS} FROM apps ORDER BY updated_at DESC, name ASC").fetchall()
        return tuple(_row_to_saved(row) for row in rows)

    def delete(self, name: str) -> bool:
        with self._db.connection() as conn:
            cursor = conn.execute("DELETE FROM apps WHERE name = ?", (name,))
        return cursor.rowcount > 0
