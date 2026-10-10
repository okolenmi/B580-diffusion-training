"""SqliteTemplateLibrary -- the TemplateLibrary port over ``templates``.

Same upsert shape as ``SqliteGraphLibrary`` (``INSERT OR IGNORE`` then
``UPDATE``, so two tabs saving one name cannot race and ``created`` falls
out of the insert's rowcount). The stored document is returned exactly as
written.
"""

from __future__ import annotations

import json
from datetime import datetime

from ...application.ports.template_library import SavedTemplate, TemplateLibrary
from .sqlite import SqliteDatabase

_COLUMNS = "name, description, kind, doc, created_at, updated_at"


def _row_to_saved(row) -> SavedTemplate:
    return SavedTemplate(
        name=row["name"],
        description=row["description"],
        kind=row["kind"],
        doc=json.loads(row["doc"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


class SqliteTemplateLibrary(TemplateLibrary):
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def save(
        self, name: str, doc: dict, *, kind: str, description: str = ""
    ) -> tuple[SavedTemplate, bool]:
        now = datetime.now().isoformat()
        payload = json.dumps(doc)
        with self._db.connection() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO templates "
                "(name, description, kind, doc, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, description, kind, payload, now, now),
            )
            created = cursor.rowcount > 0
            if not created:
                conn.execute(
                    "UPDATE templates SET description = ?, kind = ?, doc = ?, "
                    "updated_at = ? WHERE name = ?",
                    (description, kind, payload, now, name),
                )
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM templates WHERE name = ?", (name,)
            ).fetchone()
        return _row_to_saved(row), created

    def get(self, name: str) -> SavedTemplate | None:
        with self._db.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM templates WHERE name = ?", (name,)
            ).fetchone()
        return _row_to_saved(row) if row else None

    def list_templates(self, kind: str | None = None) -> tuple[SavedTemplate, ...]:
        with self._db.connection() as conn:
            if kind is None:
                rows = conn.execute(
                    f"SELECT {_COLUMNS} FROM templates "
                    "ORDER BY updated_at DESC, name ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {_COLUMNS} FROM templates WHERE kind = ? "
                    "ORDER BY updated_at DESC, name ASC",
                    (kind,),
                ).fetchall()
        return tuple(_row_to_saved(row) for row in rows)

    def delete(self, name: str) -> bool:
        with self._db.connection() as conn:
            cursor = conn.execute("DELETE FROM templates WHERE name = ?", (name,))
        return cursor.rowcount > 0
