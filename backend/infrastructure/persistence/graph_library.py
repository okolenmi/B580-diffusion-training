"""SqliteGraphLibrary -- the GraphLibrary port over ``saved_graphs``.

Upsert is ``INSERT OR IGNORE`` then ``UPDATE``: no read-then-write race
between two tabs saving the same name, and ``created`` (201 vs 200 at
the API) falls out of the insert's rowcount. The stored ``graph`` JSON
is returned exactly as written -- this adapter never re-validates it.
"""

from __future__ import annotations

import json
from datetime import datetime

from ...application.ports.graph_library import GraphLibrary, SavedGraph
from .sqlite import SqliteDatabase


def _row_to_saved(row) -> SavedGraph:
    return SavedGraph(
        name=row["name"],
        description=row["description"],
        graph=json.loads(row["graph"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


class SqliteGraphLibrary(GraphLibrary):
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def save(
        self, name: str, graph: dict, *, description: str = ""
    ) -> tuple[SavedGraph, bool]:
        now = datetime.now().isoformat()
        payload = json.dumps(graph)
        with self._db.connection() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO saved_graphs "
                "(name, description, graph, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, description, payload, now, now),
            )
            created = cursor.rowcount > 0
            if not created:
                conn.execute(
                    "UPDATE saved_graphs SET description = ?, graph = ?, "
                    "updated_at = ? WHERE name = ?",
                    (description, payload, now, name),
                )
            row = conn.execute(
                "SELECT name, description, graph, created_at, updated_at "
                "FROM saved_graphs WHERE name = ?",
                (name,),
            ).fetchone()
        return _row_to_saved(row), created

    def get(self, name: str) -> SavedGraph | None:
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT name, description, graph, created_at, updated_at "
                "FROM saved_graphs WHERE name = ?",
                (name,),
            ).fetchone()
        return _row_to_saved(row) if row else None

    def list_graphs(self) -> tuple[SavedGraph, ...]:
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT name, description, graph, created_at, updated_at "
                "FROM saved_graphs ORDER BY updated_at DESC, name ASC"
            ).fetchall()
        return tuple(_row_to_saved(row) for row in rows)

    def delete(self, name: str) -> bool:
        with self._db.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM saved_graphs WHERE name = ?", (name,)
            )
        return cursor.rowcount > 0
