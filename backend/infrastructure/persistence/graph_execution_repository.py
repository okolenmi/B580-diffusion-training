"""SqliteGraphExecutionRepository -- GraphExecutionRepository in SQLite.

All SQL for graph executions lives here; the row <-> entity mapping is
the only translation point (``graph`` JSON <-> ``GraphDefinition``,
``results`` JSON <-> ``NodeResult`` tuple). ``update_if_status`` is the
CAS every terminal transition goes through, same semantics as the runs
table: ``UPDATE ... WHERE id = ? AND status = ?`` means exactly one
racing writer (supervisor, stop, reconcile, delete-and-lose) wins.
"""

from __future__ import annotations

import json
from datetime import datetime

from ...application.ports.graph_execution_repository import GraphExecutionRepository
from ...domain.entities.graph_execution import GraphExecution
from ...domain.exceptions import DomainError
from ...domain.graph import GraphDefinition, NodeResult
from ...domain.value_objects import ExecutionId, GraphStatus
from .sqlite import SqliteDatabase

_COLUMNS = (
    "id, status, graph, results, error, created_at, updated_at, "
    "started_at, finished_at"
)

# id is generated; graph is immutable (the submission snapshot).
_MUTABLE = ("status", "results", "error", "updated_at", "started_at", "finished_at")


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _decode_results(raw: str) -> tuple[NodeResult, ...]:
    return tuple(
        NodeResult(
            node_id=str(item.get("node_id", "")),
            ok=bool(item.get("ok", False)),
            outputs=dict(item.get("outputs") or {}),
            error=item.get("error"),
            duration_ms=float(item.get("duration_ms") or 0.0),
        )
        for item in json.loads(raw or "[]")
    )


def _row_to_execution(row) -> GraphExecution:
    # restore(), not GraphExecution(): see the run mapper's note
    # (docs 08 S-16).
    return GraphExecution.restore(
        id=ExecutionId(row["id"]),
        status=GraphStatus(row["status"]),
        graph=GraphDefinition.from_dict(json.loads(row["graph"])),
        results=_decode_results(row["results"]),
        error=row["error"],
        created_at=_parse_dt(row["created_at"]),  # NOT NULL in schema
        updated_at=_parse_dt(row["updated_at"]),  # NOT NULL in schema
        started_at=_parse_dt(row["started_at"]),
        finished_at=_parse_dt(row["finished_at"]),
    )


class SqliteGraphExecutionRepository(GraphExecutionRepository):
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def add(self, execution: GraphExecution) -> GraphExecution:
        if execution.id is not None:
            raise DomainError(f"execution already has id {execution.id}")
        with self._db.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO graph_executions "
                "(status, graph, results, error, created_at, updated_at, "
                " started_at, finished_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                self._insert_values(execution),
            )
            execution.assign_id(ExecutionId(cursor.lastrowid))
        return execution

    @staticmethod
    def _insert_values(execution: GraphExecution) -> tuple[object, ...]:
        return (
            execution.status.value,
            json.dumps(execution.graph.as_dict()),
            json.dumps(
                SqliteGraphExecutionRepository._results_payload(execution.results)
            ),
            execution.error,
            execution.created_at.isoformat(),
            execution.updated_at.isoformat(),
            execution.started_at.isoformat() if execution.started_at else None,
            execution.finished_at.isoformat() if execution.finished_at else None,
        )

    @staticmethod
    def _results_payload(results: tuple[NodeResult, ...]) -> list[dict]:
        return [
            {
                "node_id": r.node_id,
                "ok": r.ok,
                "outputs": r.outputs,
                "error": r.error,
                "duration_ms": r.duration_ms,
            }
            for r in results
        ]

    @staticmethod
    def _mutable_values(execution: GraphExecution) -> tuple[object, ...]:
        return (
            execution.status.value,
            json.dumps(
                SqliteGraphExecutionRepository._results_payload(execution.results)
            ),
            execution.error,
            execution.updated_at.isoformat(),
            execution.started_at.isoformat() if execution.started_at else None,
            execution.finished_at.isoformat() if execution.finished_at else None,
        )

    def get(self, execution_id: ExecutionId) -> GraphExecution | None:
        with self._db.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM graph_executions WHERE id = ?",
                (execution_id,),
            ).fetchone()
        return _row_to_execution(row) if row else None

    def list_executions(self, *, limit: int = 50) -> list[GraphExecution]:
        with self._db.connection() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM graph_executions "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_execution(row) for row in rows]

    def update(self, execution: GraphExecution) -> bool:
        if execution.id is None:
            raise DomainError("cannot update an unpersisted execution (no id yet)")
        assignments = ", ".join(f"{name} = ?" for name in _MUTABLE)
        with self._db.connection() as conn:
            cursor = conn.execute(
                f"UPDATE graph_executions SET {assignments} WHERE id = ?",
                (*self._mutable_values(execution), execution.id),
            )
        return cursor.rowcount > 0

    def update_if_status(
        self, execution: GraphExecution, expected: GraphStatus
    ) -> bool:
        if execution.id is None:
            raise DomainError("cannot update an unpersisted execution (no id yet)")
        assignments = ", ".join(f"{name} = ?" for name in _MUTABLE)
        with self._db.connection() as conn:
            cursor = conn.execute(
                f"UPDATE graph_executions SET {assignments} WHERE id = ? AND status = ?",
                (*self._mutable_values(execution), execution.id, expected.value),
            )
        return cursor.rowcount > 0

    def find_active(self) -> GraphExecution | None:
        with self._db.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM graph_executions "
                "WHERE status IN (?, ?) ORDER BY id DESC LIMIT 1",
                (GraphStatus.QUEUED.value, GraphStatus.RUNNING.value),
            ).fetchone()
        return _row_to_execution(row) if row else None

    def list_unfinished(self) -> list[GraphExecution]:
        with self._db.connection() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM graph_executions "
                "WHERE status IN (?, ?) ORDER BY id DESC",
                (GraphStatus.QUEUED.value, GraphStatus.RUNNING.value),
            ).fetchall()
        return [_row_to_execution(row) for row in rows]

    def delete_all(self) -> int:
        with self._db.connection() as conn:
            cursor = conn.execute("DELETE FROM graph_executions")
        return cursor.rowcount
