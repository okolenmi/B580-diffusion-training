"""SqliteRunRepository -- the RunRepository port backed by SQLite.

All SQL for runs lives in this file (the old server leaked a query
into ``process_manager.py``); the row <-> entity mapping is the only
translation point.
"""

from __future__ import annotations

from datetime import datetime

from ...application.ports.run_repository import RunRepository
from ...domain.entities.run import Run
from ...domain.exceptions import DomainError
from ...domain.value_objects import RunId, RunStatus
from .sqlite import SqliteDatabase

_COLUMNS = (
    "id, status, config_path, mode, phase, total_steps, done_steps, "
    "current_loss, avg_loss, pid, exit_code, error, log_path, "
    "created_at, updated_at, started_at, finished_at"
)

# id is generated; created_at is immutable identity.
_MUTABLE_COLUMNS = (
    "status, phase, total_steps, done_steps, current_loss, avg_loss, "
    "pid, exit_code, error, log_path, updated_at, started_at, finished_at"
)


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _row_to_run(row) -> Run:
    return Run(
        id=RunId(row["id"]),
        status=RunStatus(row["status"]),
        config_path=row["config_path"],
        mode=row["mode"],
        phase=row["phase"],
        total_steps=row["total_steps"],
        done_steps=row["done_steps"],
        current_loss=row["current_loss"],
        avg_loss=row["avg_loss"],
        pid=row["pid"],
        exit_code=row["exit_code"],
        error=row["error"],
        log_path=row["log_path"],
        created_at=_parse_dt(row["created_at"]),  # NOT NULL in schema
        updated_at=_parse_dt(row["updated_at"]),  # NOT NULL in schema
        started_at=_parse_dt(row["started_at"]),
        finished_at=_parse_dt(row["finished_at"]),
    )


class SqliteRunRepository(RunRepository):
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def add(self, run: Run) -> Run:
        if run.id is not None:
            raise DomainError(f"run already has id {run.id}")
        columns = (
            "status, config_path, mode, phase, total_steps, done_steps, "
            "current_loss, avg_loss, pid, exit_code, error, log_path, "
            "created_at, updated_at, started_at, finished_at"
        )
        placeholders = ", ".join("?" for _ in columns.split(", "))
        with self._db.connection() as conn:
            cursor = conn.execute(
                f"INSERT INTO runs ({columns}) VALUES ({placeholders})",
                (
                    run.status.value,
                    run.config_path,
                    run.mode,
                    run.phase,
                    run.total_steps,
                    run.done_steps,
                    run.current_loss,
                    run.avg_loss,
                    run.pid,
                    run.exit_code,
                    run.error,
                    run.log_path,
                    run.created_at.isoformat(),
                    run.updated_at.isoformat(),
                    run.started_at.isoformat() if run.started_at else None,
                    run.finished_at.isoformat() if run.finished_at else None,
                ),
            )
            run.assign_id(RunId(cursor.lastrowid))
        return run

    def get(self, run_id: RunId) -> Run | None:
        with self._db.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return _row_to_run(row) if row else None

    def list(self, *, limit: int = 50, status: RunStatus | None = None) -> list[Run]:
        query = f"SELECT {_COLUMNS} FROM runs"
        params: list[object] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._db.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_run(row) for row in rows]

    def update(self, run: Run) -> bool:
        if run.id is None:
            raise DomainError("cannot update an unpersisted run (no id yet)")
        assignments = ", ".join(f"{name} = ?" for name in _MUTABLE_COLUMNS.split(", "))
        with self._db.connection() as conn:
            cursor = conn.execute(
                f"UPDATE runs SET {assignments} WHERE id = ?",
                (
                    run.status.value,
                    run.phase,
                    run.total_steps,
                    run.done_steps,
                    run.current_loss,
                    run.avg_loss,
                    run.pid,
                    run.exit_code,
                    run.error,
                    run.log_path,
                    run.updated_at.isoformat(),
                    run.started_at.isoformat() if run.started_at else None,
                    run.finished_at.isoformat() if run.finished_at else None,
                    run.id,
                ),
            )
        return cursor.rowcount > 0

    def find_active(self) -> Run | None:
        with self._db.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM runs WHERE status = ? "
                "ORDER BY id DESC LIMIT 1",
                (RunStatus.RUNNING.value,),
            ).fetchone()
        return _row_to_run(row) if row else None

    def delete_all(self) -> int:
        with self._db.connection() as conn:
            cursor = conn.execute("DELETE FROM runs")
        return cursor.rowcount
