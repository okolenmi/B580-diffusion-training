"""SqliteRunRepository -- the RunRepository port backed by SQLite.

All SQL for runs lives in this file (the old server leaked a query
into ``process_manager.py``); the row <-> entity mapping is the only
translation point.

``update_if_status`` is the compare-and-swap primitive every status
transition goes through: ``UPDATE ... WHERE status = expected`` means
exactly one racing writer (supervisor, stop, reconcile) wins a run's
terminal transition, and the losers see ``rowcount == 0``.
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
    "current_loss, avg_loss, cache_done, cache_total, pid, exit_code, "
    "error, log_path, created_at, updated_at, started_at, finished_at"
)

# id is generated; created_at is immutable identity.
_MUTABLE_COLUMNS = (
    "status, phase, total_steps, done_steps, current_loss, avg_loss, "
    "cache_done, cache_total, pid, exit_code, error, log_path, "
    "updated_at, started_at, finished_at"
)

# Single source for the UPDATE's parameter order (per _MUTABLE_COLUMNS).
_MUTABLE_PARAMS = (
    "status", "phase", "total_steps", "done_steps", "current_loss",
    "avg_loss", "cache_done", "cache_total", "pid", "exit_code", "error",
    "log_path", "updated_at", "started_at", "finished_at",
)


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _row_to_run(row) -> Run:
    # restore(), not Run(): a loaded row is checked for the cross-field
    # rules a single column cannot carry (running without started_at, a
    # terminal row without finished_at, more steps done than planned).
    # An impossible row is a bug in whatever wrote it, and it says so
    # here rather than three transitions later (docs 08 S-16).
    return Run.restore(
        id=RunId(row["id"]),
        status=RunStatus(row["status"]),
        config_path=row["config_path"],
        mode=row["mode"],
        phase=row["phase"],
        total_steps=row["total_steps"],
        done_steps=row["done_steps"],
        current_loss=row["current_loss"],
        avg_loss=row["avg_loss"],
        cache_done=row["cache_done"],
        cache_total=row["cache_total"],
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
        columns = _COLUMNS.replace("id, ", "", 1)  # id is AUTOINCREMENT
        placeholders = ", ".join("?" for _ in columns.split(", "))
        with self._db.connection() as conn:
            cursor = conn.execute(
                f"INSERT INTO runs ({columns}) VALUES ({placeholders})",
                self._insert_values(run),
            )
            run.assign_id(RunId(cursor.lastrowid))
        return run

    @staticmethod
    def _insert_values(run: Run) -> tuple[object, ...]:
        return (
            run.status.value,
            run.config_path,
            run.mode,
            run.phase,
            run.total_steps,
            run.done_steps,
            run.current_loss,
            run.avg_loss,
            run.cache_done,
            run.cache_total,
            run.pid,
            run.exit_code,
            run.error,
            run.log_path,
            run.created_at.isoformat(),
            run.updated_at.isoformat(),
            run.started_at.isoformat() if run.started_at else None,
            run.finished_at.isoformat() if run.finished_at else None,
        )

    @staticmethod
    def _mutable_values(run: Run) -> tuple[object, ...]:
        attrs = {
            "status": run.status.value,
            "phase": run.phase,
            "total_steps": run.total_steps,
            "done_steps": run.done_steps,
            "current_loss": run.current_loss,
            "avg_loss": run.avg_loss,
            "cache_done": run.cache_done,
            "cache_total": run.cache_total,
            "pid": run.pid,
            "exit_code": run.exit_code,
            "error": run.error,
            "log_path": run.log_path,
            "updated_at": run.updated_at.isoformat(),
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        }
        return tuple(attrs[name] for name in _MUTABLE_PARAMS)

    def get(self, run_id: RunId) -> Run | None:
        with self._db.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return _row_to_run(row) if row else None

    def list_runs(self, *, limit: int = 50,
                    status: RunStatus | None = None) -> list[Run]:
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
                (*self._mutable_values(run), run.id),
            )
        return cursor.rowcount > 0

    def update_if_status(self, run: Run, expected: RunStatus) -> bool:
        if run.id is None:
            raise DomainError("cannot update an unpersisted run (no id yet)")
        assignments = ", ".join(f"{name} = ?" for name in _MUTABLE_COLUMNS.split(", "))
        with self._db.connection() as conn:
            cursor = conn.execute(
                f"UPDATE runs SET {assignments} WHERE id = ? AND status = ?",
                (*self._mutable_values(run), run.id, expected.value),
            )
        return cursor.rowcount > 0

    def find_active(self) -> Run | None:
        with self._db.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM runs "
                "WHERE status IN (?, ?) ORDER BY id DESC LIMIT 1",
                (RunStatus.CREATED.value, RunStatus.RUNNING.value),
            ).fetchone()
        return _row_to_run(row) if row else None

    def continue_ids_above(self, run_id: RunId) -> None:
        """Seed the AUTOINCREMENT sequence above an existing run dir.

        SQLite keeps that sequence in ``sqlite_sequence`` (created with
        the table, one row per AUTOINCREMENT table once a row is
        inserted). It has no unique index, so this is update-then-insert
        rather than an upsert -- and it never lowers a sequence that is
        already higher (docs 07 F-04).
        """
        with self._db.connection() as conn:
            current = conn.execute(
                "SELECT seq FROM sqlite_sequence WHERE name = 'runs'"
            ).fetchone()
            if current is None:
                conn.execute(
                    "INSERT INTO sqlite_sequence (name, seq) VALUES ('runs', ?)",
                    (int(run_id),),
                )
            elif int(current["seq"]) < int(run_id):
                conn.execute(
                    "UPDATE sqlite_sequence SET seq = ? WHERE name = 'runs'",
                    (int(run_id),),
                )

    def list_unfinished(self) -> list[Run]:
        with self._db.connection() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM runs "
                "WHERE status IN (?, ?) ORDER BY id DESC",
                (RunStatus.CREATED.value, RunStatus.RUNNING.value),
            ).fetchall()
        return [_row_to_run(row) for row in rows]

    def delete_all(self) -> int:
        with self._db.connection() as conn:
            cursor = conn.execute("DELETE FROM runs")
        return cursor.rowcount
