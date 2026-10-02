"""SqliteDatasetTasks -- dataset task rows in backend.db.

Every status change here goes through ``compare_and_swap_status``, the
one guarded statement shared with the graph-execution repository: so
exactly one writer wins and the losers get ``False`` instead of
clobbering it. ``finalize_if_active`` covers every terminal outcome
(finished / failed / killed, used by the child reporter, the stop use
case and reconciliation), and ``update_progress`` is the same CAS -- a
progress tick arriving after the task was killed is a no-op, not a
resurrection.

This adapter used to write that statement itself, five times, in three
near-identical port methods. One definition is easier to reason about
and, more to the point, is fixed once.

The child process uses this class through its own ``SqliteDatabase``
instance (one connection per thread per process; WAL lets it write
side-by-side with the server).
"""

from __future__ import annotations

import json
from datetime import datetime

from ..application.ports.clock import Clock
from ..application.ports.dataset_tasks import (
    DatasetTask,
    DatasetTasks,
    TaskKind,
    TaskStatus,
)
from .persistence.cas import compare_and_swap_status
from .persistence.sqlite import SqliteDatabase

# The active set is derived from the enum, so SQL cannot drift from what
# ``TaskStatus.is_active`` says (docs 08 S-24). Kept as words as well as
# an SQL fragment, because the guarded-UPDATE helper takes the values and
# the SELECTs want the fragment.
_ACTIVE_STATUSES: tuple[str, ...] = tuple(
    s.value for s in TaskStatus if s.is_active
)
_ACTIVE_SQL = "status IN ({})".format(
    ", ".join(f"'{s}'" for s in _ACTIVE_STATUSES)
)


def _parse_dt(raw: str | None) -> datetime:
    if not raw:
        return datetime.min
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return datetime.min


def _status(raw: str) -> TaskStatus | str:
    """The stored word, as an enum when it is one we know.

    A row carrying a status this build does not recognise is returned
    as the raw word rather than blowing up: a list endpoint must still
    answer for the rows it can read (the value is ``str``-compatible, so
    every consumer keeps working), and the vocabulary itself is fixed
    by this code. Unknown-kind is the same, for the same reason.
    """
    try:
        return TaskStatus(raw)
    except ValueError:
        return raw


def _kind(raw: str) -> TaskKind | str:
    try:
        return TaskKind(raw)
    except ValueError:
        return raw


def _row_to_task(row) -> DatasetTask:
    params: dict = {}
    if row["params"]:
        try:
            decoded = json.loads(row["params"])
            if isinstance(decoded, dict):
                params = decoded
        except (json.JSONDecodeError, TypeError):
            pass
    return DatasetTask(
        id=int(row["id"]),
        dataset=str(row["dataset"]),
        kind=_kind(str(row["kind"])),
        status=_status(str(row["status"])),
        pid=int(row["pid"]) if row["pid"] is not None else None,
        current=int(row["current_val"]),
        total=int(row["total_val"]),
        error=row["error"],
        params=params,
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


class SqliteDatasetTasks(DatasetTasks):
    def __init__(self, database: SqliteDatabase, clock: Clock) -> None:
        self._db = database
        self._clock = clock

    def add(
        self, *, dataset: str, kind: TaskKind | str, total: int, params: dict
    ) -> DatasetTask:
        now = self._clock.now().isoformat()
        with self._db.connection() as conn:
            cur = conn.execute(
                "INSERT INTO dataset_tasks "
                "(dataset, kind, status, total_val, params, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    dataset,
                    TaskKind(kind).value,
                    TaskStatus.PENDING.value,
                    total,
                    json.dumps(params),
                    now,
                    now,
                ),
            )
            task_id = int(cur.lastrowid)
        task = self.get(task_id)
        assert task is not None
        return task

    def get(self, task_id: int) -> DatasetTask | None:
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM dataset_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return _row_to_task(row) if row is not None else None

    def list_for(
        self, dataset: str, *, active_only: bool = False
    ) -> tuple[DatasetTask, ...]:
        sql = "SELECT * FROM dataset_tasks WHERE dataset = ?"
        if active_only:
            sql += f" AND {_ACTIVE_SQL}"
        sql += " ORDER BY id DESC"
        with self._db.connection() as conn:
            rows = conn.execute(sql, (dataset,)).fetchall()
        return tuple(_row_to_task(r) for r in rows)

    def find_active(self, dataset: str) -> DatasetTask | None:
        with self._db.connection() as conn:
            row = conn.execute(
                f"SELECT * FROM dataset_tasks WHERE dataset = ? AND {_ACTIVE_SQL} "
                f"ORDER BY id DESC LIMIT 1",
                (dataset,),
            ).fetchone()
        return _row_to_task(row) if row is not None else None

    def list_unfinished(self) -> tuple[DatasetTask, ...]:
        with self._db.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM dataset_tasks WHERE {_ACTIVE_SQL} ORDER BY id DESC"
            ).fetchall()
        return tuple(_row_to_task(r) for r in rows)

    def update_progress(
        self, task_id: int, current: int, pid: int | None = None
    ) -> bool:
        columns: dict[str, object] = {"current_val": current}
        if pid is not None:
            columns["pid"] = pid
        with self._db.connection() as conn:
            return compare_and_swap_status(
                conn,
                table="dataset_tasks",
                row_id=task_id,
                from_statuses=_ACTIVE_STATUSES,
                to_status=TaskStatus.RUNNING.value,
                set_columns={**columns, "updated_at": self._clock.now().isoformat()},
            )

    def finalize_if_active(
        self, task_id: int, status: TaskStatus, *, error: str | None = None
    ) -> bool:
        """active -> ``status``. One method for finished / failed / killed.

        Three near-identical methods used to live here, two of them
        sharing a private helper and the third repeating it to add an
        ``error`` column. They differed only in which status they wrote
        and whether they wrote a reason, so the difference is now a
        parameter and the statement is shared with every other
        terminal transition in the backend.
        """
        if not status.is_terminal:
            raise ValueError(f"{status.value} is not a terminal task status")
        columns: dict[str, object] = {}
        if error is not None:
            columns["error"] = error[:2000]
        with self._db.connection() as conn:
            return compare_and_swap_status(
                conn,
                table="dataset_tasks",
                row_id=task_id,
                from_statuses=_ACTIVE_STATUSES,
                to_status=status.value,
                set_columns={**columns, "updated_at": self._clock.now().isoformat()},
            )

    # -- internals -------------------------------------------------------
