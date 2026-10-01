"""SqliteDatasetTasks -- dataset task rows in backend.db.

Compare-and-swap mirrors ``SqliteRunRepository``: every finaliser
(``finish``/``fail``/``kill``, used by the child reporter, the stop
use case, and reconciliation) writes through
``WHERE status IN ('pending','running')`` so exactly one outcome wins
and the losers get ``False`` instead of clobbering it. ``update_progress``
is the same CAS -- a progress tick arriving after the task was killed
is a no-op, not a resurrection.

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
from .persistence.sqlite import SqliteDatabase

# The active set is derived from the enum, so SQL cannot drift from what
# ``TaskStatus.is_active`` says (docs 08 S-24).
_ACTIVE_SQL = "status IN ({})".format(
    ", ".join(f"'{s.value}'" for s in TaskStatus if s.is_active)
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
        now = self._clock.now().isoformat()
        with self._db.connection() as conn:
            if pid is None:
                cur = conn.execute(
                    "UPDATE dataset_tasks SET status = ?, current_val = ?, "
                    f"updated_at = ? WHERE id = ? AND {_ACTIVE_SQL}",
                    (TaskStatus.RUNNING.value, current, now, task_id),
                )
            else:
                cur = conn.execute(
                    "UPDATE dataset_tasks SET status = ?, current_val = ?, "
                    f"pid = ?, updated_at = ? WHERE id = ? AND {_ACTIVE_SQL}",
                    (TaskStatus.RUNNING.value, current, pid, now, task_id),
                )
            return cur.rowcount == 1

    def finish_if_active(self, task_id: int) -> bool:
        return self._finalize(task_id, TaskStatus.FINISHED)

    def fail_if_active(self, task_id: int, error: str) -> bool:
        now = self._clock.now().isoformat()
        with self._db.connection() as conn:
            cur = conn.execute(
                "UPDATE dataset_tasks SET status = ?, error = ?, "
                f"updated_at = ? WHERE id = ? AND {_ACTIVE_SQL}",
                (TaskStatus.FAILED.value, error[:2000], now, task_id),
            )
            return cur.rowcount == 1

    def kill_if_active(self, task_id: int) -> bool:
        return self._finalize(task_id, TaskStatus.KILLED)

    # -- internals -------------------------------------------------------

    def _finalize(self, task_id: int, status: TaskStatus) -> bool:
        if not status.is_terminal:
            raise ValueError(f"{status.value} is not a terminal task status")
        now = self._clock.now().isoformat()
        with self._db.connection() as conn:
            cur = conn.execute(
                f"UPDATE dataset_tasks SET status = ?, updated_at = ? "
                f"WHERE id = ? AND {_ACTIVE_SQL}",
                (status.value, now, task_id),
            )
            return cur.rowcount == 1
