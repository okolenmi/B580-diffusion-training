"""DatasetTasks port -- dataset task lifecycle rows in backend.db.

Task state is *server* state (that is exactly why format v2 removed
the dataset-side ``tasks`` table): rows live in the backend database,
are written through this port by three racing actors -- the start use
case, the ingestion child process (via its reporter), and stop/startup
reconciliation -- and the compare-and-swap methods make the race safe
the same way ``RunRepository.update_if_status`` does: exactly one
final outcome wins, the losers silently no-op (``False``).

Statuses: ``pending`` (row created, child not yet reporting) ->
``running`` (first progress write, carries the pid) -> exactly one of
``finished`` | ``failed`` | ``killed``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Lifecycle of one dataset task row.

    The vocabulary used to be two module constants plus two tuples of
    strings (``ACTIVE_TASK_STATUSES``, ``TERMINAL_TASK_STATUSES``), and
    the adapter wrote the values as SQL literals -- so "is this row
    still running" was a question four files could answer differently.
    Active-ness is now a property of the status, and the adapter's SQL
    predicate is generated from it (docs 08 S-24).

    ``str``-valued: the column is TEXT and every JSON body reports the
    word, so nothing downstream changes shape.
    """

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    KILLED = "killed"

    @property
    def is_active(self) -> bool:
        """Pending or running: the row still has a process or is about to."""
        return self in _ACTIVE

    @property
    def is_terminal(self) -> bool:
        """Ended for good -- kept forever as task history."""
        return not self.is_active


_ACTIVE = frozenset({TaskStatus.PENDING, TaskStatus.RUNNING})


class TaskKind(str, Enum):
    """Which kind of work a task row is running."""

    INGEST_LORA = "ingest_lora"
    GENERATE_TEACHER = "generate_teacher"


@dataclass(frozen=True, slots=True)
class DatasetTask:
    """One task row (``params`` is the decoded JSON launch payload).

    ``status`` and ``kind`` are the enums above; repositories convert the
    stored words once, at the edge, so nothing in between re-parses them.
    """

    id: int
    dataset: str
    kind: TaskKind | str
    status: TaskStatus | str
    pid: int | None
    current: int
    total: int
    error: str | None
    params: dict = field(default_factory=dict)
    created_at: datetime = datetime.min
    updated_at: datetime = datetime.min
    #: The device-MB claim the admission ledger held when this row was
    #: admitted (MEM-03, ADR 0005). Written once with the row; read back
    #: on startup to rebuild the ledger. None means *no claim exists*
    #: (a row from before the column, or a container that could not
    #: build a ledger) -- never "claimed, size unknown".
    reserved_mb: float | None = None


class DatasetTasks(ABC):
    """Row store for dataset tasks; repositories do not raise."""

    @abstractmethod
    def add(
        self, *, dataset: str, kind: str, total: int, params: dict,
        reserved_mb: float | None = None,
    ) -> DatasetTask:
        """Insert a ``pending`` task; binds its id.

        ``reserved_mb`` travels with the row: the admission claim taken
        before the insert (a refusal writes no row). None only when no
        claim exists.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, task_id: int) -> DatasetTask | None:
        raise NotImplementedError

    @abstractmethod
    def list_for(
        self, dataset: str, *, active_only: bool = False
    ) -> tuple[DatasetTask, ...]:
        """Tasks of one dataset, newest first."""
        raise NotImplementedError

    @abstractmethod
    def find_active(self, dataset: str) -> DatasetTask | None:
        """The newest pending/running task of a dataset, or None."""
        raise NotImplementedError

    @abstractmethod
    def list_unfinished(self) -> tuple[DatasetTask, ...]:
        """All pending/running tasks, newest first (reconcile's input)."""
        raise NotImplementedError

    @abstractmethod
    def update_progress(
        self, task_id: int, current: int, pid: int | None = None
    ) -> bool:
        """CAS: active -> running with progress (+ pid when given)."""
        raise NotImplementedError

    @abstractmethod
    def finalize_if_active(
        self, task_id: int, status: TaskStatus, *, error: str | None = None
    ) -> bool:
        """CAS: active -> ``status`` (a terminal one). False when someone
        already won.

        ``error`` is written only when given, so ``finished`` and
        ``killed`` leave whatever reason was there alone -- a killed task
        keeps the error that explains why it was stuck, rather than
        having it overwritten with the fact that it was stopped.

        One method, not one per terminal status: the three that used to
        exist were the same statement with a different word in it, and
        ``TaskStatus.is_terminal`` is where the vocabulary is enforced.
        """
        raise NotImplementedError
