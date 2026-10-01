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

ACTIVE_TASK_STATUSES: tuple[str, ...] = ("pending", "running")
TERMINAL_TASK_STATUSES: tuple[str, ...] = ("finished", "failed", "killed")

KIND_INGEST_LORA = "ingest_lora"
KIND_GENERATE_TEACHER = "generate_teacher"
TASK_KINDS: tuple[str, ...] = (KIND_INGEST_LORA, KIND_GENERATE_TEACHER)


@dataclass(frozen=True, slots=True)
class DatasetTask:
    """One task row (``params`` is the decoded JSON launch payload)."""

    id: int
    dataset: str
    kind: str
    status: str
    pid: int | None
    current: int
    total: int
    error: str | None
    params: dict = field(default_factory=dict)
    created_at: datetime = datetime.min
    updated_at: datetime = datetime.min


class DatasetTasks(ABC):
    """Row store for dataset tasks; repositories do not raise."""

    @abstractmethod
    def add(
        self, *, dataset: str, kind: str, total: int, params: dict
    ) -> DatasetTask:
        """Insert a ``pending`` task; binds its id."""
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
    def finish_if_active(self, task_id: int) -> bool:
        """CAS: active -> finished. False when someone already won."""
        raise NotImplementedError

    @abstractmethod
    def fail_if_active(self, task_id: int, error: str) -> bool:
        """CAS: active -> failed with reason."""
        raise NotImplementedError

    @abstractmethod
    def kill_if_active(self, task_id: int) -> bool:
        """CAS: active -> killed (the stop path)."""
        raise NotImplementedError
