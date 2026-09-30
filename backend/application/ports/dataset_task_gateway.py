"""DatasetTaskGateway port -- spawning/observing ingestion children.

The fork task gateway: a child process runs the manager ingestion
builder and reports progress by writing task rows straight into
backend.db (WAL makes the side-by-side writer safe), so this port only
needs spawn/kill/liveness -- there is no progress channel to manage.

``is_alive`` must be PID-reuse aware (a bare ``kill(pid, 0)`` lies
after a reboot), hence the marker-in-cmdline contract on the adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetTaskLaunch:
    """Everything the child needs; ``params`` stays JSON-serialisable."""

    task_id: int
    dataset_root: Path
    kind: str
    params: dict


class DatasetTaskGateway(ABC):
    @abstractmethod
    def spawn(self, launch: DatasetTaskLaunch) -> int:
        """Start the child; returns its pid.

        Raises ``DatasetTaskLaunchError`` when the process could not be
        started at all (nothing to reap, row must be failed by caller).
        """
        raise NotImplementedError

    @abstractmethod
    def kill(self, pid: int) -> None:
        """Force-kill the task process group; fail-open on dead pids."""
        raise NotImplementedError

    @abstractmethod
    def is_alive(self, pid: int) -> bool:
        """True only for a live process that is one of our task children."""
        raise NotImplementedError
