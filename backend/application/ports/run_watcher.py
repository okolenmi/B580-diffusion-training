"""RunWatcher port -- "somebody is watching this run" as a dependency.

The supervisors that own run watching (`RunSupervisor`) are
collaborators like any other: a use case should state that it *starts*
or *re-attaches* a watcher, not which concrete class does it. Both
call sites fire and forget -- the thread is the supervisor's business,
including its name and its daemon flag.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...domain.value_objects import RunId


class RunWatcher(ABC):
    """Start or re-attach background watching of a run's process."""

    @abstractmethod
    def watch(self, *, run_id: RunId, pid: int, progress_path: Path) -> None:
        """Follow a trainer this process just spawned."""
        raise NotImplementedError

    @abstractmethod
    def adopt(self, *, run_id: RunId, pid: int, progress_path: Path) -> None:
        """Re-attach to a trainer that outlived the server (docs 07 F-11)."""
        raise NotImplementedError