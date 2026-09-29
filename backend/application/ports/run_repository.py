"""RunRepository port -- persistence for the Run aggregate."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...domain.entities.run import Run
from ...domain.value_objects import RunId, RunStatus


class RunRepository(ABC):
    """CRUD surface for runs, newest-first listing semantics."""

    @abstractmethod
    def add(self, run: Run) -> Run:
        """Persist a new run and bind its id via ``run.assign_id``.

        Raises ``DomainError`` if the run already has an id.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, run_id: RunId) -> Run | None:
        """Fetch one run, or ``None`` when it does not exist."""
        raise NotImplementedError

    @abstractmethod
    def list(self, *, limit: int = 50, status: RunStatus | None = None) -> list[Run]:
        """Newest-first page of runs, optionally filtered by status."""
        raise NotImplementedError

    @abstractmethod
    def update(self, run: Run) -> bool:
        """Persist mutations of an existing run.

        Returns ``False`` when no row matched (the caller decides what
        that means -- repositories do not raise application errors).
        """
        raise NotImplementedError

    @abstractmethod
    def find_active(self) -> Run | None:
        """The most recently started run that is currently ``running``."""
        raise NotImplementedError

    @abstractmethod
    def delete_all(self) -> int:
        """Wipe the run history; returns the number of rows removed."""
        raise NotImplementedError
