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
    def list_runs(self, *, limit: int = 50,
                    status: RunStatus | None = None) -> list[Run]:
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
    def update_if_status(self, run: Run, expected: RunStatus) -> bool:
        """Persist only if the stored row still has ``expected`` status.

        Compare-and-swap on the status column: when the supervisor, a
        stop request, and startup reconciliation race to finalise the
        same run, exactly one writer wins and the losers discard their
        outcome. Returns ``False`` when someone else already won.
        """
        raise NotImplementedError

    @abstractmethod
    def continue_ids_above(self, run_id: RunId) -> None:
        """Guarantee that the next ``add`` binds an id > ``run_id``.

        Id allocation is the repository's job, and a fresh database
        starts at 1 while ``runs/run_1/`` may already hold a run from the
        legacy server -- writing there would truncate its log. Startup
        seeds the sequence above the highest existing run directory
        (docs 07 F-04). Never moves the sequence backwards.
        """
        raise NotImplementedError

    @abstractmethod
    def find_active(self) -> Run | None:
        """The most recent unfinished run (``created`` or ``running``).

        ``created`` counts as active: a run mid-launch must block a
        second start and reconciliation must be able to sweep it.
        """
        raise NotImplementedError

    @abstractmethod
    def list_unfinished(self) -> list[Run]:
        """All runs in ``created``/``running`` state, newest first.

        Normally at most one; more means a previous process died mid-
        launch or a race left debris -- reconciliation sweeps them.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_all(self) -> int:
        """Wipe the run history; returns the number of rows removed."""
        raise NotImplementedError
