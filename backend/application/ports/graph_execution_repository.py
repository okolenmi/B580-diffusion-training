"""GraphExecutionRepository port -- persistence for GraphExecution rows.

Same contract as ``RunRepository``: the CAS (``update_if_status``) is
the primitive every status transition goes through, so the supervisor
thread, a stop request, and startup reconciliation can race to finalise
one execution and exactly one writer wins; losers see ``False`` and
discard their outcome.

Statuses: ``queued`` (row inserted, thread not yet claimed it) ->
``running`` -> exactly one of ``finished`` | ``error`` | ``stopped``.
The stored ``graph`` column is the submission snapshot; ``results``
grows node by node while running (partial results survive a crash --
reconciliation then fails the row with what got recorded).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...domain.entities.graph_execution import GraphExecution
from ...domain.value_objects import ExecutionId, GraphStatus


class GraphExecutionRepository(ABC):
    """CRUD for graph executions, newest-first listing semantics."""

    @abstractmethod
    def add(self, execution: GraphExecution) -> GraphExecution:
        """Persist a new ``queued`` execution and bind its id via
        ``execution.assign_id``.

        Raises ``DomainError`` if the execution already has an id.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, execution_id: ExecutionId) -> GraphExecution | None:
        """Fetch one execution, or ``None`` when it does not exist."""
        raise NotImplementedError

    @abstractmethod
    def list(self, *, limit: int = 50) -> list[GraphExecution]:
        """Newest-first page of executions."""
        raise NotImplementedError

    @abstractmethod
    def update(self, execution: GraphExecution) -> bool:
        """Persist mutations; ``False`` when no row matched."""
        raise NotImplementedError

    @abstractmethod
    def update_if_status(
        self, execution: GraphExecution, expected: GraphStatus
    ) -> bool:
        """Persist only if the stored row still has ``expected`` status.

        The compare-and-swap that makes racing finalisers safe. Returns
        ``False`` when someone else already won.
        """
        raise NotImplementedError

    @abstractmethod
    def find_active(self) -> GraphExecution | None:
        """The newest ``queued``/``running`` execution, or None.

        Active counts as single-slot: a second ``StartGraphExecution``
        must be refused while one lives (one B580, in-process training
        nodes -- see doc 05 section 5).
        """
        raise NotImplementedError

    @abstractmethod
    def list_unfinished(self) -> list[GraphExecution]:
        """All ``queued``/``running`` rows, newest first (reconcile's input)."""
        raise NotImplementedError

    @abstractmethod
    def delete_all(self) -> int:
        """Wipe execution history; returns the number of rows removed.

        Deletes even active rows (mirrors ``RunRepository.delete_all``):
        a worker that loses its row just loses every subsequent CAS and
        exits quietly.
        """
        raise NotImplementedError
