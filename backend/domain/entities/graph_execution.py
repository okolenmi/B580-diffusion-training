"""GraphExecution -- one node-graph run, modelled as a state machine.

Same posture as ``Run``: the entity owns its invariants.

* status only moves along the transition table below -- anything else
  raises ``InvalidTransitionError`` (``status`` is read-only);
* per-node results may only be appended while ``running``;
* terminal states are final;
* lifecycle transitions buffer a domain event, drained by the
  application layer via ``collect_events()`` after persistence;
* timestamps are passed in (``at=...``), never read from a clock.

The stored ``graph`` is the submission snapshot (what actually ran,
independent of later edits or library changes).
"""

from __future__ import annotations

from datetime import datetime

from ..events import (
    DomainEvent,
    GraphExecutionFailed,
    GraphExecutionFinished,
    GraphExecutionQueued,
    GraphExecutionStarted,
    GraphExecutionStopped,
)
from ..exceptions import DomainError, InvalidTransitionError
from ..graph import GraphDefinition, NodeResult
from ..value_objects import ExecutionId, GraphStatus


class GraphExecution:
    """A graph execution with enforced lifecycle rules."""

    _ALLOWED: dict[GraphStatus, frozenset[GraphStatus]] = {
        # ERROR is reachable from QUEUED: startup reconciliation fails
        # rows a dead process never got to, and validation-layer
        # surprises can fail a row before its thread claims it.
        GraphStatus.QUEUED: frozenset(
            {GraphStatus.RUNNING, GraphStatus.STOPPED, GraphStatus.ERROR}
        ),
        GraphStatus.RUNNING: frozenset(
            {GraphStatus.FINISHED, GraphStatus.ERROR, GraphStatus.STOPPED}
        ),
        GraphStatus.FINISHED: frozenset(),
        GraphStatus.ERROR: frozenset(),
        GraphStatus.STOPPED: frozenset(),
    }

    def __init__(
        self,
        *,
        status: GraphStatus,
        graph: GraphDefinition,
        created_at: datetime,
        id: ExecutionId | None = None,
        results: tuple[NodeResult, ...] = (),
        error: str | None = None,
        updated_at: datetime | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> None:
        if isinstance(status, str) and not isinstance(status, GraphStatus):
            status = GraphStatus(status)

        self._id: ExecutionId | None = id
        self._status: GraphStatus = status
        self._events: list[DomainEvent] = []

        self.graph = graph
        self.results: tuple[NodeResult, ...] = tuple(results)
        self.error = error
        self.created_at = created_at
        self.updated_at = updated_at if updated_at is not None else created_at
        self.started_at = started_at
        self.finished_at = finished_at

    # ------------------------------------------------------------------
    # Construction / identity
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, *, graph: GraphDefinition, created_at: datetime) -> "GraphExecution":
        """Register a new execution in ``queued`` state (no id yet)."""
        return cls(status=GraphStatus.QUEUED, graph=graph, created_at=created_at)

    @property
    def id(self) -> ExecutionId | None:
        """Persisted id, or ``None`` until the repository assigns one."""
        return self._id

    @property
    def status(self) -> GraphStatus:
        """Current lifecycle state (read-only -- no setter exists)."""
        return self._status

    def assign_id(self, execution_id: ExecutionId) -> None:
        """Bind a persisted id; emits ``GraphExecutionQueued``.

        Called by the repository right after the INSERT. Exactly once:
        re-binding would emit events claiming a different identity than
        the ones already buffered.
        """
        if self._id is not None:
            raise DomainError(f"execution already has id {self._id}")
        if execution_id < 1:
            raise DomainError(f"execution id must be a positive integer, got {execution_id}")
        self._id = execution_id
        self._events.append(
            GraphExecutionQueued(
                execution_id=execution_id,
                node_count=len(self.graph.nodes),
                occurred_at=self.created_at,
            )
        )

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def mark_running(self, *, at: datetime) -> None:
        """Worker claimed the row: ``queued`` -> ``running``."""
        execution_id = self._require_id()
        self._transition(to=GraphStatus.RUNNING, at=at)
        self.started_at = at
        self._emit(GraphExecutionStarted(execution_id=execution_id, occurred_at=at))

    def record_result(self, result: NodeResult, *, at: datetime) -> None:
        """Append one node's outcome (``running`` only, no event).

        High-frequency-ish telemetry deliberately does not emit domain
        events from here -- the supervisor publishes
        ``GraphExecutionProgressed`` itself after persisting; this
        method only grows the results tuple and stamps ``updated_at``.
        """
        self._require_status(GraphStatus.RUNNING, action="record a node result")
        self._require_id()
        self.results = (*self.results, result)
        self.updated_at = at

    def mark_finished(self, *, at: datetime) -> None:
        """Every node built: ``running`` -> ``finished``."""
        execution_id = self._require_id()
        self._transition(to=GraphStatus.FINISHED, at=at)
        self._emit(
            GraphExecutionFinished(
                execution_id=execution_id, nodes=len(self.results), occurred_at=at
            )
        )

    def mark_failed(self, *, at: datetime, error: str) -> None:
        """Failure: ``queued``/``running`` -> ``error`` with the reason."""
        execution_id = self._require_id()
        self._transition(to=GraphStatus.ERROR, at=at)
        self.error = error
        self._emit(
            GraphExecutionFailed(execution_id=execution_id, error=error, occurred_at=at)
        )

    def stop(self, *, at: datetime, reason: str = "stop requested") -> None:
        """Cancel: ``queued``/``running`` -> ``stopped``.

        ``reason`` is recorded in ``error`` -- for a stopped execution
        that field is the termination note, not a failure.
        """
        execution_id = self._require_id()
        self._transition(to=GraphStatus.STOPPED, at=at)
        self.error = reason
        self._emit(
            GraphExecutionStopped(
                execution_id=execution_id, reason=reason, occurred_at=at
            )
        )

    # ------------------------------------------------------------------
    # Event buffer
    # ------------------------------------------------------------------

    def collect_events(self) -> list[DomainEvent]:
        """Drain and return buffered events (second call returns [])."""
        drained = self._events
        self._events = []
        return drained

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _transition(self, *, to: GraphStatus, at: datetime) -> None:
        allowed = self._ALLOWED[self._status]
        if to not in allowed:
            where = self._id if self._id is not None else "<new>"
            raise InvalidTransitionError(
                f"execution {where}: {self._status.value} -> {to.value} is not allowed"
            )
        self._status = to
        self.updated_at = at
        if to.is_terminal:
            self.finished_at = at

    def _require_status(self, status: GraphStatus, *, action: str) -> None:
        if self._status is not status:
            raise InvalidTransitionError(
                f"execution {self._id if self._id is not None else '<new>'}: "
                f"cannot {action} while {self._status.value} (needs {status.value})"
            )

    def _require_id(self) -> ExecutionId:
        if self._id is None:
            raise DomainError("execution has no id yet -- persist it first")
        return self._id

    def _emit(self, event: DomainEvent) -> None:
        self._events.append(event)

    def __repr__(self) -> str:
        where = self._id if self._id is not None else "<new>"
        return (
            f"GraphExecution(id={where}, status={self._status.value}, "
            f"{len(self.results)}/{len(self.graph.nodes)} nodes)"
        )
