"""GraphExecution -- one node-graph run, modelled as a state machine.

Same posture as ``Run``: the entity owns its invariants, keeps its state
private behind read-only properties, and holds a ``StatusMachine``
rather than restating the guard (docs 08 S-11, S-13).

* status only moves along ``GRAPH_TRANSITIONS`` -- anything else raises
  ``InvalidTransitionError``;
* per-node results may only be appended while ``running``;
* terminal states are final (derived from the table, not restated);
* lifecycle transitions buffer a domain event, drained by the
  application layer via ``collect_events()`` after persistence;
* timestamps are passed in (``at=...``), never read from a clock.

The stored ``graph`` is the submission snapshot (what actually ran,
independent of later edits or library changes).

Two ways in: ``create`` for a new execution, ``restore`` for a row
loaded from the database (which validates the cross-field rules --
docs 08 S-16).
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
from ..exceptions import DomainError
from ..graph import GraphDefinition, NodeResult
from ..lifecycle import StatusMachine
from ..memory_settings import EffectiveMemory
from ..value_objects import GRAPH_TRANSITIONS, ExecutionId, GraphStatus


class GraphExecution:
    """A graph execution with enforced lifecycle rules."""

    def __init__(
        self,
        *,
        status: GraphStatus,
        graph: GraphDefinition,
        created_at: datetime,
        id: ExecutionId | None = None,
        results: tuple[NodeResult, ...] = (),
        error: str | None = None,
        memory: EffectiveMemory | None = None,
        reserved_mb: float | None = None,
        updated_at: datetime | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> None:
        if isinstance(status, str) and not isinstance(status, GraphStatus):
            status = GraphStatus(status)

        self._life: StatusMachine[GraphStatus, ExecutionId] = StatusMachine(
            transitions=GRAPH_TRANSITIONS,
            status=status,
            label="execution",
            entity_id=id,
        )
        self._graph = graph
        self._results: tuple[NodeResult, ...] = tuple(results)
        self._error = error
        self._memory = memory
        self._reserved_mb = reserved_mb
        self._created_at = created_at
        self._updated_at = updated_at if updated_at is not None else created_at
        self._started_at = started_at
        self._finished_at = finished_at

    # ------------------------------------------------------------------
    # Construction / identity
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        graph: GraphDefinition,
        created_at: datetime,
        memory: EffectiveMemory | None = None,
        reserved_mb: float | None = None,
    ) -> GraphExecution:
        """Register a new execution in ``queued`` state (no id yet)."""
        return cls(
            status=GraphStatus.QUEUED,
            graph=graph,
            created_at=created_at,
            memory=memory,
            reserved_mb=reserved_mb,
        )

    @classmethod
    def restore(cls, **fields: object) -> GraphExecution:
        """Rebuild an execution from a persisted row, cross-field rules
        checked (see ``Run.restore`` for why: a loaded aggregate can be
        impossible as a whole even when every column is well-formed)."""
        status = fields.get("status")
        execution = cls(**fields)  # type: ignore[arg-type]  # a mapper's **fields is dict[str, object]
        # by construction; the constructor wants each field's own
        # type. Not a convenience ignore -- the _require_consistent
        # call below is the runtime check, and the row's types come
        # from the reader.
        #
        # The two-space-then-# form of the reason is not a style
        # choice: mypy 2.4 accepts a trailing reason only after a
        # second '#'. Written as `# type: ignore[arg-type] -- ...`
        # this marker suppresses NOTHING and mypy reports it as an
        # invalid ignore -- which is how the four arg-type errors
        # below this line were visible at all.
        assert isinstance(status, (GraphStatus, str))
        execution._require_consistent(GraphStatus(status))
        return execution

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------

    @property
    def id(self) -> ExecutionId | None:
        """Persisted id, or ``None`` until the repository assigns one."""
        return self._life.entity_id  # type: ignore[return-value]

    @property
    def status(self) -> GraphStatus:
        """Current lifecycle state."""
        return self._life.status

    @property
    def graph(self) -> GraphDefinition:
        return self._graph

    @property
    def memory(self) -> EffectiveMemory | None:
        """The effective memory values computed when this execution was
        admitted (``memory_json``), or ``None`` for a row written before
        that column existed."""
        return self._memory

    @property
    def reserved_mb(self) -> float | None:
        """The device-MB claim the admission ledger held for this row
        (``reserved_mb``), or ``None`` when no claim exists -- a row
        from before the column, or a container with no ledger. Never
        "claimed, size unknown": that shape is what rule 2 forbids."""
        return self._reserved_mb

    @property
    def results(self) -> tuple[NodeResult, ...]:
        return self._results

    @property
    def error(self) -> str | None:
        """Failure text, or the termination note for a stopped run."""
        return self._error

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        return self._updated_at

    @property
    def started_at(self) -> datetime | None:
        return self._started_at

    @property
    def finished_at(self) -> datetime | None:
        return self._finished_at

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def assign_id(self, execution_id: ExecutionId) -> None:
        """Bind a persisted id; emits ``GraphExecutionQueued``.

        Called by the repository right after the INSERT. Exactly once:
        re-binding would emit events claiming a different identity than
        the ones already buffered.
        """
        self._life.bind(
            execution_id,
            GraphExecutionQueued(
                execution_id=execution_id,
                node_count=len(self._graph.nodes),
                occurred_at=self._created_at,
            ),
        )

    def mark_running(self, *, at: datetime) -> None:
        """Worker claimed the row: ``queued`` -> ``running``."""
        execution_id = self.require_id()
        self._life.move(GraphStatus.RUNNING)
        self._started_at = at
        self._updated_at = at
        self._life.buffer(
            GraphExecutionStarted(execution_id=execution_id, occurred_at=at)
        )

    def record_result(self, result: NodeResult, *, at: datetime) -> None:
        """Append one node's outcome (``running`` only, no event).

        High-frequency-ish telemetry deliberately does not emit domain
        events from here -- the supervisor publishes
        ``GraphExecutionProgressed`` itself after persisting; this
        method only grows the results tuple and stamps ``updated_at``.
        """
        self._life.require(GraphStatus.RUNNING, action="record a node result")
        self.require_id()
        self._results = (*self._results, result)
        self._updated_at = at

    def mark_finished(self, *, at: datetime) -> None:
        """Every node built: ``running`` -> ``finished``."""
        execution_id = self.require_id()
        self._life.move(GraphStatus.FINISHED)
        self._updated_at = at
        self._finished_at = at
        self._life.buffer(
            GraphExecutionFinished(
                execution_id=execution_id, nodes=len(self._results), occurred_at=at
            )
        )

    def mark_failed(self, *, at: datetime, error: str) -> None:
        """Failure: ``queued``/``running`` -> ``error`` with the reason."""
        execution_id = self.require_id()
        self._life.move(GraphStatus.ERROR)
        self._error = error
        self._updated_at = at
        self._finished_at = at
        self._life.buffer(
            GraphExecutionFailed(execution_id=execution_id, error=error, occurred_at=at)
        )

    def stop(self, *, at: datetime, reason: str = "stop requested") -> None:
        """Cancel: ``queued``/``running`` -> ``stopped``.

        ``reason`` is recorded in ``error`` -- for a stopped execution
        that field is the termination note, not a failure.
        """
        execution_id = self.require_id()
        self._life.move(GraphStatus.STOPPED)
        self._error = reason
        self._updated_at = at
        self._finished_at = at
        self._life.buffer(
            GraphExecutionStopped(
                execution_id=execution_id, reason=reason, occurred_at=at
            )
        )

    # ------------------------------------------------------------------
    # Event buffer
    # ------------------------------------------------------------------

    def collect_events(self) -> list[DomainEvent]:
        """Drain and return buffered events (second call returns [])."""
        return self._life.drain()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_consistent(self, status: GraphStatus) -> None:
        """Cross-field rules for a *loaded* row (see ``restore``)."""
        if status is GraphStatus.RUNNING and self._started_at is None:
            raise DomainError(f"execution {self.id}: running but started_at is null")
        if status.is_terminal and self._finished_at is None:
            raise DomainError(
                f"execution {self.id}: {status.value} but finished_at is null"
            )
        if (
            status is not GraphStatus.RUNNING
            and self._results
            and len(self._results) > len(self._graph.nodes)
        ):
            raise DomainError(
                f"execution {self.id}: {len(self._results)} results for a "
                f"{len(self._graph.nodes)}-node graph"
            )

    def require_id(self) -> ExecutionId:
        """The persisted id, or ``DomainError`` if the row is not saved.

        ``id`` is ``ExecutionId | None`` because an aggregate has no identity
        until it is inserted. That is the right model, but it means
        every caller that has *just* inserted or loaded the aggregate
        had to either assert or ``# type: ignore`` -- five of them did
        the latter (docs 08 Q2). This is the honest version of that
        assertion: the case where it raises cannot happen, and if it
        ever did, the message says which aggregate and why.
        """
        return self._life.require_id()

    def __repr__(self) -> str:
        return (
            f"GraphExecution(id={self._life.where}, status={self.status.value}, "
            f"{len(self._results)}/{len(self._graph.nodes)} nodes)"
        )