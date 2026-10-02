"""Lifecycle writers -- "persist this decision, and only announce it if
you won the row".

Every terminal path in the application layer ended with the same four
lines, written six times for runs and six more for executions::

    if not repo.update_if_status(aggregate, expected=expected):
        return False          # someone else finished this row first
    events.publish(aggregate)  # ... and only the winner announces it

The order is the rule, not the boilerplate: the entity buffers its
events, the writer drains them, and a writer that *lost* the
compare-and-swap must publish nothing -- otherwise the stream announces
a state the database does not hold, and the dashboard renders a run that
does not exist. Spelling that out at each call site is how one of them
inverted the two statements (docs 08 S-13).

One service per aggregate, built in the composition root, with the
repository and the publisher it needs. It also owns the insert-then-
announce shape (``StartTraining``, ``StartGraphExecution``), which is the
same rule with nothing to race against.
"""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from .event_publisher import EventPublisher, EventSource
from ..domain.entities.graph_execution import GraphExecution
from ..domain.value_objects import GraphStatus

# S appears only in parameter position (``expected: S``), which is
# contravariant. Declaring it invariant made StatusRepository narrower
# than intended -- exactly backwards for a protocol whose stated purpose
# is to be *narrower* than the full repository port (mypy: "invariant
# type variable in protocol where contravariant one is expected").
S = TypeVar("S", contravariant=True)
R = TypeVar("R", bound=EventSource)
"""An aggregate that buffers its domain events --
``GraphExecution``, and anything shaped like them."""


class StatusRepository(Protocol[R, S]):
    """The two repository methods a writer uses. Narrower than the
    full ``RunRepository`` / ``GraphExecutionRepository`` ports on
    purpose: this service needs nothing else from storage, so a test can
    satisfy it with two methods."""

    def add(self, aggregate: R) -> R: ...

    def update_if_status(self, aggregate: R, *, expected: S) -> bool: ...


class LifecycleWriter(Generic[R, S]):
    """Persists one aggregate's lifecycle decisions and announces the
    ones that won."""

    def __init__(
        self,
        *,
        repository: StatusRepository[R, S],
        events: EventPublisher,
    ) -> None:
        self._repository = repository
        self._events = events

    def insert(self, aggregate: R) -> R:
        """Persist a brand-new aggregate and announce it.

        Nothing to race against: the row did not exist a moment ago, and
        the repository binds the id the events are emitted with.
        """
        stored = self._repository.add(aggregate)
        self._events.publish(stored)
        return stored

    def commit(self, aggregate: R, *, expected: S, prior: tuple[R, ...] = ()) -> bool:
        """Compare-and-swap the row, then publish -- winner only.

        ``expected`` is the status the writer *read*, which is what makes
        the update atomic: if another writer already moved the row, this
        one changes nothing and announces nothing.

        ``prior`` names aggregates whose buffered events belong on the
        stream *before* this one's -- a start that failed between the
        insert and the launch leaves its ``RunCreated`` buffer behind,
        and the stream has to say created-then-failed, not the other way
        round.
        """
        if not self._repository.update_if_status(aggregate, expected=expected):
            return False
        self._events.publish_all(*prior, aggregate)
        return True


class ExecutionLifecycleWriter(LifecycleWriter[GraphExecution, GraphStatus]):
    """The graph-execution flavour."""