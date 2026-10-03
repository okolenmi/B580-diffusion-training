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

The *statement* is shared with the other aggregate that has a terminal
state. ``infrastructure/persistence/cas.py`` holds it, and dataset-task
rows go through the same one; what this service adds on top is the
second half, announcing. Dataset tasks have no events, so they call the
statement directly and publish nothing.

So the two halves now have one definition each, rather than one rule
written twice: the swap in ``cas.py``, the announcement here. Giving
dataset tasks events would let them use this service too, and is a
feature decision rather than a cleanup.
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
    """The graph-execution flavour, and where its terminal-repair rule lives.

    Three callers need "this row cannot still be running -- put it in a
    terminal state and say so": the supervisor that owns the thread, the
    startup sweep that finds rows a dead process left, and (before
    2026-10-02) the run supervisor's equivalent. Each wrote its own copy
    of the same three steps -- read the status, mark failed with a note,
    CAS from the status just read -- and a fix to that sequence would have
    had to be made three times.

    ``fail_if_unfinished`` is that sequence once. The *note* stays the
    caller's, deliberately: "the supervisor crashed" and "the server
    restarted" are different facts and only the caller knows which one it
    is reporting.
    """

    def __init__(self, *, repository, events, clock) -> None:
        super().__init__(repository=repository, events=events)
        self._clock = clock

    def fail_if_unfinished(self, execution: GraphExecution, *, error: str) -> bool:
        """Move an unfinished execution to ``failed`` with ``error``.

        False if it was already terminal, or if another writer moved it
        first -- in which case this changes and announces nothing, which
        is the same outcome the caller would have had by checking first.
        """
        if execution.status.is_terminal:
            return False
        expected = execution.status
        execution.mark_failed(at=self._clock.now(), error=error)
        return self.commit(execution, expected=expected)

    def finalise_from_record(
        self,
        execution: GraphExecution,
        *,
        error: str | None,
        results: tuple,
    ) -> bool:
        """Settle an unfinished execution using what the run itself recorded.

        The same compare-and-swap-then-announce sequence as
        ``fail_if_unfinished``, with two differences that are the whole
        point of a separate method rather than a parameter:

        * the verdict is the run's, not the caller's -- ``error is None``
          means it finished, and a caller that could only ever write
          "failed" would report a completed run as a crash;
        * the node results are persisted on the way through, because a run
          that completed unobserved produced them and the row is the only
          place they will ever be readable.

        ``results`` is a plain tuple rather than a ``NodeResult`` tuple so
        this module does not have to name the domain graph types; the
        entity's own validation still applies to each one.
        """
        if execution.status.is_terminal:
            return False
        expected = execution.status
        now = self._clock.now()
        for result in results:
            execution.record_result(result, at=now)
        if error is None:
            execution.mark_finished(at=now)
        else:
            execution.mark_failed(at=now, error=error)
        return self.commit(execution, expected=expected)