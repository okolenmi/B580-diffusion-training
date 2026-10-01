"""StatusMachine -- the enforced half of a lifecycle.

``Run`` and ``GraphExecution`` both had their own copy of the same five
small pieces: the current status, the transition table, "you must be in
this status to do that", "you need a persisted id first", and the buffer
of events a writer drains after persisting. Five copies of a rule is
five places to forget one (docs 08 S-13).

This class is that rule, once. It is deliberately *not* an inheritance
base for the aggregates -- they keep their own fields and their own
transition methods, and hold a machine rather than inheriting from it,
because what a machine does (guard a status) is smaller than what an
aggregate is (a run, an execution), and inheriting would couple the two
lifecycles for no gain.

Two things stay with the aggregate, because only it knows them:

* the transition *table* (it lives in ``value_objects`` beside the enum,
  which is also what makes ``status.is_terminal`` derivable);
* the timestamps, since stamping them is part of the entity's meaning
  rather than the guard's.

Everything else -- validation, identity, event buffering -- is here.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from .events import DomainEvent
from .exceptions import DomainError, InvalidTransitionError

S = TypeVar("S")
"""A lifecycle status: a ``str``-valued enum member, compared by value."""


class StatusMachine(Generic[S]):
    """Guards one aggregate's status, identity and event buffer.

    ``label`` is what error messages call the aggregate ("run",
    "execution") so the messages are the ones the API has always
    returned.
    """

    def __init__(
        self,
        *,
        transitions: dict[S, frozenset[S]],
        status: S,
        label: str,
        entity_id: object | None = None,
    ) -> None:
        self._transitions = transitions
        self._status = status
        self._label = label
        self._id = entity_id
        self._events: list[DomainEvent] = []

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def status(self) -> S:
        return self._status

    @property
    def entity_id(self) -> object | None:
        """The persisted id, or ``None`` before the INSERT."""
        return self._id

    @property
    def where(self) -> str:
        """How the aggregate is named in error messages."""
        return str(self._id) if self._id is not None else "<new>"

    def ends_lifecycle(self, status: S) -> bool:
        """Is ``status`` a state with no way out?"""
        return not self._transitions[status]

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def move(self, to: S) -> None:
        """Move to ``to``, or raise ``InvalidTransitionError``."""
        if to not in self._transitions[self._status]:
            raise InvalidTransitionError(
                f"{self._label} {self.where}: "
                f"{_value(self._status)} -> {_value(to)} is not allowed"
            )
        self._status = to

    def require(self, status: S, *, action: str) -> None:
        """Raise unless the aggregate is in ``status`` right now."""
        if self._status != status:
            raise InvalidTransitionError(
                f"{self._label} {self.where}: cannot {action} while "
                f"{_value(self._status)} (needs {_value(status)})"
            )

    def bind(self, entity_id: object, event: DomainEvent) -> None:
        """Attach a persisted id and buffer the event that announces it.

        Exactly once: re-binding raises, because the buffered events
        would otherwise claim a different identity than the one they
        were emitted with.
        """
        if self._id is not None:
            raise DomainError(
                f"{self._label} already has id {self._id}"
            )
        if not isinstance(entity_id, int) or isinstance(entity_id, bool) or entity_id < 1:
            raise DomainError(
                f"{self._label} id must be a positive integer, got {entity_id!r}"
            )
        self._id = entity_id
        self._events.append(event)

    def require_id(self) -> object:
        """The persisted id, or ``DomainError`` if there is none yet."""
        if self._id is None:
            raise DomainError(
                f"{self._label} has no id yet -- persist it first"
            )
        return self._id

    # ------------------------------------------------------------------
    # Event buffer
    # ------------------------------------------------------------------

    def buffer(self, event: DomainEvent) -> None:
        self._events.append(event)

    def drain(self) -> list[DomainEvent]:
        """Take the buffered events; a second call returns ``[]``."""
        drained, self._events = self._events, []
        return drained

    def __repr__(self) -> str:
        return f"StatusMachine({self._label}={self.where}, {_value(self._status)})"


def _value(status: object) -> str:
    """The enum's value when there is one, so messages read ``created``
    rather than ``RunStatus.CREATED``."""
    return getattr(status, "value", str(status))