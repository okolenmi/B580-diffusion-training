"""EventPublisher -- "publish what this aggregate decided".

Nine places across the application layer carried the same three lines::

    for event in run.collect_events():
        self._events.publish(event)

The loop looks trivial, but it is the rule that keeps the event stream
ordered and non-duplicated: entities buffer their domain events until
a writer drains them exactly once, and only the writer that won the
status compare-and-swap may publish. Spelling it out nine times is nine
chances to publish from a writer that lost (docs 08 S-07).

This collaborator owns it. It takes anything that can ``collect_events``
-- ``Run``, ``GraphExecution`` -- so a new aggregate gets the behaviour
for free, and the use cases stop holding an ``EventBus`` only to drain
a buffer.
"""

from __future__ import annotations

from typing import Protocol

from .ports.event_bus import EventBus
from ..domain.events import DomainEvent


class EventSource(Protocol):
    """Anything that buffers domain events until a writer drains them."""

    def collect_events(self) -> list[DomainEvent]: ...


class EventPublisher:
    def __init__(self, *, events: EventBus) -> None:
        self._events = events

    def emit(self, event: DomainEvent) -> None:
        """Publish one event that is not buffered on an aggregate.

        Telemetry is the case: progress events are produced by the
        supervisor, not by an entity transition, so there is no buffer to
        drain.
        """
        self._events.publish(event)

    def publish(self, source: EventSource) -> int:
        """Drain and publish; returns how many events went out."""
        count = 0
        for event in source.collect_events():
            self._events.publish(event)
            count += 1
        return count

    def publish_all(self, *sources: EventSource) -> int:
        """Publish several buffers in order (a start repairs two rows)."""
        return sum(self.publish(source) for source in sources)