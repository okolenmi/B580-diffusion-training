"""CallbackEventBus -- in-process, thread-safe fan-out of domain events.

Handlers run **on the publisher's thread**, so publishers (use cases,
the training monitor thread) never block on transport concerns; the SSE
subscriber re-dispatches onto the event loop itself. A failing handler
is logged and skipped -- one broken subscriber must not stop the rest
or abort the use case that published.
"""

from __future__ import annotations

import logging
import threading

from ...application.ports.event_bus import (
    EventBus,
    EventHandler,
    Subscription,
)
from ...domain.events import DomainEvent

logger = logging.getLogger(__name__)


class _CallbackSubscription(Subscription):
    def __init__(self, bus: CallbackEventBus, handler: EventHandler) -> None:
        self._bus = bus
        self._handler = handler
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._bus._unsubscribe(self._handler)


class CallbackEventBus(EventBus):
    """No history, no replay: subscribers only see later events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handlers: list[EventHandler] = []

    def publish(self, event: DomainEvent) -> None:
        with self._lock:
            handlers = tuple(self._handlers)
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "event handler failed for %s; continuing with remaining handlers",
                    type(event).__name__,
                )

    def subscribe(self, handler: EventHandler) -> Subscription:
        with self._lock:
            self._handlers.append(handler)
        return _CallbackSubscription(self, handler)

    def _unsubscribe(self, handler: EventHandler) -> None:
        with self._lock:
            try:
                self._handlers.remove(handler)
            except ValueError:
                pass  # already removed -- close() is idempotent
