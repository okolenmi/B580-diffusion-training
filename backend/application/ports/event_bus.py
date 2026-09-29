"""EventBus port -- publish/subscribe for domain events."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from ...domain.events import DomainEvent

EventHandler = Callable[[DomainEvent], None]
"""Receives every published event; called on the publisher's thread."""


class Subscription(ABC):
    """Handle for one active subscription; ``close()`` is idempotent."""

    @abstractmethod
    def close(self) -> None:
        """Stop receiving events."""
        raise NotImplementedError


class EventBus(ABC):
    """Fan-out of domain events to independent subscribers.

    No history and no replay: a subscriber only sees events published
    after it subscribed. Implementations must be safe to call from any
    thread.
    """

    @abstractmethod
    def publish(self, event: DomainEvent) -> None:
        """Deliver ``event`` to every current subscriber."""
        raise NotImplementedError

    @abstractmethod
    def subscribe(self, handler: EventHandler) -> Subscription:
        """Register ``handler`` for all future events."""
        raise NotImplementedError
