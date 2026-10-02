"""EventBus port -- publish/subscribe for domain events."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from ...domain.events import DomainEvent


@dataclass(frozen=True, slots=True)
class Sequenced:
    """One published event and its position in the stream.

    The sequence number is assigned by the bus at publish time, not by the
    domain: `RunCompleted` has no opinion about being the forty-second
    event, and putting a counter on `DomainEvent` would put a transport
    concern in the layer that must not know about transports.

    **Process-local, and it resets when the process restarts.** That is
    stated in `docs/design/backend/09-event-contract.md` together with
    what it costs: a client reconnecting with a `Last-Event-ID` from a
    previous process is told `resync_required` rather than handed a
    plausible-looking partial replay.
    """

    seq: int
    event: DomainEvent

    @property
    def event_type(self) -> str:
        """The wire name, without the caller having to unwrap first."""
        return self.event.event_type


@dataclass(frozen=True, slots=True)
class Replay:
    """What a bus can still tell a reconnecting client.

    ``complete`` is specifically about **lifecycle** events: deltas are
    never ring-buffered, because they are coalesced per client anyway and
    the value of an old progress sample is negative. So ``complete`` does
    not mean "you missed nothing", it means "every lifecycle event after
    your last id is in ``events``", which is the claim a client can act on.
    """

    events: tuple[Sequenced, ...]
    complete: bool


EventHandler = Callable[[Sequenced], None]
"""Receives every published event, with its sequence number; called on
the publisher's thread."""


class Subscription(ABC):
    """Handle for one active subscription; ``close()`` is idempotent."""

    @abstractmethod
    def close(self) -> None:
        """Stop receiving events."""
        raise NotImplementedError


class EventBus(ABC):
    """Fan-out of domain events to independent subscribers.

    A subscriber only sees events published after it subscribed; there is
    no history on the *subscriber* side. The bus itself keeps a bounded
    ring of recent **lifecycle** events so a client that reconnects can
    ask for what it missed -- see `replay_since` and
    `docs/design/backend/09-event-contract.md`. Implementations must be
    safe to call from any thread.
    """

    @abstractmethod
    def publish(self, event: DomainEvent) -> None:
        """Deliver ``event`` to every current subscriber, with a sequence
        number this bus assigns."""
        raise NotImplementedError

    @abstractmethod
    def subscribe(self, handler: EventHandler) -> Subscription:
        """Register ``handler`` for all future events."""
        raise NotImplementedError

    @abstractmethod
    def replay_since(self, last_seq: int) -> Replay:
        """Buffered lifecycle events published after ``last_seq``.

        ``complete=False`` means the bus cannot honour the request: the
        id is from a previous process, or older than the ring. A client
        that gets an incomplete replay must refetch, which is what the
        ``resync_required`` flag in the stream's opening frame tells it to
        do.
        """
        raise NotImplementedError
