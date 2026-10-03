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
    stated in `docs/design/backend/09-event-contract.md`. What a client
    sends back is therefore not a bare `seq` but an `EventCursor`, which
    pairs the number with the epoch of the process that issued it -- so
    "id 40" from a previous process is recognised as such instead of
    being mistaken for a position in this one.
    """

    seq: int
    event: DomainEvent

    @property
    def event_type(self) -> str:
        """The wire name, without the caller having to unwrap first."""
        return self.event.event_type


@dataclass(frozen=True, slots=True)
class EventCursor:
    """Where a client got to: which server process, and how far into it.

    The sequence number alone cannot say that. Numbers restart at 1 in
    every process, so a client that reconnects after a restart carrying
    ``Last-Event-ID: 40`` is indistinguishable from one that is genuinely
    40 events into this process -- and the bus can only recognise the
    first case by noticing that 40 is *ahead* of anything it has
    published. Once the new process has itself published more than 40
    events, that test passes and the client's id is accepted as valid, so
    it is told its history is continuous when it is not, and skips the
    refetch that would have corrected it.

    Measured on the round-3 reproduction: `replay_since(40)` against a new
    process holding 60 events returned `complete=True` and events 41..60.

    The epoch is what makes the two cases separable. It is minted per bus
    and never repeats, and the pair travels as one opaque string on the
    wire, because that is the shape `EventSource` hands back verbatim in
    `Last-Event-ID` -- the browser never parses it and the frontend never
    inspects it.
    """

    epoch: str
    seq: int

    def wire(self) -> str:
        """The form a client sends back. Opaque to the client by design."""
        return f"{self.epoch}:{self.seq}"

    @classmethod
    def parse(cls, raw: str) -> EventCursor | None:
        """Read a client-supplied id, or ``None`` if it is not one of ours.

        ``None`` covers all three ways a header can be unusable -- absent,
        malformed, or a bare number from a client predating the epoch --
        and all three mean the same thing to the caller: this client has
        told us nothing we can act on.
        """
        epoch, separator, seq = raw.strip().partition(":")
        if not separator or not epoch:
            return None
        try:
            value = int(seq)
        except ValueError:
            return None
        if value <= 0:
            return None
        return cls(epoch=epoch, seq=value)


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

    @property
    @abstractmethod
    def epoch(self) -> str:
        """This process's event-stream epoch. Minted once, never repeats."""
        raise NotImplementedError

    def replay_since(self, cursor: EventCursor | None) -> Replay:
        """Buffered lifecycle events published after ``cursor``.

        ``complete=False`` means the bus cannot honour the request: the
        cursor is from a previous process, or older than the ring, or the
        client sent nothing usable. A client that gets an incomplete
        replay must refetch, which is what the ``resync_required`` flag in
        the stream's opening frame tells it to do.
        """
        raise NotImplementedError
