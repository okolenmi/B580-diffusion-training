"""CallbackEventBus -- in-process, thread-safe fan-out of domain events.

Handlers run **on the publisher's thread**, so publishers (use cases,
the training monitor thread) never block on transport concerns; the SSE
subscriber re-dispatches onto the event loop itself. A failing handler
is logged and skipped -- one broken subscriber must not stop the rest
or abort the use case that published.

Since 2026-10-02 this also assigns each event a **sequence number** and
keeps a bounded ring of recent lifecycle events, so a client that
reconnects can ask for what it missed
(`docs/design/backend/09-event-contract.md`). Two consequences worth
stating:

* the ring holds **lifecycle** events only. Deltas are coalesced per
  client anyway, and the value of an old progress sample is negative --
  so buffering them would cost memory for nothing.
* sequence numbers are **process-local**. They start at 1 and there is no
  attempt to make them comparable across a restart; a client asking for
  an id from a previous process is told `complete=False` and refetches.
"""

from __future__ import annotations

import logging
import threading
from collections import deque

from ...application.limits import EVENT_REPLAY_RING
from ...application.ports.event_bus import (
    EventBus,
    EventHandler,
    Replay,
    Sequenced,
    Subscription,
)
from ...domain.events import DomainEvent
from ...application.event_delivery import is_lifecycle

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
    """In-process fan-out with a bounded lifecycle replay ring."""

    def __init__(self, *, replay_size: int = EVENT_REPLAY_RING) -> None:
        self._lock = threading.Lock()
        self._handlers: list[EventHandler] = []
        self._replay: deque[Sequenced] = deque(maxlen=replay_size)
        self._next_seq = 1

    # -- publish ---------------------------------------------------------

    def publish(self, event: DomainEvent) -> None:
        # The number is assigned under the same lock that reads the
        # handler list, so handlers always see an increasing seq even when
        # two threads publish at once. Assignment *before* dispatch, not
        # after: a subscriber that receives seq 7 and then asks the ring
        # must already find 7 in it.
        with self._lock:
            sequenced = Sequenced(seq=self._next_seq, event=event)
            self._next_seq += 1
            if is_lifecycle(event.event_type):
                self._replay.append(sequenced)
            handlers = tuple(self._handlers)
        for handler in handlers:
            try:
                handler(sequenced)
            except Exception:
                logger.exception(
                    "event handler failed for %s; continuing with remaining handlers",
                    type(event).__name__,
                )

    def subscribe(self, handler: EventHandler) -> Subscription:
        with self._lock:
            self._handlers.append(handler)
        return _CallbackSubscription(self, handler)

    # -- replay ----------------------------------------------------------

    def replay_since(self, last_seq: int) -> Replay:
        """Buffered lifecycle events after ``last_seq``.

        Subscribe first, then ask: events published in between are
        delivered live, so a caller has to drop anything the replay
        already covered (`presentation/sse.py` does, by seq).
        """
        with self._lock:
            newest = self._next_seq - 1
            ring = tuple(self._replay)

        if last_seq >= newest and last_seq > 0:
            # Either the client is up to date, or -- and this is the case
            # the design note is about -- its id is from a process that
            # published more events than this one has. Both mean there is
            # nothing here to give it, and the second must not be
            # reported as a clean "nothing missed".
            if last_seq > newest:
                return Replay(events=(), complete=False)
            return Replay(events=(), complete=True)

        after = tuple(item for item in ring if item.seq > last_seq)
        if not ring:
            # Nothing was ever ringed, so nothing can be promised about
            # the lifecycle events this client missed.
            return Replay(events=(), complete=False)
        oldest = ring[0].seq
        # The client's id must be *adjacent* to what we still hold. If
        # anything was evicted in between, the replay has a hole.
        complete = last_seq + 1 >= oldest
        return Replay(events=after, complete=complete)

    @property
    def last_seq(self) -> int:
        """Highest sequence number assigned so far."""
        with self._lock:
            return self._next_seq - 1

    def _unsubscribe(self, handler: EventHandler) -> None:
        with self._lock:
            try:
                self._handlers.remove(handler)
            except ValueError:
                pass  # already removed -- close() is idempotent