"""SSE bridge -- EventBus port -> text/event-stream response.

Threading contract: bus handlers run on the *publisher's* thread (any
thread -- use cases, the training monitor). The bridge therefore hops
onto the event loop with ``call_soon_threadsafe`` before touching the
client buffer.

Backpressure (docs 07 F-09, corrected by docs 08 N-04): the buffer is
bounded, and *how* overflow is resolved follows each event kind's
**delivery class**, not its recency. That distinction is the whole
design; getting it wrong silently loses information the client needs.

======================  ==========  ====================================
Kind                    Class       Policy
======================  ==========  ====================================
``run_progressed``      state       coalesce per ``run_id``: a newer
                                    sample supersedes the queued one
                                    for the same run, and only that
                                    one.
``graph_execution_      delta       never coalesced, never evicted.
progressed``                        Every node's completion is a fact
                                    that happened; six of them must
                                    arrive as six. Evicting them was
                                    what dropped 6 node events to 1
                                    (reproduced, docs 08 N-04).
everything else          lifecycle   never dropped for the sake of a
                                    newer event of any other kind.
======================  ==========  ====================================

Only at `QUEUE_MAX`, and only if nothing above applies, is the oldest
frame dropped -- whichever class it is. Every drop is counted and
logged; unbounded growth would trade lost events for unbounded memory,
and the client's refetch-on-(re)open plus the dashboard's slow poll
close that gap.

Serialization: frames go through :func:`backend.json_safe.strict_dumps`,
so a diverged loss is ``null`` + a ``nonfinite`` marker rather than a
bare ``NaN`` token the browser cannot parse (docs 07 F-03).

Heartbeat every ``HEARTBEAT_SECONDS`` (``: ping`` comment) keeps
proxies from reaping an idle stream and lets the generator notice a
dead client promptly.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import asdict
from datetime import datetime, UTC

from fastapi import Request
from fastapi.responses import StreamingResponse

from ..application.ports.event_bus import EventBus
from ..domain.events import DomainEvent
from ..json_safe import sanitize, strict_dumps

logger = logging.getLogger(__name__)

QUEUE_MAX = 256
HEARTBEAT_SECONDS = 15.0

# Delivery classes, keyed by event type. Anything absent is lifecycle,
# which is the safe default: it means "never drop for someone else's
# sake", not "coalescible because it is probably redundant".
STATE_EVENT_TYPES = frozenset({"run_progressed"})
DELTA_EVENT_TYPES = frozenset({"graph_execution_progressed"})


def delivery_class(event_type: str) -> str:
    """``"state"``, ``"delta"`` or ``"lifecycle"`` for one event type."""
    if event_type in DELTA_EVENT_TYPES:
        return "delta"
    if event_type in STATE_EVENT_TYPES:
        return "state"
    return "lifecycle"


def coalesce_key(event_type: str, payload: str) -> str | None:
    """The identity whose *newest* value supersedes an older one, or
    ``None`` if this kind must never be coalesced.

    Only state events get a key, and only the one field that identifies
    the thing being reported on: two samples for the same run supersede
    each other, two samples for different runs do not. A delta event
    returns ``None`` unconditionally -- the caller must never treat
    "has a key" as permission to evict, or N-04 comes straight back.
    """
    if delivery_class(event_type) != "state":
        return None
    try:
        import json

        run_id = json.loads(payload).get("run_id")
    except (ValueError, AttributeError):
        return None  # unparseable: treat as unkeyed, never coalesce
    return None if run_id is None else f"run:{run_id}"


def serialize_event(event: DomainEvent) -> str:
    """JSON payload for one event: ``type``, ``occurred_at``, fields.

    Always strict JSON: a non-finite float is ``null`` with its path
    named in ``nonfinite`` (docs 07 F-03).
    """
    payload: dict = {
        "type": event.event_type,
        "occurred_at": event.occurred_at.isoformat(),
    }
    for key, value in asdict(event).items():
        if key != "occurred_at":
            payload[key] = value
    return strict_dumps(sanitize(payload))


class ClientBuffer:
    """Bounded per-client frame buffer, resolved by delivery class.

    Both ``put`` and ``get`` run on the event loop (``put`` is hopped
    onto it from the publisher's thread), so the deque needs no lock.
    """

    def __init__(self, maxsize: int = QUEUE_MAX) -> None:
        self._items: deque[tuple[str, str, str | None]] = deque()
        self._maxsize = maxsize
        self._wake = asyncio.Event()
        self.coalesced = 0     # state frames superseded by a newer one
        self.dropped_delta = 0  # delta frames lost to a full buffer
        self.dropped = 0        # anything else lost to a full buffer

    def __len__(self) -> int:
        return len(self._items)

    def put(self, event_type: str, payload: str, key: str | None = None) -> None:
        """Append one frame, making room first if the buffer is full.

        `key` is the coalescing identity, computed by the caller (the
        event object is available there; the serialized payload is what
        this method receives). Computed here as a fallback when omitted,
        so a direct caller cannot accidentally get the old behaviour by
        forgetting it.

        A caller-supplied key is *not* trusted for the decision: it is
        re-checked against `coalesce_key` for this event type, so a
        delta can never be coalesced no matter what its caller passed.
        N-04 was exactly a caller treating "progress" as "redundant",
        and the failure mode of getting that wrong is silent event loss,
        so the check belongs inside the buffer rather than in every call
        site that might forget it.
        """
        expected = coalesce_key(event_type, payload)
        if expected is None:
            key = None  # not coalescible: a passed key is ignored
        elif key is None:
            key = expected
        if key is not None:
            # State only: replace the queued frame for THIS key, and
            # nothing else. A run's newest sample never costs another
            # run's sample, and never costs a node event.
            self.coalesced += self._drop_key(key)
        if len(self._items) >= self._maxsize:
            self._make_room(event_type)
        self._items.append((event_type, payload, key))
        self._wake.set()

    def _drop_key(self, key: str) -> int:
        indexes = [
            index
            for index, (_kind, _payload, queued_key) in enumerate(self._items)
            if queued_key == key
        ]
        for index in reversed(indexes):
            del self._items[index]
        return len(indexes)

    def _make_room(self, incoming: str) -> None:
        """Overflow policy: sacrifice the least valuable queued frame.

        Order matters, and it is the delivery classes from the module
        docstring: supersedeable state first (it is by definition
        reconstructible from a newer value of the same key), then the
        oldest delta (a fact that happened, lost, counted), then the
        oldest frame of any kind -- which at that point can only be
        lifecycle, since state and delta have both been tried.
        """
        for index, (kind, _payload, _key) in enumerate(self._items):
            if delivery_class(kind) == "state":
                del self._items[index]
                self.coalesced += 1
                logger.warning(
                    "SSE buffer full -- dropped a superseded-able %s to make "
                    "room for %s", kind, incoming,
                )
                return

        for index, (kind, _payload, _key) in enumerate(self._items):
            if delivery_class(kind) == "delta":
                del self._items[index]
                self.dropped_delta += 1
                logger.warning(
                    "SSE buffer full for a slow client -- dropped the oldest "
                    "delta %s to make room for %s (counted in dropped_delta; "
                    "the client will refetch node state on reconnect)",
                    kind, incoming,
                )
                return

        dropped_type, _payload, _key = self._items.popleft()
        self.dropped += 1
        logger.warning(
            "SSE buffer full of lifecycle events for a slow client -- "
            "dropped the oldest %s to make room for %s", dropped_type, incoming,
        )

    async def get(self, timeout: float) -> str | None:
        """Next payload, or ``None`` when nothing arrives within timeout."""
        while not self._items:
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout)
            except TimeoutError:
                return None
        return self._items.popleft()[1]


async def event_stream(bus: EventBus, request: Request) -> StreamingResponse:
    """Subscribe before returning so no event can be missed by the client."""
    loop = asyncio.get_running_loop()
    buffer = ClientBuffer()

    def on_event(event: DomainEvent) -> None:
        payload = serialize_event(event)
        loop.call_soon_threadsafe(buffer.put, event.event_type, payload)

    subscription = bus.subscribe(on_event)

    async def generate():
        try:
            opened = sanitize({
                "type": "stream_opened",
                "occurred_at": datetime.now(UTC).isoformat(),
            })
            yield f"data: {strict_dumps(opened)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                item = await buffer.get(HEARTBEAT_SECONDS)
                if item is None:
                    yield ": ping\n\n"
                    continue
                yield f"data: {item}\n\n"
        finally:
            subscription.close()
        logger.info(
            "SSE client gone -- coalesced=%d dropped_delta=%d dropped=%d",
            buffer.coalesced, buffer.dropped_delta, buffer.dropped,
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )