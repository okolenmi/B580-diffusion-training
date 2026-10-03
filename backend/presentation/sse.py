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

from ..application.ports.event_bus import EventBus, EventCursor, Sequenced
from ..domain.events import DomainEvent
from ..application.event_delivery import delivery_class, is_coalescible
from ..application.limits import SSE_HEARTBEAT_SECONDS, SSE_QUEUE_MAX
from ..json_safe import sanitize, strict_dumps

logger = logging.getLogger(__name__)

# Both live in application/limits.py with every other budget. Aliased to
# the historical local names because they are read in several places
# below and renaming a constant inside one module is churn, not clarity.
QUEUE_MAX = SSE_QUEUE_MAX
HEARTBEAT_SECONDS = SSE_HEARTBEAT_SECONDS

def coalesce_key(event_type: str, payload: str) -> str | None:
    """The identity whose *newest* value supersedes an older one, or
    ``None`` if this kind must never be coalesced.

    Only state events get a key, and only the one field that identifies
    the thing being reported on: two samples for the same run supersede
    each other, two samples for different runs do not. A delta event
    returns ``None`` unconditionally -- the caller must never treat
    "has a key" as permission to evict, or N-04 comes straight back.

    ``is_coalescible`` is the predicate rather than a ``== "state"``
    re-derivation: "may this be coalesced" and "is this a state event"
    happen to coincide today because state is the only coalescible class,
    but they are different questions, and this one is the one being asked.
    """
    if not is_coalescible(event_type):
        return None
    try:
        import json

        run_id = json.loads(payload).get("run_id")
    except (ValueError, AttributeError):
        return None  # unparseable: treat as unkeyed, never coalesce
    return None if run_id is None else f"run:{run_id}"


def serialize_event(event: DomainEvent, seq: int | None = None) -> str:
    """JSON payload for one event: ``seq``, ``type``, ``occurred_at``, fields.

    Always strict JSON: a non-finite float is ``null`` with its path
    named in ``nonfinite`` (docs 07 F-03).

    ``seq`` is what a client sends back as ``Last-Event-ID``, so it is
    part of the payload rather than the SSE ``id:`` field -- one JSON
    frame carries everything, and a client that reads only ``data`` still
    has it. Omitting it is allowed for the synthetic frames this module
    itself sends (``stream_opened`` and friends), which are not bus
    events and have no place in the sequence.
    """
    payload: dict = {}
    if seq is not None:
        payload["seq"] = seq
    payload.update({
        "type": event.event_type,
        "occurred_at": event.occurred_at.isoformat(),
    })
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
        self._items: deque[tuple[str, str, str | None, int | None]] = deque()
        self._maxsize = maxsize
        self._wake = asyncio.Event()
        self.coalesced = 0     # state frames superseded by a newer one
        self.dropped_delta = 0  # delta frames lost to a full buffer
        self.dropped = 0        # anything else lost to a full buffer

    def __len__(self) -> int:
        return len(self._items)

    def put(self, event_type: str, payload: str, key: str | None = None,
            seq: int | None = None) -> None:
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
        self._items.append((event_type, payload, key, seq))
        self._wake.set()

    def _drop_key(self, key: str) -> int:
        indexes = [
            index
            for index, (_kind, _payload, queued_key, _seq) in enumerate(self._items)
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
        for index, (kind, _payload, _key, _seq) in enumerate(self._items):
            if delivery_class(kind) == "state":
                del self._items[index]
                self.coalesced += 1
                logger.warning(
                    "SSE buffer full -- dropped a superseded-able %s to make "
                    "room for %s", kind, incoming,
                )
                return

        for index, (kind, _payload, _key, _seq) in enumerate(self._items):
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

        dropped_type, _payload, _key, _seq = self._items.popleft()
        self.dropped += 1
        logger.warning(
            "SSE buffer full of lifecycle events for a slow client -- "
            "dropped the oldest %s to make room for %s", dropped_type, incoming,
        )

    async def get(self, timeout: float) -> tuple[int | None, str | None]:
        """Next ``(seq, payload)``, or ``(None, None)`` on timeout.

        The sequence number travels with the frame so the stream can drop
        anything the replay already delivered -- the client cannot see the
        difference otherwise, and a duplicate lifecycle event is a row
        applied twice.
        """
        while not self._items:
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout)
            except TimeoutError:
                return None, None
        _kind, payload, _key, seq = self._items.popleft()
        return seq, payload


def _last_event_id(request: Request) -> EventCursor | None:
    """The client's ``Last-Event-ID``, if it is one we can use.

    `EventSource` sends it automatically on every automatic reconnect, so
    this is not a client-side change -- it is what the browser already
    does, finally being read.

    What comes back is an ``EventCursor``: ``"{epoch}:{seq}"``, opaque to
    the client, which never parses it and cannot be asked to. A header that
    is absent, or is a bare number from a client that predates the epoch,
    is treated as *absent* -- the client then gets live events and
    ``resync_required``, which is the same place it would have started
    without this feature. That is not a graceful degradation to shrug at:
    it is the honest answer, because a bare number cannot say which
    process issued it.
    """
    raw = request.headers.get("last-event-id")
    if raw is None:
        return None
    cursor = EventCursor.parse(raw)
    if cursor is None:
        logger.info("ignoring unusable Last-Event-ID %r", raw)
        return None
    return cursor


def _frame(payload: str, seq: int | None, epoch: str | None) -> str:
    """One SSE frame: an ``id:`` line, then the ``data:`` line.

    The ``id:`` line is the load-bearing half. `EventSource` tracks the
    last id it saw and sends it back as ``Last-Event-ID`` **on automatic
    reconnection only**, so without it this whole feature would never
    fire for the browser that motivates it -- the client would have to
    track the sequence itself, which is the client-side change the design
    note says is not needed.

    The id is the epoch and the sequence together, ``"{epoch}:{seq}"``,
    because a bare sequence cannot be acted on: every process numbers its
    events from 1, so a number from a previous process is only
    distinguishable while it happens to be ahead of everything published
    here. Once the new process has published past it -- which is exactly
    what happens when a client reconnects late after a restart -- the
    number looks valid and the client is told its history is continuous
    when it is not. Pairing them removes the question.

    The bare sequence is *also* inside the JSON payload, which is
    redundant on purpose: it costs 12 bytes and it means a client that
    consumes `data` without touching the SSE framing (the tests, and any
    hand-written reader) still has the position it needs to reason about.
    Only the framed id is used for reconnection, and only the bus reads it.
    """
    if seq is None or epoch is None:
        return f"data: {payload}\n\n"
    return f"id: {epoch}:{seq}\ndata: {payload}\n\n"


async def event_stream(bus: EventBus, request: Request) -> StreamingResponse:
    """One client stream: optional replay, then live.

    **Subscribe first, then replay.** The opposite order has a hole: an
    event published between the replay and the subscription is lost
    forever, which is the bug this whole feature exists to fix. Doing it
    in this order means such an event is delivered *twice* instead, which
    the client can detect -- frames carry their `seq`, so anything the
    replay already covered is dropped on the way out.
    """
    loop = asyncio.get_running_loop()
    buffer = ClientBuffer()
    last_seen = _last_event_id(request)

    def on_event(sequenced: Sequenced) -> None:
        payload = serialize_event(sequenced.event, sequenced.seq)
        loop.call_soon_threadsafe(
            buffer.put, sequenced.event_type, payload, None, sequenced.seq
        )

    subscription = bus.subscribe(on_event)
    # Asked unconditionally, including for a first connect: the bus knows
    # that a client which sent nothing cannot be told its history is
    # complete, and saying so here as well meant two places to keep in
    # step. `last_seen is None` reaching it is a normal answer, not an
    # error case.
    replay = bus.replay_since(last_seen)
    # Nothing published while we were subscribing is above the replay's
    # reach, so this is the watermark the live path must skip back to.
    #
    # The fallback is the client's own cursor, but only when the cursor is
    # from *this* process: the watermark means "how far into this stream
    # the client has read", and a cursor from a previous process says
    # nothing about that. Defaulting it to that number would make the live
    # path discard the new stream's first events -- the client's old seq 40
    # would suppress everything up to 40 here, which it has never seen.
    known = (
        last_seen
        if last_seen is not None and last_seen.epoch == bus.epoch
        else None
    )
    replayed_through = max(
        (item.seq for item in replay.events),
        default=known.seq if known is not None else 0,
    )

    async def generate():
        try:
            opened = sanitize({
                "type": "stream_opened",
                "occurred_at": datetime.now(UTC).isoformat(),
                # "Refetch the source of truth." True when the client
                # sent no Last-Event-ID -- a first connect has missed
                # everything published before it subscribed, whatever the
                # ring holds -- and when the ring cannot cover what it did
                # miss. False only when a replay covered the whole gap.
                #
                # The flag is deliberately about lifecycle events alone:
                # a missed *delta* is not recoverable by refetching the
                # runs table, and pretending otherwise would make the
                # frontend refetch on every reconnect forever.
                "resync_required": last_seen is None or not replay.complete,
                # Where the client is once it has read this frame AND the
                # replay that follows it -- so it is the end of the replay,
                # not the id it sent. Named for that: "last_seq" was read
                # as the latter by the first test to use it.
                "replayed_through": replayed_through or None,
            })
            yield f"data: {strict_dumps(opened)}\n\n"
            for missed in replay.events:
                yield _frame(
                    serialize_event(missed.event, missed.seq), missed.seq,
                    bus.epoch,
                )
            while True:
                if await request.is_disconnected():
                    break
                seq, payload = await buffer.get(HEARTBEAT_SECONDS)
                if payload is None:
                    yield ": ping\n\n"
                    continue
                if seq is not None and seq <= replayed_through:
                    continue  # already delivered by the replay above
                yield _frame(payload, seq, bus.epoch)
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