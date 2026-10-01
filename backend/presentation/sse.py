"""SSE bridge -- EventBus port -> text/event-stream response.

Threading contract: bus handlers run on the *publisher's* thread (any
thread -- use cases, the training monitor). The bridge therefore hops
onto the event loop with ``call_soon_threadsafe`` before touching the
client buffer.

Backpressure (docs 07 F-09): the buffer is bounded, and overflow is
resolved by *kind*, not by recency. ``run_progressed`` /
``graph_execution_progressed`` are transient -- only their newest
values matter -- so they are coalesced: enqueuing a new progress frame
evicts every older queued progress frame. Lifecycle frames
(``run_completed``, ``run_failed``, ...) are never dropped for the sake
of progress. If the buffer somehow fills with lifecycle frames alone,
the oldest goes, the drop is counted and logged -- unbounded growth
would trade lost events for unbounded memory, and the client's
refetch-on-(re)open plus the dashboard's slow poll close that gap.

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
from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import StreamingResponse

from ..application.ports.event_bus import EventBus
from ..domain.events import DomainEvent
from ..json_safe import sanitize, strict_dumps

logger = logging.getLogger(__name__)

QUEUE_MAX = 256
HEARTBEAT_SECONDS = 15.0

# Transient telemetry: coalescible, safe to lose to a newer value.
PROGRESS_EVENT_TYPES = frozenset({"run_progressed", "graph_execution_progressed"})


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
    """Bounded per-client frame buffer with kind-aware overflow.

    Both ``put`` and ``get`` run on the event loop (``put`` is hopped
    onto it from the publisher's thread), so the deque needs no lock.
    """

    def __init__(self, maxsize: int = QUEUE_MAX) -> None:
        self._items: deque[tuple[str, str]] = deque()
        self._maxsize = maxsize
        self._wake = asyncio.Event()
        self.coalesced = 0  # progress frames dropped to make room
        self.dropped = 0    # lifecycle frames dropped (logged, never silent)

    def __len__(self) -> int:
        return len(self._items)

    def put(self, event_type: str, payload: str) -> None:
        """Append one frame, making room first if the buffer is full.

        Progress is coalesced eagerly -- a queued progress frame is
        superseded by the one being enqueued, so a client that falls
        behind sees the newest values, not a backlog of stale ones.
        """
        if event_type in PROGRESS_EVENT_TYPES:
            self.coalesced += self._drop_progress()
        if len(self._items) >= self._maxsize:
            self._make_room(event_type)
        self._items.append((event_type, payload))
        self._wake.set()

    def _drop_progress(self) -> int:
        indexes = [
            index
            for index, (kind, _) in enumerate(self._items)
            if kind in PROGRESS_EVENT_TYPES
        ]
        for index in reversed(indexes):
            del self._items[index]
        return len(indexes)

    def _make_room(self, incoming: str) -> None:
        if self._drop_progress():
            return
        # Only lifecycle frames queued: an all-lifecycle backlog means
        # the client is far slower than the event rate. Drop the oldest
        # and say so (rule: no silent swallowing).
        dropped_type, _ = self._items.popleft()
        self.dropped += 1
        logger.warning(
            "SSE buffer full of lifecycle events for a slow client -- "
            "dropped the oldest %s to make room for %s",
            dropped_type, incoming,
        )

    async def get(self, timeout: float) -> str | None:
        """Next payload, or ``None`` when nothing arrives within timeout."""
        while not self._items:
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout)
            except asyncio.TimeoutError:
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
                "occurred_at": datetime.now(timezone.utc).isoformat(),
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

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )