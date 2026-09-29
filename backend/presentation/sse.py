"""SSE bridge -- EventBus port -> text/event-stream response.

Threading contract: bus handlers run on the *publisher's* thread (any
thread -- use cases, the training monitor). The bridge therefore hops
onto the event loop with ``call_soon_threadsafe`` before touching the
asyncio queue. Backpressure: the queue is bounded; if a consumer falls
behind, newest events are dropped (SSE is a live tail, not history --
replay belongs to the database).

Heartbeat every ``HEARTBEAT_SECONDS`` (``: ping`` comment) keeps
proxies from reaping an idle stream and lets the generator notice a
dead client promptly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import StreamingResponse

from ..application.ports.event_bus import EventBus
from ..domain.events import DomainEvent

logger = logging.getLogger(__name__)

QUEUE_MAX = 256
HEARTBEAT_SECONDS = 15.0


def serialize_event(event: DomainEvent) -> str:
    """JSON payload for one event: ``type``, ``occurred_at``, fields."""
    payload: dict = {
        "type": event.event_type,
        "occurred_at": event.occurred_at.isoformat(),
    }
    for key, value in asdict(event).items():
        if key != "occurred_at":
            payload[key] = value
    return json.dumps(payload)


async def event_stream(bus: EventBus, request: Request) -> StreamingResponse:
    """Subscribe before returning so no event can be missed by the client."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_MAX)

    def on_event(event: DomainEvent) -> None:
        payload = serialize_event(event)

        def enqueue() -> None:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                logger.debug("SSE queue full -- dropping %s", event.event_type)

        loop.call_soon_threadsafe(enqueue)

    subscription = bus.subscribe(on_event)

    async def generate():
        try:
            opened = {
                "type": "stream_opened",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
            }
            yield f"data: {json.dumps(opened)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
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
