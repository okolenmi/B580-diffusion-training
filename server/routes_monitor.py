"""SSE endpoint for MonitorNode live data.

Mirrors routes_sse.py's pattern (subscribe -> async generator reading
the queue -> StreamingResponse), reading the MonitorBus from
request.app.state instead of a module-level `sse` instance -- see
monitor_bus.py's docstring for why that's the one thing deliberately
different from routes_sse.py's precedent here.
"""

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/nodegraph/monitor")


@router.get("/{monitor_id}/stream")
async def monitor_stream(monitor_id: str, request: Request):
    bus = request.app.state.monitor_bus
    q = bus.subscribe(monitor_id)

    async def event_generator():
        yield "data: {\"type\": \"connected\"}\n\n"
        try:
            while True:
                data = await q.get()
                yield data
        except asyncio.CancelledError:
            bus.unsubscribe(monitor_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
