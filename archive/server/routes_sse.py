"""Server-Sent Events routes."""

import asyncio
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from .sse import sse

router = APIRouter()


@router.get("/sse")
async def sse_stream(run_id: int = Query(0)):
    """Server-Sent Events endpoint."""
    q = sse.subscribe(run_id)

    async def event_generator():
        # Send initial connection confirmation
        yield "data: {\"type\": \"connected\"}\n\n"
        try:
            while True:
                data = await q.get()
                yield data
        except asyncio.CancelledError:
            sse.unsubscribe(run_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
