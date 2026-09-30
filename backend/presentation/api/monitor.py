"""Monitor stream endpoint -- SSE of live monitor-node telemetry.

Frame contract mirrors the legacy ``/api/nodegraph/monitor/{id}/stream``
byte for byte (pinned in ``03-migration-strategy.md`` section 4): a
``{"type": "connected"}`` opener, then the bus's pre-rendered frames
(history replay first, then live reports, plus ``{"type": "clear"}``
broadcasts). The page reads ``monitor_id`` from its own URL; nothing
server-side keys on it beyond the bus lookup.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ...application.services import ApplicationServices
from ..deps import get_services

router = APIRouter(tags=["monitor"])

# Same header set as the legacy endpoint and the events stream: tell
# every intermediary not to buffer, because a mid-run dashboard opened
# behind a stale buffer sees nothing until it expires.
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.get("/api/v1/monitor/{monitor_id}/stream")
async def monitor_stream(
    monitor_id: str,
    request: Request,
    services: ApplicationServices = Depends(get_services),
) -> StreamingResponse:
    """``text/event-stream`` of one ``monitor_id``'s raw step reports.

    ``subscribe`` replays history before returning, so the first frames
    after the opener restore a chart that has been running all along.
    """

    async def frames():
        queue = services.monitor_bus.subscribe(monitor_id)
        try:
            yield 'data: {"type": "connected"}\n\n'
            while True:
                # Blocks until a frame or cancellation; a client
                # disconnect cancels this task, which runs the finally.
                yield await queue.get()
        finally:
            services.monitor_bus.unsubscribe(monitor_id, queue)

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
