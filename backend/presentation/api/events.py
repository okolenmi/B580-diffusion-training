"""Events endpoint -- live Server-Sent Events stream of domain events."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ...application.services import ApplicationServices
from ..deps import get_services
from ..sse import event_stream

router = APIRouter(tags=["events"])


@router.get("/api/v1/events")
async def stream_events(
    request: Request,
    services: ApplicationServices = Depends(get_services),
) -> StreamingResponse:
    """``text/event-stream``; each frame is ``data: {json}`` with a
    ``type`` field (``stream_opened``, ``run_started``, ...)."""
    return await event_stream(services.event_bus, request)
