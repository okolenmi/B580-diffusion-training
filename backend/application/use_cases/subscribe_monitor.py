"""SubscribeMonitor -- open a monitor_id's telemetry stream.

Presentation used to call ``services.monitor_bus.subscribe(...)``
directly, which made the application aggregate a partially-open service
locator: exactly the two members it claims to encapsulate were the two
handlers used directly (docs 08 S-04). This use case is the front door.

The frames are still the bus's pre-rendered SSE strings -- that is
``MonitorBus``'s documented shape and a separate port-design question
(docs 08 S-19, deferred). What changes here is *who decides to open a
stream*: an application service, with a name and a subscription handle,
instead of a route handler reaching into infrastructure.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ..ports.monitor_bus import MonitorBus


class SubscribeMonitor:
    def __init__(self, *, bus: MonitorBus) -> None:
        self._bus = bus

    @asynccontextmanager
    async def open(self, monitor_id: str) -> AsyncIterator[AsyncIterator[str]]:
        """Subscribe for the duration of the block; always unsubscribe.

        Yields an async iterator of frames, so a caller streams without
        ever holding the queue or remembering the unsubscribe.
        """
        queue = self._bus.subscribe(monitor_id)
        try:
            yield self._frames(queue)
        finally:
            self._bus.unsubscribe(monitor_id, queue)

    @staticmethod
    async def _frames(queue: "asyncio.Queue[str]") -> AsyncIterator[str]:
        while True:
            yield await queue.get()