"""Thread-safe pub-sub for MonitorNode live data.

Same core pattern as server/sse.py's SSEManager (asyncio.Queue per
subscriber, broadcasts marshaled onto the event loop via
call_soon_threadsafe since graph execution runs in a FastAPI worker
thread, not the event loop) -- not reused directly because the domain is
different enough to not share a class cleanly: string monitor_id keys
instead of int run_id, a generic payload dict instead of a fixed
progress/status shape, and a history buffer so a dashboard opened after a
run has already started still shows recent data instead of just future
events.

Deliberately NOT a module-level singleton the way SSEManager's `sse =
SSEManager()` is: no instance is created here. server/main.py's lifespan
creates exactly one and attaches it to app.state; server/routes_monitor.py
and GraphExecutor both receive it as an explicit constructor/dependency
parameter. Same reason this lives at the top level, not under server/:
nodes/ can accept a MonitorBus instance without importing server/, and
tests can hand any code here a fresh, isolated instance instead of
sharing process-wide state through an import.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque

HISTORY_LIMIT = 500


class MonitorBus:

    def __init__(self):
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LIMIT))
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None

    def subscribe(self, monitor_id: str) -> asyncio.Queue:
        """Call only from within a running event loop (an async route
        handler) -- captures the loop on first use, same as SSEManager."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        for item in self._history[monitor_id]:
            q.put_nowait(f"data: {json.dumps(item)}\n\n")
        self._subscribers[monitor_id].append(q)
        return q

    def unsubscribe(self, monitor_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(monitor_id)
        if subs and q in subs:
            subs.remove(q)

    def report(self, monitor_id: str, data: dict) -> None:
        """Safe to call from any thread -- this is the side graph
        execution actually calls, from a FastAPI worker thread."""
        self._history[monitor_id].append(data)
        if self._loop is None:
            return  # nothing subscribed yet; history above still keeps it for later
        payload = f"data: {json.dumps(data)}\n\n"
        for q in list(self._subscribers.get(monitor_id, [])):
            self._loop.call_soon_threadsafe(self._safe_put, q, payload)

    @staticmethod
    def _safe_put(q: asyncio.Queue, data: str) -> None:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass
