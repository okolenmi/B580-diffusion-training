"""Server-Sent Events manager for live training progress."""

import asyncio
import json
from collections import defaultdict


class SSEManager:
    """Manages SSE connections and broadcasts events to subscribers.

    Each run_id has its own channel. Clients subscribe to a run_id and
    receive events as they are published.

    broadcast() is called from RunMonitor's background threading.Thread
    (not the asyncio event loop) while a training run is being polled --
    asyncio.Queue isn't documented as thread-safe, so put_nowait() must
    never be called on it directly from that thread. All broadcasts are
    marshaled onto the event loop via call_soon_threadsafe() instead.
    """

    def __init__(self):
        # run_id -> list of asyncio.Queue
        self._subscribers: dict[int, list[asyncio.Queue]] = defaultdict(list)
        # Global subscribers (run_id=0)
        self._global: list[asyncio.Queue] = []
        # Captured on first subscribe() (always called from within the
        # event loop, since it's only ever invoked from an async route
        # handler) -- there's exactly one loop for the whole app (single
        # uvicorn worker), so capturing it once is enough.
        self._loop: asyncio.AbstractEventLoop | None = None

    def subscribe(self, run_id: int = 0) -> asyncio.Queue:
        """Return a queue that will receive events for the given run_id."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        q: asyncio.Queue = asyncio.Queue()
        if run_id == 0:
            self._global.append(q)
        else:
            self._subscribers[run_id].append(q)
        return q

    def unsubscribe(self, run_id: int, q: asyncio.Queue):
        if run_id == 0:
            if q in self._global:
                self._global.remove(q)
        else:
            if q in self._subscribers.get(run_id, []):
                self._subscribers[run_id].remove(q)

    @staticmethod
    def _safe_put(q: asyncio.Queue, data: str):
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass

    def broadcast(self, run_id: int, event: dict):
        """Push an event to all subscribers of the given run_id and global.

        Safe to call from any thread. If no one has subscribed yet (no
        event loop captured), this is a no-op -- there are, by
        construction, no queues to put into in that case.
        """
        if self._loop is None:
            return

        data = f"data: {json.dumps(event)}\n\n"
        targets = list(self._subscribers.get(run_id, [])) + list(self._global)
        for q in targets:
            self._loop.call_soon_threadsafe(self._safe_put, q, data)

    def progress(self, run_id: int, step: int, total: int,
                 loss: float | None = None, avg_loss: float | None = None,
                 lr: float | None = None, status: str | None = None,
                 **kwargs):
        """Convenience: broadcast a progress update."""
        payload = {
            "type": "progress",
            "run_id": run_id,
            "step": step,
            "total": total,
            "loss": loss,
            "avg_loss": avg_loss,
            "lr": lr,
            "status": status,
        }
        payload.update(kwargs)
        self.broadcast(run_id, payload)

    def status(self, run_id: int, status: str, error: str | None = None):
        """Convenience: broadcast a status change."""
        self.broadcast(run_id, {
            "type": "status",
            "run_id": run_id,
            "status": status,
            "error": error,
        })


# Global instance
sse = SSEManager()
