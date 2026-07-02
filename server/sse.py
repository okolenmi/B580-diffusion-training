"""Server-Sent Events manager for live training progress."""

import asyncio
import json
from collections import defaultdict


class SSEManager:
    """Manages SSE connections and broadcasts events to subscribers.

    Each run_id has its own channel. Clients subscribe to a run_id and
    receive events as they are published.
    """

    def __init__(self):
        # run_id -> list of asyncio.Queue
        self._subscribers: dict[int, list[asyncio.Queue]] = defaultdict(list)
        # Global subscribers (run_id=0)
        self._global: list[asyncio.Queue] = []

    def subscribe(self, run_id: int = 0) -> asyncio.Queue:
        """Return a queue that will receive events for the given run_id."""
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

    def broadcast(self, run_id: int, event: dict):
        """Push an event to all subscribers of the given run_id and global."""
        data = f"data: {json.dumps(event)}\n\n"

        # Send to run-specific subscribers
        for q in list(self._subscribers.get(run_id, [])):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass

        # Send to global subscribers
        for q in list(self._global):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass

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
