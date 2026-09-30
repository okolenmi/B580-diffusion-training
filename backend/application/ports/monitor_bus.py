"""MonitorBus port -- live monitor-node telemetry, one stream per id.

The contract is pinned by ``docs/design/backend/03-migration-strategy.md``
section 4 and mirrors the repo-root ``monitor_bus.MonitorBus`` the
legacy server and ``nodes/`` already use, so payloads and replay
semantics stay byte-identical across the migration:

* ``report``/``clear`` are called from graph-executor *worker threads*
  (a trainer's step loop) -- implementations must be thread-safe and
  must never require the event loop;
* ``subscribe`` must be called from a running event loop and **replays
  the buffered history into the new queue first** -- a dashboard opened
  mid-run (or reloaded) restores its chart instead of starting empty;
* frames are pre-rendered SSE strings (``data: {json}\\n\\n``), not
  dicts: the history buffer *is* the wire format.

asyncio appears in this port deliberately: the consumer is an SSE
response, and the replay-before-first-frame ordering only exists at
that level. Pushing the queue anywhere else would just misplace it.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class MonitorBus(ABC):
    """Per-``monitor_id`` pub/sub with bounded history and replay."""

    @abstractmethod
    def report(self, monitor_id: str, data: dict) -> None:
        """Buffer ``data`` (unwrapped) and fan it out to subscribers."""
        raise NotImplementedError

    @abstractmethod
    def clear(self, monitor_id: str) -> None:
        """Drop ``monitor_id``'s history and broadcast a clear frame."""
        raise NotImplementedError

    @abstractmethod
    def subscribe(self, monitor_id: str) -> asyncio.Queue:
        """New subscriber queue, history replayed into it first.

        Only legal from a running event loop (it captures the loop to
        marshal cross-thread reports onto it).
        """
        raise NotImplementedError

    @abstractmethod
    def unsubscribe(self, monitor_id: str, queue: asyncio.Queue) -> None:
        """Detach ``queue``; idempotent for an already-detached one."""
        raise NotImplementedError
