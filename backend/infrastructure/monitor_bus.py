"""SharedMonitorBus -- live monitor telemetry: one bus per server process, shared by the graph runtime and the SSE endpoint.

The runtime hands it to MonitorNode through ``ExecutionContext`` (``nodes/`` duck-types ``report`` and ``clear``), and the
SSE route subscribes to the same instance, which is what lets a dashboard opened mid-run get the whole history back.
No module-level instance exists: the composition root creates one, and tests hand code a fresh one.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque

from ..application.ports.monitor_bus import MonitorBus as MonitorBusPort
from ..json_safe import sanitize

#: What a (re)connecting dashboard gets replayed. It equals the dashboard's own record cap on purpose: a page reload
#: must restore exactly what the chart would have kept, and a smaller replay silently truncates the chart and the
#: elapsed readout (which spans the oldest report seen). Cost is about 2 KB per report; clear() drops it all when the
#: next run starts.
HISTORY_LIMIT = 100000

#: Per-subscriber backlog beyond the replay. A dashboard that cannot keep up (a backgrounded tab) must not hold every
#: report in RAM; the newest step report is what a chart needs, so a full queue gives up its oldest step report and
#: keeps the frames that change state (clear, run_end).
QUEUE_MAX = 512


class SharedMonitorBus(MonitorBusPort):
    def __init__(self) -> None:
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LIMIT))
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None
        #: Exception types already warned about, so a stream broken in a loop says it once, not once per frame.
        self._warned: set[type] = set()

    def _encode(self, item: dict) -> str | None:
        """``data: {...}\\n\\n`` for one item, or None if it cannot be serialized.

        Both callers run on somebody else's thread (report: a training step on a worker thread; subscribe: the event
        loop), so an unserializable value must not raise into them. The frame is dropped rather than replaced:
        a substitute would be a number nobody measured.
        """
        try:
            return f"data: {json.dumps(item)}\n\n"
        except (TypeError, ValueError) as exc:
            if type(exc) not in self._warned:
                self._warned.add(type(exc))
                print(
                    f"monitor bus: dropping an unserializable frame ({type(exc).__name__}: {exc}); "
                    f"further frames of this kind will be dropped silently"
                )
            return None

    def subscribe(self, monitor_id: str) -> asyncio.Queue:
        """Only from a running event loop (an async route handler): the loop is captured on first use."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        # The queue is sized QUEUE_MAX *above* the replay: the replay is the whole stored history and must fit, or
        # subscribing raises QueueFull on every reconnect after a long run, while a client that stalls afterwards is
        # still bounded.
        replay = [p for p in map(self._encode, tuple(self._history[monitor_id])) if p is not None]
        queue: asyncio.Queue = asyncio.Queue(maxsize=len(replay) + QUEUE_MAX)
        for payload in replay:
            queue.put_nowait(payload)
        self._subscribers[monitor_id].append(queue)
        return queue

    def unsubscribe(self, monitor_id: str, queue: asyncio.Queue) -> None:
        subscribers = self._subscribers.get(monitor_id)
        if subscribers and queue in subscribers:
            subscribers.remove(queue)

    def clear(self, monitor_id: str) -> None:
        """Drop this id's history and tell every connected dashboard to reset its chart.

        Called when a new run starts reporting to the id. A monitor id outlives any one run, so without this the old
        run's history is replayed ahead of the new run's and both land on the same step axis.
        """
        self._history[monitor_id].clear()
        if self._loop is None:
            return
        payload = f"data: {json.dumps({'type': 'clear'})}\n\n"
        for queue in list(self._subscribers.get(monitor_id, [])):
            self._loop.call_soon_threadsafe(self._safe_put, queue, payload)

    def report(self, monitor_id: str, data: dict) -> None:
        """Safe from any thread; cannot raise. A frame that will not serialize is dropped and not stored.

        Sanitized first: a diverged loss would otherwise be framed as a bare ``NaN`` that every browser refuses to
        parse, and the page would silently lose the frame.
        """
        data = sanitize(data)
        payload = self._encode(data)
        if payload is None:
            return
        self._history[monitor_id].append(data)
        if self._loop is None:
            return  # nothing subscribed yet; the history keeps it for later
        for queue in list(self._subscribers.get(monitor_id, [])):
            self._loop.call_soon_threadsafe(self._safe_put, queue, payload)

    @staticmethod
    def _safe_put(queue: asyncio.Queue, data: str) -> None:
        """Never block the publisher; on a full backlog drop the oldest *step report*, never a state-changing frame."""
        try:
            queue.put_nowait(data)
            return
        except asyncio.QueueFull:
            pass
        try:
            oldest = queue.get_nowait()
        except asyncio.QueueEmpty:  # raced with the consumer
            return
        if '"type"' in oldest and '"step"' not in oldest:
            queue.put_nowait(oldest)  # a state change: keep it and give up on this frame instead
            return
        try:
            queue.put_nowait(data)
        except asyncio.QueueFull:
            pass
