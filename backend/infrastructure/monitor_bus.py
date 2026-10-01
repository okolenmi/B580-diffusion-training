"""SharedMonitorBus -- the port over the repo-root ``monitor_bus``.

Deliberate import bridge (same category as ``nodes`` in
``infrastructure/graph/runtime``): one bus instance per server process
feeds both the graph runtime (through ``ExecutionContext``) and the
SSE endpoint -- which is exactly what makes "open the dashboard
mid-run and see the whole history" work.

Wrapping, not copying: the repo-root class's history cap, thread
marshalling and frame format are pinned by
``smoke_test_monitor_bus.py`` and the legacy dashboard, and ``nodes/``
duck-types ``report``/``clear`` on whatever object the context carries,
so the adapter's methods are the entire contract.

One thing the adapter does add: reports are sanitized here
(:mod:`backend.json_safe`) before they reach the bus, because the
repo-root formatter uses plain ``json.dumps`` -- a diverged loss would
otherwise be framed as a bare ``NaN`` that every browser refuses to
parse, and the monitor page would silently lose the frame (docs 07
F-03).
"""

from __future__ import annotations

import asyncio

from ..application.ports.monitor_bus import MonitorBus as MonitorBusPort
from ..json_safe import sanitize
from monitor_bus import MonitorBus as ProcessMonitorBus


class SharedMonitorBus(MonitorBusPort):
    """Delegates to one repo-root ``MonitorBus`` per process."""

    def __init__(self) -> None:
        self._bus = ProcessMonitorBus()

    def report(self, monitor_id: str, data: dict) -> None:
        self._bus.report(monitor_id, sanitize(data))

    def clear(self, monitor_id: str) -> None:
        self._bus.clear(monitor_id)

    def subscribe(self, monitor_id: str) -> asyncio.Queue:
        return self._bus.subscribe(monitor_id)

    def unsubscribe(self, monitor_id: str, queue: asyncio.Queue) -> None:
        self._bus.unsubscribe(monitor_id, queue)
