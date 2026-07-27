"""Runtime contract for anything a MonitorNode-family node can report to."""

from __future__ import annotations

from abc import ABC, abstractmethod


class MonitorHandle(ABC):

    @abstractmethod
    def report(self, data: dict) -> None:
        """Publish a snapshot (e.g. {"step": 10, "loss": 0.4}). Safe to
        call from any thread -- callers include TrainerNode's step loop,
        which runs in a FastAPI worker thread, not the event loop."""


class LiveMonitorHandle(MonitorHandle):
    """Reports into a MonitorBus instance under a client-chosen ID.

    Takes the bus as a constructor argument rather than importing a
    module-level instance -- see monitor_bus.py's docstring. The bus
    itself comes from the Node's injected ExecutionContext (see
    nodes/core.py and nodes/monitor/training_progress.py), not a global.
    """

    def __init__(self, monitor_id: str, bus):
        self.monitor_id = monitor_id
        self._bus = bus

    def report(self, data: dict) -> None:
        if self._bus is not None:
            self._bus.report(self.monitor_id, data)
