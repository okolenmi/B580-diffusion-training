"""Time on the device: synchronised wall time and, where the backend has them, device events."""

from __future__ import annotations

import time
from typing import Any

import torch


class DeviceClock:
    """``stamp`` marks a point on the device's queue without waiting; ``elapsed_ms`` reads two stamps (after ``synchronize``)."""

    def __init__(self, device: str = "cpu"):
        kind = str(device).split(":")[0]
        self.kind = kind
        self.module = getattr(torch, kind, None) if kind != "cpu" else None
        self.has_events = self.module is not None and hasattr(self.module, "Event")

    def synchronize(self) -> None:
        if self.module is not None and hasattr(self.module, "synchronize"):
            self.module.synchronize()

    def capturing(self) -> bool:
        """True while the current stream records a device graph: events made then carry no timing, and the work in it does not run."""
        for owner in (self.module, getattr(self.module, "graphs", None)):
            probe = getattr(owner, "is_current_stream_capturing", None)
            if probe is not None:
                try:
                    return bool(probe())
                except RuntimeError:
                    return False
        return False

    def stamp(self) -> Any:
        if self.has_events:
            event = self.module.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def elapsed_ms(self, start: Any, end: Any) -> float:
        if self.has_events:
            return float(start.elapsed_time(end))
        return (end - start) * 1000.0

    def wall(self) -> float:
        return time.perf_counter()
