"""SystemClock -- the real-time implementation of the Clock port."""

from __future__ import annotations

from datetime import datetime, UTC

from ..application.ports.clock import Clock


class SystemClock(Clock):
    """Wall-clock time in UTC."""

    def now(self) -> datetime:
        return datetime.now(UTC)
