"""The child's own host-memory ceiling (``memory.ram_max_mb``).

Cooperative, like every stop of a run: when the process holds more than its ceiling the guard asks the run to stop (the same
event a Stop request sets), and the run ends at its next safe point with the reason as its error. A stop cannot free memory a
node is in the middle of allocating, so the ceiling is a limit on how long the run may stay over, not a hard cap.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from .host_memory import process_held_mb

logger = logging.getLogger(__name__)


class HostRamGuard:
    def __init__(self, ceiling_mb: float, cancel: threading.Event, read: Callable[[], float | None] = process_held_mb) -> None:
        if ceiling_mb <= 0:
            raise ValueError(f"ceiling_mb must be above zero, got {ceiling_mb!r}")
        self._ceiling_mb = float(ceiling_mb)
        self._cancel = cancel
        self._read = read
        self._breach: str | None = None
        self._unreadable_logged = False

    @property
    def breach(self) -> str | None:
        """Why the run was stopped, or None while it never went over."""
        return self._breach

    def check(self) -> None:
        """One reading; asks the run to stop the first time it is over. An unreadable process is not over (unknown is not a number)."""
        if self._breach is not None:
            return
        held = self._read()
        if held is None:
            if not self._unreadable_logged:
                self._unreadable_logged = True
                logger.warning("host memory of this process cannot be read here; memory.ram_max_mb is not enforced")
            return
        if held > self._ceiling_mb:
            self._breach = (
                f"host memory: the run holds {held:.0f} MB, over its ceiling of {self._ceiling_mb:.0f} MB "
                f"(memory.ram_max_mb); it was asked to stop and ended at its next safe point"
            )
            logger.warning(self._breach)
            self._cancel.set()
