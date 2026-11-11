"""PlannedRunsPump -- asks every few seconds whether a planned run can start now.

Memory comes back when a run ends, but also when another program lets go of the card, so the check is a timer and not
an event: one indexed query when nothing is planned."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class PlannedRunsPump:
    def __init__(self, promote: Callable[[], int], *, interval_s: float = 3.0) -> None:
        self._promote = promote
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="planned-runs", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def tick(self) -> int:
        """One check, for the loop and for tests; a failure is logged and the next tick tries again."""
        try:
            return self._promote()
        except Exception:  # noqa: BLE001 -- the pump must outlive one bad row
            logger.exception("starting planned runs failed")
            return 0

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.tick()
