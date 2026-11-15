"""A background log of device memory against what the run is doing, to find spikes the step reports cannot show."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from ..devices import Accelerator

COLUMNS = ("seconds", "label", "allocated_mb", "reserved_mb", "peak_allocated_mb", "device_used_mb", "process_mb", "active_mb", "split_free_mb", "segments", "alloc_retries")


class MemorySampler:
    """Writes one CSV row per ``interval`` seconds. ``label`` is whatever the caller says is happening now."""

    def __init__(self, accelerator: Accelerator, path: str | Path, interval: float = 0.25):
        self.accelerator, self.path, self.interval = accelerator, Path(path), interval
        self.label = "start"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="memory-sampler", daemon=True)

    def __enter__(self) -> MemorySampler:
        self._t0 = time.monotonic()
        self._file = self.path.open("w", encoding="utf8")
        self._file.write(",".join(COLUMNS) + "\n")
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._file.close()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            stats = self.accelerator.memory()
            row = (round(time.monotonic() - self._t0, 2), self.label.replace(",", ";"), stats.get("allocated", ""), stats.get("reserved", ""),
                   stats.get("peak_allocated", ""), stats.get("device_used", ""), stats.get("process", ""), stats.get("active", ""), stats.get("split_free", ""), stats.get("segments", ""),
                   stats.get("alloc_retries", ""))
            self._file.write(",".join(map(str, row)) + "\n")
            self._file.flush()
