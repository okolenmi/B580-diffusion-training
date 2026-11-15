"""Average GPU power over a window, from the energy counter the kernel driver exposes.

"Load" says a kernel is resident, power says how much of the chip it is using: stalled or launch-bound work draws far less than
saturated matrix work at the same load figure, which no time measurement shows. The counter is the hwmon ``energy*_input`` file (microjoules) of an Intel card
under ``/sys/class/drm``; where there is none the meter reports nothing.
"""

from __future__ import annotations

import time
from pathlib import Path
from collections.abc import Callable

INTEL_VENDOR = "0x8086"


class PowerMeter:
    def __init__(self, root: str | Path = "/sys/class/drm", clock: Callable[[], float] = time.perf_counter):
        self.counter = self._find(Path(root))
        self._clock = clock
        self._start: tuple[float, int] | None = None

    @staticmethod
    def _find(root: Path) -> Path | None:
        for card in sorted(root.glob("card[0-9]*")):
            vendor = card / "device" / "vendor"
            if vendor.exists() and vendor.read_text().strip() != INTEL_VENDOR:
                continue
            for counter in sorted(card.glob("device/hwmon/hwmon*/energy*_input")):
                return counter
        return None

    def _read(self) -> int | None:
        try:
            return int(self.counter.read_text())
        except (OSError, ValueError, AttributeError):
            return None

    def start(self) -> None:
        energy = self._read()
        self._start = None if energy is None else (self._clock(), energy)

    def stop(self) -> float | None:
        """Average watts since ``start``, or None when there is no counter or no time has passed."""
        energy, begun = self._read(), self._start
        self._start = None
        if energy is None or begun is None:
            return None
        seconds = self._clock() - begun[0]
        return (energy - begun[1]) / 1e6 / seconds if seconds > 0 and energy >= begun[1] else None
