"""DeviceProbe: the card readings the child and the admission check rely on (MEM-05).

The probe wraps the engine's ``Accelerator``; here a scripted torch-like module stands in, so no torch or card is
needed. The point: every reading the backend cannot get is ``None``/``False`` -- never a zero that reads as "full".

Run directly: python backend/tests/test_device_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.infrastructure.device_probe import DeviceProbe
from backend.tests.support import check, finish

MB = 1024 ** 2


class _Module:
    """What ``torch.xpu`` offers, as far as the probe asks."""

    def __init__(self, **have):
        self.have = have

    def is_available(self):
        return True

    def current_device(self):
        return 0

    def get_device_properties(self, _index):
        return type("Props", (), {"total_memory": 12000 * MB})()

    def mem_get_info(self):
        if "free_fails" in self.have:
            raise RuntimeError("driver said no")
        return 9000 * MB, 12000 * MB

    def set_per_process_memory_fraction(self, fraction):
        self.have["fraction"] = fraction


class _Accelerator:
    def __init__(self, module, numbers=None):
        self.module = module
        self._numbers = numbers or {}

    def memory(self):
        return self._numbers

    def empty_cache(self):
        self.emptied = True


def test_readings_in_megabytes() -> None:
    probe = DeviceProbe(_Accelerator(_Module()))
    check(probe.total_memory_mb() == 12000.0 and probe.free_memory_mb() == 9000.0, "total and free are read in MB")
    check(probe.set_per_process_memory_fraction(0.5), "the allocator cap is applied")


def test_unknown_is_not_zero() -> None:
    cpu = DeviceProbe(_Accelerator(None))
    check(cpu.total_memory_mb() is None and cpu.free_memory_mb() is None, "no backend module: unknown")
    check(cpu.memory_stats() is None and not cpu.set_per_process_memory_fraction(0.5), "and no stats, no cap")
    broken = DeviceProbe(_Accelerator(_Module(free_fails=True)))
    check(broken.free_memory_mb() is None, "a failing driver query is unknown, not a full card")


def test_stats_use_the_probe_names() -> None:
    probe = DeviceProbe(_Accelerator(_Module(), {"allocated": 4000.0, "reserved": 5000.0, "peak_allocated": 4500.0}))
    stats = probe.memory_stats()
    check(stats == {"allocated_mb": 4000.0, "reserved_mb": 5000.0, "peak_allocated_mb": 4500.0, "peak_reserved_mb": 5000.0},
          f"stats carry the *_mb names the telemetry reads, peak falling back to the current value (got {stats})")


def main() -> None:
    for test in (test_readings_in_megabytes, test_unknown_is_not_zero, test_stats_use_the_probe_names):
        test()
    finish()


if __name__ == "__main__":
    main()
