"""DeviceProbe -- what the graph child and the admission check ask the card.

A thin reading layer over the engine's ``Accelerator``: the worker needs a few device numbers (free, total, the
allocator's own counters) and one lever (the allocator cap), and nothing about how the engine moves memory around.
Built per run and only inside the child (or a run thread), because the engine's device module imports torch and the
Intel shader-cache variables must be set before that happens.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MB = 1024 ** 2


class DeviceProbe:
    """Every answer is ``None`` (or ``False``) when the backend cannot give it: unknown is never a zero."""

    def __init__(self, accelerator) -> None:
        self._accel = accelerator

    @classmethod
    def for_device(cls, name: str) -> "DeviceProbe":
        from next.engine.devices import Accelerator, resolve_device  # noqa: PLC0415 - imports torch

        return cls(Accelerator(resolve_device(name)))

    @property
    def _module(self):
        return self._accel.module

    def empty_cache(self) -> None:
        self._accel.empty_cache()

    def total_memory_mb(self) -> float | None:
        mod = self._module
        if mod is None or not mod.is_available():
            return None
        try:
            return mod.get_device_properties(mod.current_device()).total_memory / _MB
        except Exception:  # noqa: BLE001 - the caller's unknown branch
            return None

    def free_memory_mb(self) -> float | None:
        mod = self._module
        if mod is None or not mod.is_available() or not hasattr(mod, "mem_get_info"):
            return None
        try:
            return mod.mem_get_info()[0] / _MB
        except Exception as exc:  # noqa: BLE001 - say why once, rather than degrade into a check that "passed"
            logger.warning("mem_get_info could not be read: %s", exc)
            return None

    def memory_stats(self) -> dict[str, float] | None:
        """The allocator's current and peak numbers in MB; ``None`` on the CPU."""
        numbers = self._accel.memory()
        if "allocated" not in numbers:
            return None
        return {
            "allocated_mb": numbers["allocated"],
            "reserved_mb": numbers.get("reserved", 0.0),
            "peak_allocated_mb": numbers.get("peak_allocated", numbers["allocated"]),
            "peak_reserved_mb": numbers.get("peak_reserved", numbers.get("reserved", 0.0)),
        }

    def set_per_process_memory_fraction(self, fraction: float) -> bool:
        mod = self._module
        if mod is None or not hasattr(mod, "set_per_process_memory_fraction"):
            return False
        mod.set_per_process_memory_fraction(fraction)
        return True
