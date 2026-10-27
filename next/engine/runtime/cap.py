"""The line the allocator is held under: the card's limit minus what lives outside the allocator minus one margin."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

from ..devices import Accelerator
from .card import outside_or_unknown
from .policy import MemoryBudgetExceeded, MemoryPolicy


class AllocatorCap:
    """Settles the limit, then keeps the allocator's cap following what the driver holds outside it.

    Every number here is the driver's view of the whole card, other programs included: that is what runs out. Past the
    cap the allocator frees its cached blocks instead of letting them crowd out the driver, whose allocations (a new
    shape's kernels) it cannot satisfy by releasing anything.
    """

    CHANGE_MB = 128.0  # a new cap closer than this to the current one is not worth the driver call

    def __init__(self, accelerator: Accelerator, policy: MemoryPolicy, note: Callable[..., None]) -> None:
        self.accelerator, self.policy, self._note = accelerator, policy, note
        self.limit_mb: float | None = None
        self.cap_mb: float | None = None
        self._assumed_outside_mb = 0.0  # what the cap assumes the driver holds outside the allocator, margin included

    @property
    def ceiling_mb(self) -> float | None:
        return None if self.limit_mb is None else self.limit_mb - self.policy.margin_mb

    def settle(self) -> None:
        """Take the limit and set the first cap. Refuses early when the card cannot give what was asked for."""
        stats = self.accelerator.memory()
        if "device_total" not in stats:
            self.limit_mb = self.policy.limit_mb
            return
        total, free = stats["device_total"], stats["device_total"] - stats["device_used"]
        if self.policy.floor_mb and free < self.policy.floor_mb:
            raise MemoryBudgetExceeded(f"only {free:.0f} MB are free on the card; the floor is {self.policy.floor_mb:.0f} MB")
        if self.policy.limit_mb is not None and self.policy.limit_mb > total:
            raise MemoryBudgetExceeded(f"the budget is {self.policy.limit_mb:.0f} MB but the card has {total:.0f} MB")
        self.limit_mb = self.policy.limit_mb if self.policy.limit_mb is not None else total
        self._note("limit", limit_mb=round(self.limit_mb), in_use_mb=round(stats["device_used"]), card_mb=round(total),
                   margin_mb=round(self.policy.margin_mb),
                   allocator_config=os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_XPU_ALLOC_CONF") or "default")
        self._set(max(stats["device_used"] - stats.get("reserved", 0.0), 0.0), "start", stats)

    def follow(self, stats: Mapping[str, float], steps_seen: int) -> None:
        """The outside share grows as new shapes bring kernels, and the cap follows it down; once calibrated it is set
        from what was measured."""
        if "device_used" not in stats or "reserved" not in stats or self.cap_mb is None:
            return
        outside = max(stats["device_used"] - stats["reserved"], 0.0)
        calibrated = steps_seen == self.policy.calibration_steps
        if outside + self.policy.margin_mb > self._assumed_outside_mb or calibrated:
            self._set(outside, "calibrated" if calibrated else "outside memory grew", stats)

    def retreat(self, stats: Mapping[str, float]) -> None:
        """After an out-of-memory error: the assumption already holds one margin; this adds another."""
        if self.cap_mb is not None:
            self._set(self._assumed_outside_mb, "after out of memory", stats)

    def fits(self, stats: Mapping[str, float], mb: float, hard: bool) -> bool:
        """Whether ``mb`` more fits on the card and under the cap; ``hard`` allows the margin to be used."""
        line = self.limit_mb if hard else self.ceiling_mb
        if "device_used" in stats and line is not None and stats["device_used"] + mb > line:
            return False
        line = None if self.cap_mb is None else self.cap_mb + (self.policy.margin_mb if hard else 0.0)
        return not ("allocated" in stats and line is not None and stats["allocated"] + mb > line)

    def _set(self, outside_mb: float, why: str, stats: Mapping[str, float]) -> None:
        margin_mb = self.policy.margin_mb
        if self.limit_mb is None:
            return
        assumed = outside_mb + margin_mb
        cap = max(self.limit_mb - assumed, stats.get("allocated", 0.0) + margin_mb)
        if self.cap_mb is not None and abs(cap - self.cap_mb) < self.CHANGE_MB:
            return
        if self.accelerator.cap_allocator(cap):
            self._assumed_outside_mb, self.cap_mb = assumed, cap
            own, other = outside_or_unknown(stats)
            self._note("allocator_cap", why=why, cap_mb=round(cap), limit_mb=round(self.limit_mb), outside_mb=round(outside_mb),
                       ours_outside_mb=None if own is None else round(own), other_programs_mb=round(other), margin_mb=round(margin_mb))
