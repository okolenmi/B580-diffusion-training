"""What a run may put on the device, what is there now, and what moves off when it gets tight."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ...graph.settings import AUTO
from ..devices import Accelerator
from .residents import DeviceResident

MB = 2**20


class MemoryBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class MemoryPolicy:
    """The ``memory.*`` and ``runtime.*`` settings. ``limit_mb`` None means: what the card has left for this process."""

    limit_mb: float | None = None
    floor_mb: float = 0.0
    strict: bool = True
    safety_margin: float = 0.1
    calibration_steps: int = 3
    empty_cache_every: int = 1

    @classmethod
    def from_settings(cls, settings: Any) -> "MemoryPolicy":
        limit = settings.get("memory.vram_max_mb")
        return cls(None if limit in (None, AUTO) else float(limit), float(settings.get("memory.vram_min_mb") or 0),
                   bool(settings.get("memory.strict")), float(settings.get("runtime.residency.safety_margin")),
                   int(settings.get("runtime.residency.calibration_steps")), int(settings.get("runtime.empty_cache_every_n_steps")))


@dataclass
class _Entry:
    resident: DeviceResident
    rank: int
    pinned: bool
    parked: bool = False  # offloaded after every step, reloaded before the next
    away: bool = False


@dataclass
class ResidencyManager:
    """Knows every resident, measures the card the way the driver sees it, and reacts between steps.

    Reaction is deliberately between steps: what is already allocated cannot be taken back mid-step, so the
    manager measures after a step and, when the card is tighter than the policy allows, applies the next lever
    for the following steps. Levers go from cheap to costly: empty the allocator cache every step, then park
    residents (offloaded after each step, reloaded before the next) from the lowest rank up.
    """

    accelerator: Accelerator
    policy: MemoryPolicy = field(default_factory=MemoryPolicy)
    report: Callable[[dict], None] | None = None

    def __post_init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self.limit_mb: float | None = None
        self.peak_used_mb = 0.0
        self.last_used_mb: float | None = None
        self.last_stats: dict[str, float] = {}
        self._pressure_noted = 0.0
        self.empty_cache_every = self.policy.empty_cache_every
        self.steps_seen = 0
        self.trace: list[dict] = []

    def register(self, name: str, resident: DeviceResident, *, rank: int = 0, pinned: bool = False) -> None:
        """Lower ranks move off the device first; pinned residents never move."""
        self._entries[name] = _Entry(resident, rank, pinned)

    def used_mb(self) -> float | None:
        """Device memory in use as the driver sees it (other processes and the allocator's cache included)."""
        stats = self.accelerator.memory()
        return stats.get("device_used")

    def start(self) -> None:
        """Settle the limit. Refuses early when the card cannot give what was asked for."""
        stats = self.accelerator.memory()
        if "device_total" not in stats:
            self.limit_mb = self.policy.limit_mb
            return
        free_for_us = stats["device_total"] - (stats["device_used"] - stats.get("reserved", 0.0))
        if self.policy.floor_mb and free_for_us < self.policy.floor_mb:
            raise MemoryBudgetExceeded(f"only {free_for_us:.0f} MB are free for this run; the floor is {self.policy.floor_mb:.0f} MB")
        if self.policy.limit_mb is not None and self.policy.strict and self.policy.limit_mb > free_for_us:
            raise MemoryBudgetExceeded(f"the budget is {self.policy.limit_mb:.0f} MB but the card can give {free_for_us:.0f} MB")
        self.limit_mb = self.policy.limit_mb if self.policy.limit_mb is not None else free_for_us
        self._note("limit", limit_mb=round(self.limit_mb))

    @property
    def ceiling_mb(self) -> float | None:
        return None if self.limit_mb is None else self.limit_mb * (1 - self.policy.safety_margin)

    def breakdown(self, stats: Mapping[str, float] | None = None) -> dict[str, float]:
        """Where the card's memory is, in MB, for the monitor: each resident, other live tensors, the step's activation
        peak, free blocks the allocator keeps, and what the driver holds outside the allocator."""
        stats = stats if stats is not None else self.accelerator.memory()
        out = dict(self.summary())
        if "allocated" not in stats:
            return out
        out["other live"] = max(stats["allocated"] - sum(out.values()), 0.0)
        out["activation peak"] = max(stats.get("peak_allocated", stats["allocated"]) - stats["allocated"], 0.0)
        out["allocator cache"] = max(stats.get("reserved", 0.0) - stats["allocated"], 0.0)
        if "device_used" in stats:
            out["outside allocator"] = max(stats["device_used"] - stats.get("reserved", 0.0), 0.0)
        return {name: round(value, 1) for name, value in out.items()}

    def summary(self) -> dict[str, float]:
        return {name: round(e.resident.footprint_bytes() / MB, 1) for name, e in self._entries.items()}

    def make_room(self, mb: float, why: str) -> list[str]:
        """Offload unpinned residents, lowest rank first, until ``mb`` more fits under the ceiling."""
        moved: list[str] = []
        for name, entry in sorted(self._entries.items(), key=lambda kv: kv[1].rank):
            used = self.used_mb()
            if used is None or self.ceiling_mb is None or used + mb <= self.ceiling_mb:
                break
            if entry.pinned or entry.away or entry.resident.footprint_bytes() == 0:
                continue
            entry.resident.offload()
            entry.away = True
            self.accelerator.empty_cache()
            moved.append(name)
        used = self.used_mb()
        if moved:
            self._note("evict", why=why, moved=moved)
        if used is not None and self.ceiling_mb is not None and used + mb > self.ceiling_mb:
            raise MemoryBudgetExceeded(f"{why}: {mb:.0f} MB do not fit ({used:.0f} MB in use, ceiling {self.ceiling_mb:.0f} MB; "
                                       f"residents {self.summary()})")
        return moved

    def restore(self, names: list[str]) -> None:
        for name in names:
            entry = self._entries[name]
            entry.resident.reload()
            entry.away = False

    def before_step(self, step: int) -> None:
        for entry in self._entries.values():
            if entry.parked and entry.away:
                entry.resident.reload()
                entry.away = False

    def after_step(self, step: int) -> None:
        self.steps_seen += 1
        stats = self.accelerator.memory()
        used = self.last_used_mb = stats.get("device_used")
        self.last_stats = stats
        if used is not None:
            if self.steps_seen == 1 or used > self.peak_used_mb * 1.01:
                self._note("breakdown", step=step + 1, used_mb=round(used), parts=self.breakdown(stats))
            self.peak_used_mb = max(self.peak_used_mb, used)
            self._react(step, used)
        moved = False
        for entry in self._entries.values():
            if entry.parked and not entry.away:
                entry.resident.offload()
                entry.away = moved = True
        self.accelerator.reset_peak()
        if moved or (self.empty_cache_every and self.steps_seen % self.empty_cache_every == 0):
            self.accelerator.empty_cache()

    def _react(self, step: int, used: float) -> None:
        if self.limit_mb is None:
            return
        if used > self.limit_mb and self.policy.strict:
            raise MemoryBudgetExceeded(f"step {step + 1}: {used:.0f} MB in use is over the {self.limit_mb:.0f} MB budget "
                                       f"(residents {self.summary()})")
        if used <= self.ceiling_mb or self.steps_seen > self.policy.calibration_steps:
            if used > self.ceiling_mb and used > self._pressure_noted:
                self._pressure_noted = used
                self._note("pressure", step=step + 1, used_mb=round(used), settled=True)
            return
        self._note("pressure", step=step + 1, used_mb=round(used), settled=False)
        self._escalate()

    def _escalate(self) -> None:
        if self.empty_cache_every != 1:
            self.empty_cache_every = 1
            self._note("lever", name="empty the allocator cache every step")
            return
        for name, entry in sorted(self._entries.items(), key=lambda kv: kv[1].rank):
            if not entry.pinned and not entry.parked and entry.resident.footprint_bytes() > 0:
                entry.parked = True
                self._note("lever", name=f"park {name} between steps")
                return
        self._note("lever", name="none left")

    def _note(self, event: str, **fields: Any) -> None:
        record: Mapping[str, Any] = {"event": event, **fields}
        self.trace.append(dict(record))
        if self.report is not None:
            self.report(dict(record))
