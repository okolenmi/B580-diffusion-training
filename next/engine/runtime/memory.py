"""What a run may put on the device, what is there now, and what moves off when it gets tight."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ...graph.settings import AUTO
from ..devices import Accelerator
from .residents import DeviceResident

MB = 2**20


class MemoryBudgetExceeded(RuntimeError):
    pass


OOM_MARKERS = ("out of memory", "out_of_resources", "out_of_device_memory", "out of device memory")


def is_out_of_memory(error: BaseException) -> bool:
    """True for the allocator's error and for the driver's own refusals, which arrive as plain runtime errors."""
    return isinstance(error, MemoryError) or type(error).__name__ == "OutOfMemoryError" or \
        (isinstance(error, RuntimeError) and any(marker in str(error).lower() for marker in OOM_MARKERS))


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
    manager measures after a step and, when the card is tighter than the policy allows, parks one more resident for
    the following steps (offloaded after each step, reloaded before the next), smallest first because the size is
    the transfer cost. Parking is never undone, so the run does not flip between resident and parked. The
    allocator cache is emptied once something has been released, and also whenever the cache itself crowds the card
    past the ceiling (it costs a synchronise, so only then).
    """

    MIN_TRIM_MB = 256.0
    OUTSIDE_HEADROOM_MB = 1024.0  # first guess for what the driver still takes outside the allocator (kernels, primitives)
    OUTSIDE_SLACK_MB = 512.0      # kept free beyond what is measured outside the allocator

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
        self.empty_cache_every = self.policy.empty_cache_every  # applies once something is released
        self.steps_seen = 0
        self.trace: list[dict] = []
        self._outside_mb = 0.0  # what the allocator cap currently assumes the driver holds outside it
        self._cap_mb: float | None = None

    def register(self, name: str, resident: DeviceResident, *, rank: int = 0, pinned: bool = False) -> None:
        """Lower ranks move off the device first; pinned residents never move."""
        self._entries[name] = _Entry(resident, rank, pinned)

    def used_mb(self) -> float | None:
        """Device memory in use as the driver sees it (other processes and the allocator's cache included)."""
        stats = self.accelerator.memory()
        return stats.get("device_used")

    @staticmethod
    def need_mb(stats: Mapping[str, float]) -> float | None:
        """What the step needs that cannot be given back: its peak live tensors plus what the driver holds outside the
        allocator. Free blocks the allocator keeps are left out because it releases them when an allocation would fail."""
        if "device_used" not in stats:
            return None
        if "peak_allocated" not in stats or "reserved" not in stats:
            return stats["device_used"]
        return stats["peak_allocated"] + max(stats["device_used"] - stats["reserved"], 0.0)

    def start(self) -> None:
        """Settle the limit. Refuses early when the card cannot give what was asked for.

        Every number here is the driver's view of the whole card, other programs included: that is what runs out.
        """
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
        self._note("limit", limit_mb=round(self.limit_mb), in_use_mb=round(stats["device_used"]))
        self._cap(max(stats["device_used"] - stats.get("reserved", 0.0), 0.0) + self.OUTSIDE_HEADROOM_MB, stats)

    def _cap(self, outside_mb: float, stats: Mapping[str, float]) -> None:
        """Keep the allocator below the limit minus what lives outside it. Past that line it frees its cached blocks
        instead of letting them crowd out the driver, whose allocations (a new shape's kernels) it cannot satisfy."""
        if self.limit_mb is None:
            return
        cap = max(self.limit_mb - outside_mb, stats.get("allocated", 0.0) + self.OUTSIDE_SLACK_MB)
        if self._cap_mb is not None and abs(cap - self._cap_mb) < 128:
            return
        if self.accelerator.cap_allocator(cap):
            self._outside_mb, self._cap_mb = outside_mb, cap
            self._note("allocator_cap", cap_mb=round(cap), outside_mb=round(outside_mb))

    def _follow_outside(self, stats: Mapping[str, float]) -> None:
        """The outside share grows as new shapes bring kernels, and the cap follows it down; once calibrated it is set from what was measured."""
        if "device_used" not in stats or "reserved" not in stats or self._cap_mb is None:
            return
        measured = max(stats["device_used"] - stats["reserved"], 0.0) + self.OUTSIDE_SLACK_MB
        if measured > self._outside_mb or self.steps_seen == self.policy.calibration_steps:
            self._cap(measured, stats)

    def recover(self, step: int, error: BaseException) -> None:
        """After an out-of-memory error: drop what the failed step held, give the cache back, and assume the driver
        needs more room from here on."""
        gc.collect()
        self.accelerator.empty_cache()
        stats = self.accelerator.memory()
        self._note("out_of_memory", step=step + 1, error=str(error).splitlines()[0][:160] if str(error) else type(error).__name__,
                   parts=self.breakdown(stats))
        if self._cap_mb is not None:
            self._cap(self._outside_mb + self.OUTSIDE_SLACK_MB, stats)

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
        """Offload unpinned residents, lowest rank first, until ``mb`` more fits under the ceiling; refuse past the limit."""
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
        if used is not None and self.limit_mb is not None and used + mb > self.limit_mb:
            raise MemoryBudgetExceeded(f"{why}: {mb:.0f} MB do not fit ({used:.0f} MB in use of {self.limit_mb:.0f} MB; "
                                       f"{self.breakdown()})")
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
        need = self.need_mb(stats)
        if used is not None:
            if self.steps_seen == 1 or used > self.peak_used_mb * 1.01:
                self._note("breakdown", step=step + 1, used_mb=round(used), need_mb=round(need), parts=self.breakdown(stats))
            self.peak_used_mb = max(self.peak_used_mb, used)
            self._follow_outside(stats)
            self._react(step, need)
        moved = False
        for entry in self._entries.values():
            if entry.parked and not entry.away:
                entry.resident.offload()
                entry.away = moved = True
        self.accelerator.reset_peak()
        released = moved or any(e.parked or e.away for e in self._entries.values())
        if released and self.empty_cache_every and self.steps_seen % self.empty_cache_every == 0:
            self.accelerator.empty_cache()
        elif used is not None and self._cache_crowds_card(stats):
            self.accelerator.empty_cache()
            self._note("trim", step=step + 1, cached_mb=round(stats["reserved"] - stats["allocated"]), used_mb=round(used))

    def _cache_crowds_card(self, stats: Mapping[str, float]) -> bool:
        """Cached blocks are free to give back, but memory the driver hands out outside the allocator (kernel and
        primitive scratch for a new shape) cannot make the allocator release them: it fails while the cache sits on the card."""
        if self.limit_mb is None or "reserved" not in stats or "allocated" not in stats:
            return False
        return stats["device_used"] > self.ceiling_mb and stats["reserved"] - stats["allocated"] > self.MIN_TRIM_MB

    def _react(self, step: int, used: float) -> None:
        """``used`` is the step's need (see ``need_mb``)."""
        if self.limit_mb is None:
            return
        if used > self.limit_mb and self.policy.strict:
            raise MemoryBudgetExceeded(f"step {step + 1}: {used:.0f} MB needed is over the {self.limit_mb:.0f} MB budget "
                                       f"(residents {self.summary()})")
        if used <= self.ceiling_mb or self.steps_seen > self.policy.calibration_steps:
            if used > self.ceiling_mb and used > self._pressure_noted:
                self._pressure_noted = used
                self._note("pressure", step=step + 1, used_mb=round(used), settled=True)
            return
        self._note("pressure", step=step + 1, used_mb=round(used), settled=False)
        self._escalate()

    def _escalate(self) -> None:
        """Park the cheapest resident to move (smallest footprint) that is not parked yet; once parked, it stays parked."""
        movable = [(e.resident.footprint_bytes(), name, e) for name, e in self._entries.items()
                   if not e.pinned and not e.parked and e.resident.footprint_bytes() > 0]
        if not movable:
            self._note("lever", name="none left")
            return
        _, name, entry = min(movable, key=lambda item: item[:2])
        entry.parked = True
        self._note("lever", name=f"park {name} between steps")

    def _note(self, event: str, **fields: Any) -> None:
        record: Mapping[str, Any] = {"event": event, **fields}
        self.trace.append(dict(record))
        if self.report is not None:
            self.report(dict(record))
