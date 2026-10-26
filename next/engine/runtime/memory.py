"""What a run may put on the device, what is there now, and what moves off when it gets tight."""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable, Mapping

from ...graph.settings import AUTO
from ..devices import Accelerator
from .planner import StepPlanner
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
    """The ``memory.*`` and ``runtime.*`` settings. ``limit_mb`` None means the whole card.

    ``margin_mb`` is the one safety margin: memory kept free below the limit for what cannot be predicted (kernels
    and primitives of a shape not seen yet, driver growth). Every threshold below is the limit minus measured
    outside memory minus this, and nothing adds a second margin on top.
    """

    limit_mb: float | None = None
    floor_mb: float = 0.0
    strict: bool = True
    margin_mb: float = 512.0
    calibration_steps: int = 3
    empty_cache_every: int = 1

    @classmethod
    def from_settings(cls, settings: Any) -> MemoryPolicy:
        limit = settings.get("memory.vram_max_mb")
        return cls(None if limit in (None, AUTO) else float(limit), float(settings.get("memory.vram_min_mb") or 0),
                   bool(settings.get("memory.strict")), float(settings.get("memory.margin_mb")),
                   int(settings.get("runtime.residency.calibration_steps")), int(settings.get("runtime.empty_cache_every_n_steps")))


@dataclass
class _Entry:
    resident: DeviceResident
    purpose: str
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
    MIN_NOTE_MB = 64.0
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
        self.planner = StepPlanner()
        self._planned: tuple[float, int] | None = None  # live memory and pixel count the coming step was planned with
        self._started = False
        self._outside_mb = 0.0  # what the allocator cap currently assumes the driver holds outside it
        self._cap_mb: float | None = None

    def claim(self, name: str, resident: DeviceResident, *, purpose: str, rank: int = 0, pinned: bool = False) -> None:
        """Every part of a run that keeps memory on the device says what it keeps and why. Lower ranks move off the
        device first; pinned claims never move."""
        if not purpose.strip():
            raise ValueError(f"the claim {name!r} must say what the memory is for")
        self._entries[name] = _Entry(resident, purpose, rank, pinned)

    def ledger(self, stats: Mapping[str, float] | None = None) -> list[dict]:
        """Every megabyte the card holds, in exactly one row: claims, live tensors no claim covers, the allocator's
        free memory, what this process holds outside the allocator, and other programs. The monitor shows these rows."""
        stats = stats if stats is not None else self.accelerator.memory()
        rows = [{"name": name, "mb": round(e.resident.footprint_bytes() / MB, 1), "purpose": e.purpose}
                for name, e in self._entries.items()]
        if "allocated" not in stats:
            return rows
        claimed = sum(r["mb"] for r in rows)
        rows.append({"name": "unclaimed live", "mb": round(max(stats["allocated"] - claimed, 0.0), 1),
                     "purpose": "live tensors no claim covers (inputs and outputs of the step in flight, cached prompts, leftovers)"})
        free = max(stats.get("reserved", 0.0) - stats["allocated"], 0.0)
        if "split_free" in stats:
            stuck = min(stats["split_free"], free)
            rows.append({"name": "allocator free, returnable", "mb": round(free - stuck, 1),
                         "purpose": "whole free segments; handed back when the card needs them"})
            rows.append({"name": "allocator free, stuck", "mb": round(stuck, 1),
                         "purpose": "free pieces between live blocks; they stay while a neighbour lives (fragmentation)"})
        else:
            rows.append({"name": "allocator free blocks", "mb": round(free, 1), "purpose": "memory the allocator keeps for reuse"})
        if "device_used" in stats:
            own, other = self.outside(stats)
            if own is None:
                rows.append({"name": "outside allocator", "mb": round(other, 1),
                             "purpose": "this process (context, kernels, primitives) and other programs; the kernel does not say how it splits"})
            else:
                rows.append({"name": "outside allocator (ours)", "mb": round(own, 1),
                             "purpose": "this process's context, compiled kernels, primitive scratch and driver pools"})
                rows.append({"name": "other programs", "mb": round(other, 1), "purpose": "the desktop and everything else on the card"})
        return rows

    @staticmethod
    def outside(stats: Mapping[str, float]) -> tuple[float | None, float]:
        """``(this process's memory outside the allocator, everything else on the card)``; the first is None when the
        kernel does not report this process's share, and then the second is the whole outside."""
        outside = max(stats["device_used"] - stats.get("reserved", 0.0), 0.0)
        if "process" not in stats:
            return None, outside
        own = max(stats["process"] - stats.get("reserved", 0.0), 0.0)
        return own, max(outside - own, 0.0)

    @staticmethod
    def returnable_mb(stats: Mapping[str, float]) -> float:
        """What emptying the cache can really give back: free memory in whole segments, not pieces between live blocks."""
        free = max(stats.get("reserved", 0.0) - stats.get("allocated", 0.0), 0.0)
        return max(free - stats.get("split_free", 0.0), 0.0)

    def give_back(self, why: str) -> None:
        """Empty the allocator's cache and note what it returned and what the driver then showed, so memory that does not
        come back (or appears elsewhere) is visible at the moment it happens."""
        before = self.accelerator.memory()
        self.accelerator.empty_cache()
        after = self.accelerator.memory()
        freed = before.get("reserved", 0.0) - after.get("reserved", 0.0)
        if freed >= self.MIN_NOTE_MB:
            own_before, other_before = self.outside(before) if "device_used" in before else (None, 0.0)
            own_after, other_after = self.outside(after) if "device_used" in after else (None, 0.0)
            self._note("cache_returned", why=why, returned_mb=round(freed), card_freed_mb=round(before.get("device_used", 0.0) - after.get("device_used", 0.0)),
                       ours_outside_mb=None if own_after is None else round(own_after - own_before),
                       other_programs_mb=round(other_after - other_before))

    def settle(self) -> None:
        """Once every persistent claim is in place: drop the garbage of loading and give back the cache, so what the
        run keeps for good sits together and the steps' transient blocks form the rest."""
        gc.collect()
        self.give_back("settle")
        self._note("claims", rows=self.ledger())

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
        if self._started:
            return
        self._started = True
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
                   margin_mb=round(self.policy.margin_mb), allocator_config=os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_XPU_ALLOC_CONF") or "default")
        self._cap(max(stats["device_used"] - stats.get("reserved", 0.0), 0.0), "start", stats)

    def _cap(self, outside_mb: float, why: str, stats: Mapping[str, float]) -> None:
        """Keep the allocator below the limit minus what lives outside it (measured) minus the margin. Past that line
        it frees its cached blocks instead of letting them crowd out the driver, whose allocations (a new shape's
        kernels) it cannot satisfy by releasing anything."""
        margin_mb = self.policy.margin_mb
        if self.limit_mb is None:
            return
        assumed = outside_mb + margin_mb
        cap = max(self.limit_mb - assumed, stats.get("allocated", 0.0) + margin_mb)
        if self._cap_mb is not None and abs(cap - self._cap_mb) < 128:
            return
        if self.accelerator.cap_allocator(cap):
            self._outside_mb, self._cap_mb = assumed, cap
            own, other = self.outside(stats) if "device_used" in stats else (None, 0.0)
            self._note("allocator_cap", why=why, cap_mb=round(cap), limit_mb=round(self.limit_mb), outside_mb=round(outside_mb),
                       ours_outside_mb=None if own is None else round(own), other_programs_mb=round(other), margin_mb=round(margin_mb))

    def _follow_outside(self, stats: Mapping[str, float]) -> None:
        """The outside share grows as new shapes bring kernels, and the cap follows it down; once calibrated it is set from what was measured."""
        if "device_used" not in stats or "reserved" not in stats or self._cap_mb is None:
            return
        outside = max(stats["device_used"] - stats["reserved"], 0.0)
        if outside + self.policy.margin_mb > self._outside_mb or self.steps_seen == self.policy.calibration_steps:
            self._cap(outside, "outside memory grew" if self.steps_seen != self.policy.calibration_steps else "calibrated", stats)

    def budget_mb(self) -> float | None:
        """What live and transient tensors together may reach: the allocator's cap."""
        return self._cap_mb

    def graph_fits(self, cost_mb: float) -> bool:
        """Can another captured graph, which takes ``cost_mb`` outside the allocator and so lowers its cap by as much, be added while the
        cap still holds the largest step planned so far?"""
        budget = self.budget_mb()
        if budget is None or not self._planned:
            return True
        need = self.planner.predicted_peak_mb(*self._planned)
        return need is None or budget - cost_mb >= need

    def plan(self, count: int, units_per_sample: int) -> list[int]:
        """Sizes of the chunks a batch of ``count`` samples (each ``units_per_sample`` latent pixels) runs as."""
        floor = self.accelerator.memory().get("allocated", 0.0)
        sizes = self.planner.chunks(count, units_per_sample, floor, self.budget_mb())
        units = max(sizes) * units_per_sample
        self._planned = (floor, max(units, self._planned[1])) if self._planned else (floor, units)
        if len(sizes) > 1:
            self._note("split", samples=count, chunks=sizes, predicted_mb=round(self.planner.predicted_peak_mb(floor, count * units_per_sample) or 0),
                       budget_mb=round(self.budget_mb() or 0))
        return sizes

    @contextmanager
    def measuring(self, units: int):
        """Run something whose peak says what ``units`` latent pixels cost (the probe), leaving no trace in the peak.
        Yields a dict that holds ``cost`` (megabytes per pixel) once the block has run."""
        self.accelerator.reset_peak()
        floor = self.accelerator.memory().get("allocated", 0.0)
        result: dict[str, float] = {}
        yield result
        stats = self.accelerator.memory()
        self.planner.learn(floor, stats.get("peak_allocated", floor), units)
        result["cost"] = max(stats.get("peak_allocated", floor) - floor, 0.0) / units if units else 0.0
        self.accelerator.reset_peak()
        own, other = self.outside(stats) if "device_used" in stats else (None, 0.0)
        self._note("cost", mb_per_megapixel=round(result["cost"] * 2**20, 1), floor_mb=round(floor),
                   peak_mb=round(stats.get("peak_allocated", floor)), ours_outside_mb=None if own is None else round(own),
                   other_programs_mb=round(other))

    def recover(self, step: int, error: BaseException) -> None:
        """After an out-of-memory error: drop what the failed step held, give the cache back, and assume the driver
        needs more room from here on."""
        gc.collect()
        self.give_back("out of memory")
        stats = self.accelerator.memory()
        self._note("out_of_memory", step=step + 1, error=str(error).splitlines()[0][:160] if str(error) else type(error).__name__,
                   parts=self.breakdown(stats))
        if self._cap_mb is not None:
            self._cap(self._outside_mb, "after out of memory", stats)  # the assumption already holds one margin; this adds another

    def ensure_started(self) -> None:
        """Nodes that bring anything to the device before training call this first, so the allocator is capped
        and room is made from the first allocation on."""
        self.start()

    @property
    def ceiling_mb(self) -> float | None:
        return None if self.limit_mb is None else self.limit_mb - self.policy.margin_mb

    def breakdown(self, stats: Mapping[str, float] | None = None) -> dict[str, float]:
        """The ledger as ``{name: megabytes}``."""
        return {row["name"]: row["mb"] for row in self.ledger(stats)}

    def summary(self) -> dict[str, float]:
        return {name: round(e.resident.footprint_bytes() / MB, 1) for name, e in self._entries.items()}

    def _fits(self, mb: float, hard: bool) -> bool:
        """Whether ``mb`` more fits on the card and under the allocator's cap; ``hard`` allows the margin to be used."""
        stats = self.accelerator.memory()
        line = self.limit_mb if hard else self.ceiling_mb
        if "device_used" in stats and line is not None and stats["device_used"] + mb > line:
            return False
        line = None if self._cap_mb is None else self._cap_mb + (self.policy.margin_mb if hard else 0.0)
        return not ("allocated" in stats and line is not None and stats["allocated"] + mb > line)

    def make_room(self, mb: float, why: str) -> list[str]:
        """Offload unpinned residents, lowest rank first, until ``mb`` more fits under the ceiling and the allocator's
        cap; refuse (naming where the memory is) when it does not fit even then."""
        moved: list[str] = []
        for name, entry in sorted(self._entries.items(), key=lambda kv: kv[1].rank):
            if self._fits(mb, hard=False):
                break
            if entry.pinned or entry.away or entry.resident.footprint_bytes() == 0:
                continue
            entry.resident.offload()
            entry.away = True
            self.give_back(f"{name} moved off the device")
            moved.append(name)
        if moved:
            self._note("evict", why=why, moved=moved)
        if not self._fits(mb, hard=True):
            used = self.used_mb()
            raise MemoryBudgetExceeded(
                f"{why}: {mb:.0f} MB do not fit ({0 if used is None else used:.0f} MB in use of {self.limit_mb:.0f} MB, "
                f"allocator cap {self._cap_mb or 0:.0f} MB; {self.breakdown()}). Raise memory.vram_max_mb, lower memory.margin_mb, "
                "or train a smaller model state (rank, batch size, int8 base weights).")
        return moved

    def restore(self, names: list[str]) -> None:
        for name in names:
            entry = self._entries[name]
            entry.resident.reload()
            entry.away = False

    def before_step(self, step: int) -> None:
        for entry in self._entries.values():
            if entry.away:  # parked ones, and ones moved to make room for something that is not on the device now
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
                self._note("breakdown", step=step + 1, used_mb=round(used), need_mb=round(need), peak_mb=round(stats.get("peak_allocated", 0.0)),
                           segments=stats.get("segments"), alloc_retries=stats.get("alloc_retries"), parts=self.breakdown(stats))
            self.peak_used_mb = max(self.peak_used_mb, used)
            self._follow_outside(stats)
            self._react(step, need)
        if self._planned is not None and "peak_allocated" in stats:
            self.planner.learn(self._planned[0], stats["peak_allocated"], self._planned[1])
            self._planned = None
        moved = False
        for entry in self._entries.values():
            if entry.parked and not entry.away:
                entry.resident.offload()
                entry.away = moved = True
        self.accelerator.reset_peak()
        released = moved or any(e.parked or e.away for e in self._entries.values())
        if released and self.empty_cache_every and self.steps_seen % self.empty_cache_every == 0:
            self.give_back("a resident was released")
        elif used is not None and self._cache_crowds_card(stats):
            self.give_back("cache crowds the card")
            self._note("trim", step=step + 1, returnable_mb=round(self.returnable_mb(stats)), used_mb=round(used))

    def _cache_crowds_card(self, stats: Mapping[str, float]) -> bool:
        """Cached blocks are free to give back, but memory the driver hands out outside the allocator (kernel and
        primitive scratch for a new shape) cannot make the allocator release them: it fails while the cache sits on the card."""
        if self.limit_mb is None or self._cap_mb is not None or "reserved" not in stats or "allocated" not in stats:
            return False  # under an allocator cap the allocator frees its own cache at the line; doing it every step only costs time
        return stats["device_used"] > self.ceiling_mb and self.returnable_mb(stats) > self.MIN_TRIM_MB

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

    def note(self, event: str, **fields: Any) -> None:
        self._note(event, **fields)

    def _note(self, event: str, **fields: Any) -> None:
        record: Mapping[str, Any] = {"event": event, **fields}
        self.trace.append(dict(record))
        if self.report is not None:
            self.report(dict(record))
