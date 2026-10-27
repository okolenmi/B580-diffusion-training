"""What a run keeps on the device, and what moves off when the card gets tight."""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ..devices import Accelerator
from . import card
from .cap import AllocatorCap
from .planner import StepPlanner
from .policy import MemoryBudgetExceeded, MemoryPolicy
from .residents import Claim, DeviceResident

__all__ = ["MemoryBudgetExceeded", "MemoryPolicy", "ResidencyManager", "is_out_of_memory"]

OOM_MARKERS = ("out of memory", "out_of_resources", "out_of_device_memory", "out of device memory")


def is_out_of_memory(error: BaseException) -> bool:
    """True for the allocator's error and for the driver's own refusals, which arrive as plain runtime errors."""
    return isinstance(error, MemoryError) or type(error).__name__ == "OutOfMemoryError" or \
        (isinstance(error, RuntimeError) and any(marker in str(error).lower() for marker in OOM_MARKERS))


@dataclass
class ResidencyManager:
    """Knows every resident and reacts between steps; the allocator's cap (``AllocatorCap``) and the arithmetic on
    the driver's numbers (``card``) are separate.

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
        self._claims: dict[str, Claim] = {}
        self.cap = AllocatorCap(self.accelerator, self.policy, self._note)
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

    @property
    def limit_mb(self) -> float | None:
        return self.cap.limit_mb

    @property
    def ceiling_mb(self) -> float | None:
        return self.cap.ceiling_mb

    def budget_mb(self) -> float | None:
        """What live and transient tensors together may reach: the allocator's cap."""
        return self.cap.cap_mb

    # -- claims -------------------------------------------------------------------------------------------------

    def claim(self, name: str, resident: DeviceResident, *, purpose: str, rank: int = 0, pinned: bool = False) -> None:
        """Every part of a run that keeps memory on the device says what it keeps and why. Lower ranks move off the
        device first; pinned claims never move."""
        if not purpose.strip():
            raise ValueError(f"the claim {name!r} must say what the memory is for")
        self._claims[name] = Claim(resident, purpose, rank, pinned)

    def ledger(self, stats: Mapping[str, float] | None = None) -> list[dict]:
        return card.ledger_rows(self._claims, stats if stats is not None else self.accelerator.memory())

    def breakdown(self, stats: Mapping[str, float] | None = None) -> dict[str, float]:
        """The ledger as ``{name: megabytes}``."""
        return {row["name"]: row["mb"] for row in self.ledger(stats)}

    def summary(self) -> dict[str, float]:
        return {name: round(c.resident.footprint_bytes() / card.MB, 1) for name, c in self._claims.items()}

    def used_mb(self) -> float | None:
        """Device memory in use as the driver sees it (other processes and the allocator's cache included)."""
        return self.accelerator.memory().get("device_used")

    # -- start and the cache ------------------------------------------------------------------------------------

    def start(self) -> None:
        if not self._started:
            self._started = True
            self.cap.settle()

    def ensure_started(self) -> None:
        """Nodes that bring anything to the device before training call this first, so the allocator is capped
        and room is made from the first allocation on."""
        self.start()

    def give_back(self, why: str) -> None:
        """Empty the allocator's cache and note what it returned and what the driver then showed, so memory that does not
        come back (or appears elsewhere) is visible at the moment it happens."""
        before = self.accelerator.memory()
        self.accelerator.empty_cache()
        after = self.accelerator.memory()
        freed = before.get("reserved", 0.0) - after.get("reserved", 0.0)
        if freed >= self.MIN_NOTE_MB:
            own_before, other_before = card.outside_or_unknown(before)
            own_after, other_after = card.outside_or_unknown(after)
            self._note("cache_returned", why=why, returned_mb=round(freed), card_freed_mb=round(before.get("device_used", 0.0) - after.get("device_used", 0.0)),
                       ours_outside_mb=None if own_after is None else round(own_after - own_before),
                       other_programs_mb=round(other_after - other_before))

    def settle(self) -> None:
        """Once every persistent claim is in place: drop the garbage of loading and give back the cache, so what the
        run keeps for good sits together and the steps' transient blocks form the rest."""
        gc.collect()
        self.give_back("settle")
        self._note("claims", rows=self.ledger())

    def recover(self, step: int, error: BaseException) -> None:
        """After an out-of-memory error: drop what the failed step held, give the cache back, and assume the driver
        needs more room from here on."""
        gc.collect()
        self.give_back("out of memory")
        stats = self.accelerator.memory()
        self._note("out_of_memory", step=step + 1, error=str(error).splitlines()[0][:160] if str(error) else type(error).__name__,
                   parts=self.breakdown(stats))
        self.cap.retreat(stats)

    # -- planning the step --------------------------------------------------------------------------------------

    def graph_fits(self, cost_mb: float, units: int = 0) -> bool:
        """Can another captured graph be added? It takes ``cost_mb`` outside the allocator, and the pool the graphs share
        keeps the activations of its largest step (``units`` latent pixels) for itself: eager steps cannot use that
        memory even while it is free. The cap must still hold the largest step planned so far on top of both."""
        budget = self.budget_mb()
        if budget is None or not self._planned:
            return True
        floor = self._planned[0]
        need = self.planner.predicted_peak_mb(*self._planned)
        pool = max((self.planner.predicted_peak_mb(floor, units) or floor) - floor, 0.0) if units else 0.0
        return need is None or budget - cost_mb - pool >= need

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
        own, other = card.outside_or_unknown(stats)
        self._note("cost", mb_per_megapixel=round(result["cost"] * 2**20, 1), floor_mb=round(floor),
                   peak_mb=round(stats.get("peak_allocated", floor)), ours_outside_mb=None if own is None else round(own),
                   other_programs_mb=round(other))

    # -- making room --------------------------------------------------------------------------------------------

    def make_room(self, mb: float, why: str) -> list[str]:
        """Offload unpinned residents, lowest rank first, until ``mb`` more fits under the ceiling and the allocator's
        cap; refuse (naming where the memory is) when it does not fit even then."""
        moved: list[str] = []
        for name, claim in sorted(self._claims.items(), key=lambda kv: kv[1].rank):
            if self.cap.fits(self.accelerator.memory(), mb, hard=False):
                break
            if claim.pinned or claim.away or claim.resident.footprint_bytes() == 0:
                continue
            claim.resident.offload()
            claim.away = True
            self.give_back(f"{name} moved off the device")
            moved.append(name)
        if moved:
            self._note("evict", why=why, moved=moved)
        if not self.cap.fits(self.accelerator.memory(), mb, hard=True):
            used = self.used_mb()
            raise MemoryBudgetExceeded(
                f"{why}: {mb:.0f} MB do not fit ({0 if used is None else used:.0f} MB in use of {self.limit_mb:.0f} MB, "
                f"allocator cap {self.cap.cap_mb or 0:.0f} MB; {self.breakdown()}). Raise memory.vram_max_mb, lower memory.margin_mb, "
                "or train a smaller model state (rank, batch size, int8 base weights).")
        return moved

    def restore(self, names: list[str]) -> None:
        for name in names:
            claim = self._claims[name]
            claim.resident.reload()
            claim.away = False

    # -- around a step ------------------------------------------------------------------------------------------

    def before_step(self, step: int) -> None:
        for claim in self._claims.values():
            if claim.away:  # parked ones, and ones moved to make room for something that is not on the device now
                claim.resident.reload()
                claim.away = False

    def after_step(self, step: int) -> None:
        self.steps_seen += 1
        stats = self.accelerator.memory()
        used = self.last_used_mb = stats.get("device_used")
        self.last_stats = stats
        need = card.need_mb(stats)
        if used is not None:
            if self.steps_seen == 1 or used > self.peak_used_mb * 1.01:
                self._note("breakdown", step=step + 1, used_mb=round(used), need_mb=round(need), peak_mb=round(stats.get("peak_allocated", 0.0)),
                           segments=stats.get("segments"), alloc_retries=stats.get("alloc_retries"), parts=self.breakdown(stats))
            self.peak_used_mb = max(self.peak_used_mb, used)
            self.cap.follow(stats, self.steps_seen)
            self._react(step, need)
        if self._planned is not None and "peak_allocated" in stats:
            self.planner.learn(self._planned[0], stats["peak_allocated"], self._planned[1])
            self._planned = None
        moved = False
        for claim in self._claims.values():
            if claim.parked and not claim.away:
                claim.resident.offload()
                claim.away = moved = True
        self.accelerator.reset_peak()
        released = moved or any(c.parked or c.away for c in self._claims.values())
        if released and self.empty_cache_every and self.steps_seen % self.empty_cache_every == 0:
            self.give_back("a resident was released")
        elif used is not None and self._cache_crowds_card(stats):
            self.give_back("cache crowds the card")
            self._note("trim", step=step + 1, returnable_mb=round(card.returnable_mb(stats)), used_mb=round(used))

    def _cache_crowds_card(self, stats: Mapping[str, float]) -> bool:
        """Cached blocks are free to give back, but memory the driver hands out outside the allocator (kernel and
        primitive scratch for a new shape) cannot make the allocator release them: it fails while the cache sits on the card."""
        if self.limit_mb is None or self.cap.cap_mb is not None or "reserved" not in stats or "allocated" not in stats:
            return False  # under an allocator cap the allocator frees its own cache at the line; doing it every step only costs time
        return stats["device_used"] > self.ceiling_mb and card.returnable_mb(stats) > self.MIN_TRIM_MB

    def _react(self, step: int, used: float) -> None:
        """``used`` is the step's need (see ``card.need_mb``)."""
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
        movable = [(c.resident.footprint_bytes(), name, c) for name, c in self._claims.items()
                   if not c.pinned and not c.parked and c.resident.footprint_bytes() > 0]
        if not movable:
            self._note("lever", name="none left")
            return
        _, name, claim = min(movable, key=lambda item: item[:2])
        claim.parked = True
        self._note("lever", name=f"park {name} between steps")

    def note(self, event: str, **fields: Any) -> None:
        self._note(event, **fields)

    def _note(self, event: str, **fields: Any) -> None:
        record: Mapping[str, Any] = {"event": event, **fields}
        self.trace.append(dict(record))
        if self.report is not None:
            self.report(dict(record))
