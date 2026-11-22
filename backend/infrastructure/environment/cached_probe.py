"""A cache in front of a device probe, so a web request never triggers a torch import."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeGuard, TypeVar

from ...application.limits import DEVICE_REFRESH_MIN_SECONDS, READINESS_CACHE_SECONDS
from ...application.ports.environment import DeviceProbe, DeviceReport

#: A cache entry: (taken_at, ...payload). Any shape the two caches use fits; the TypeVar lets the freshness guard narrow both.
_CacheEntry = TypeVar("_CacheEntry", bound=tuple)


@dataclass(slots=True)
class CachedDeviceProbe(DeviceProbe):
    """Single-flight and cached, so one torch import answers many requests.

    **The problem this exists for, measured.** `GET /installer/devices`
    costs a torch import (1.7 s on this machine) and a plain GET is
    deliberately *not* behind the Origin check -- ADR 0001 guards only
    state-changing methods, because a read cannot be the attack. So any web
    page the user has open can trigger it with an `<img src=...>`, and
    before this:

        8 concurrent check.execute() -> 8 probe invocations, peak 8
        3 sequential  check.execute() -> 3 probes (no cache at all)

    Each of those eight initialises the accelerator runtime. On a 12 GB card
    that is not merely slow, it competes for VRAM with whatever else is
    using the device -- a training run, or a game.

    **It wraps the port rather than the use case**, and that placement is
    the point: `/installer/devices` asks the same question of the same
    subprocess. A cache on `CheckRequirements` would have left the newer
    endpoint exactly as exposed as the old one.

    **Single-flight and cache are separate jobs.** The lock is what stops
    eight requests becoming eight processes; the TTL is what stops the
    *next* request becoming another one. A lock alone still costs one torch
    import per request in sequence.

    **Refuses to probe while a run is active.** `is_busy` is asked before
    starting a probe, and the answer is a report saying so rather than an
    empty device list -- see the reason text. Taking VRAM from a running
    training step to answer a wizard question is the wrong trade, and the
    user is told which trade was made for them.

    Monotonic time, not the injected `Clock`: a TTL must not be affected by
    a wall-clock jump, and `Clock` returns a datetime for exactly that
    reason. `now` is injectable so tests need not sleep.
    """

    inner: DeviceProbe
    #: Defaults to the limits rather than repeating them. A literal 30.0
    #: here was a second copy of a documented constant, and it was dead --
    #: both production wiring and every test pass the constant explicitly,
    #: so changing it there would have left this behind, still reading as
    #: though it governed anything.
    ttl: float = READINESS_CACHE_SECONDS
    refresh_floor: float = DEVICE_REFRESH_MIN_SECONDS
    #: Asked before starting a probe. True means "a run is using the card".
    is_busy: Callable[[], bool] = lambda: False
    now: Callable[[], float] = time.monotonic

    _condition: threading.Condition = field(
        default_factory=lambda: threading.Condition(), repr=False)
    #: (taken_at, report) for `report()`; the same shape for the list.
    _report_cache: tuple[float, DeviceReport] | None = None
    _list_cache: tuple[float, tuple[DeviceReport, ...], str | None] | None = None
    _probing: bool = field(default=False, repr=False)
    #: When the last *real* probe finished.
    _last_probe_at: float | None = None
    #: When a refresh was last *granted*. Separate from `_last_probe_at`
    #: because they bound different things: the floor is about how often
    #: the user may insist, not how recently anything was measured.
    #:
    #: Measuring the floor from `_last_probe_at` made "Re-check" dead on
    #: arrival -- populating the cache on page load sets that timestamp, so
    #: the first click after load was always rate limited, which is the
    #: click a user is most likely to make.
    _last_refresh_at: float | None = None
    #: Counted so a test can assert a probe did or did not happen.
    probes_run: int = field(default=0, repr=False)

    # -- the two questions, each cached separately ----------------------

    def _fresh(self, cache, ttl: float) -> TypeGuard[_CacheEntry]:
        """Whether a cache entry is present *and* still inside its TTL.

        A `TypeGuard` rather than a plain bool because every caller
        immediately indexes the entry, and "fresh" already means "present"
        -- stating that lets the callers read as `if self._fresh(c): c[1]`
        instead of re-checking `c is not None` six times, which is how the
        first version of this had six identical mypy errors instead of
        one.
        """
        return cache is not None and (self.now() - cache[0]) < ttl

    def _busy_report(self) -> DeviceReport:
        """The answer given instead of touching the card.

        `present=False` with a reason, so a caller that only checks
        `present` sees "not usable" -- and the reason says the accelerator
        was deliberately left alone rather than being missing. Those are
        different facts and the UI renders them differently, the same
        distinction `ReadinessReport.device_checked` exists for.
        """
        return DeviceReport(
            present=False,
            reason=(
                "a training run is using the graphics card, so it was not "
                "asked -- starting a probe here would take memory from the "
                "run. This is a deliberate choice, not a missing card."
            ),
        )

    def report(self) -> DeviceReport:
        cached = self._report_cache
        if self._fresh(cached, self.ttl):
            return cached[1]

        with self._condition:
            # Re-check inside the lock: a thread that queued behind another
            # probe should take its answer, not start its own.
            cached = self._report_cache
            if self._fresh(cached, self.ttl):
                return cached[1]
            while self._probing:
                self._condition.wait(timeout=self.ttl + 5.0)
                cached = self._report_cache
                if self._fresh(cached, self.ttl):
                    return cached[1]
                break
            if self.is_busy():
                return self._busy_report()
            self._probing = True

        try:
            result = self.inner.report()
        finally:
            with self._condition:
                self._probing = False
                self._last_probe_at = self.now()
                self.probes_run += 1
                self._condition.notify_all()
        with self._condition:
            self._report_cache = (self._last_probe_at, result)
        return result

    def devices_with_reason(self) -> tuple[tuple[DeviceReport, ...], str | None]:
        cached = self._list_cache
        if self._fresh(cached, self.ttl):
            return cached[1], cached[2]

        with self._condition:
            cached = self._list_cache
            if self._fresh(cached, self.ttl):
                return cached[1], cached[2]
            while self._probing:
                self._condition.wait(timeout=self.ttl + 5.0)
                cached = self._list_cache
                if self._fresh(cached, self.ttl):
                    return cached[1], cached[2]
                break
            if self.is_busy():
                return (), self._busy_report().reason
            self._probing = True

        try:
            rows, reason = self.inner.devices_with_reason()
        finally:
            with self._condition:
                self._probing = False
                self._last_probe_at = self.now()
                self.probes_run += 1
                self._condition.notify_all()
        with self._condition:
            self._list_cache = (self._last_probe_at, rows, reason)
        return rows, reason

    def devices(self) -> tuple[DeviceReport, ...]:
        rows, _ = self.devices_with_reason()
        return rows

    @property
    def enumerate_all(self) -> bool:
        return self.inner.enumerate_all

    @property
    def backend(self) -> str:
        return getattr(self.inner, "backend", "xpu")

    # -- the refresh path -------------------------------------------------

    def invalidate(self) -> bool:
        """Drop the cache and probe now, subject to the refresh floor.

        Returns whether a probe actually ran. `?refresh=true` maps onto
        this, so a "Re-check" button that is clicked three times in a second
        causes one probe and reports the other two as rate limited, rather
        than starting three torch imports.
        """
        with self._condition:
            now = self.now()
            if (self._last_refresh_at is not None
                    and (now - self._last_refresh_at) < self.refresh_floor):
                return False
            self._last_refresh_at = now
            self._report_cache = None
            self._list_cache = None
        self.report()
        return True

