"""The device probe must cost one torch import, not one per request.

Round-4 finding R4-01. Measured on this machine before the fix:

    8 concurrent check.execute()  -> 8 probe invocations, peak concurrency 8
    3 sequential  check.execute()  -> 3 probes (no cache at all)

Each probe imports torch and initialises the accelerator runtime. On a 12 GB
card that competes for VRAM with whatever else is using the device -- a
training run, or a game. And a plain GET is deliberately *not* behind the
Origin check (ADR 0001 guards state-changing methods), so any web page the
user has open can trigger it with an `<img src=...>`.

So the assertions below are about **counts**, measured with a counting fake
and an injected clock, not about behaviour in the abstract. A test that
only checked "the second call returns the same object" would pass against a
cache that still let eight concurrent callers through, which is the case
that matters.

The cache wraps the *port* rather than the use case, so `/installer/devices`
is covered by the same tests -- it is the same subprocess answering a
different question about the same card.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.limits import (  # noqa: E402
    DEVICE_REFRESH_MIN_SECONDS,
    READINESS_CACHE_SECONDS,
)
from backend.application.ports.environment import (  # noqa: E402
    CachedDeviceProbe,
    DeviceProbe,
    DeviceReport,
)
from backend.tests.support import check, finish  # noqa: E402


class CountingProbe(DeviceProbe):
    """Counts invocations and records the peak concurrency.

    `delay` stands in for the torch import. Long enough that a fan-out is
    unambiguous -- if eight callers really do run at once, eight threads
    are inside `report()` and the peak says so.
    """

    def __init__(self, delay: float = 0.05, present: bool = True):
        self.delay = delay
        self._report = DeviceReport(
            present=present, backend="xpu", name="Intel(R) Arc(TM) B580 Graphics",
            total_memory_mb=12216,
        )
        self.calls = 0
        self.live = 0
        self.peak = 0
        self._lock = threading.Lock()

    def report(self) -> DeviceReport:
        with self._lock:
            self.calls += 1
            self.live += 1
            self.peak = max(self.peak, self.live)
        time.sleep(self.delay)
        with self._lock:
            self.live -= 1
        return self._report


class Clock:
    """Monotonic time, advanced by hand. No sleeping in a TTL test."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def cached(inner, clock, busy=None, ttl=None, floor=None):
    return CachedDeviceProbe(
        inner=inner,
        ttl=ttl if ttl is not None else READINESS_CACHE_SECONDS,
        refresh_floor=floor if floor is not None else DEVICE_REFRESH_MIN_SECONDS,
        is_busy=busy or (lambda: False),
        now=clock,
    )


# ==========================================================================
print("-- eight concurrent callers cause one probe --")

inner = CountingProbe()
clock = Clock()
probe = cached(inner, clock)

threads = [threading.Thread(target=probe.report) for _ in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()

check(inner.calls == 1,
      f"8 concurrent report() calls run the probe once ({inner.calls})")
check(inner.peak == 1,
      f"and never more than one at a time ({inner.peak}) -- a lock that "
      f"only serialises *after* the probe starts would still show 8")
answers = [probe.report() for _ in range(8)]
check(len({a.name for a in answers}) == 1 and answers[0].present,
      f"and every caller gets the same answer ({answers[0].name!r})")
check(probe.probes_run == 1,
      f"the wrapper counts one probe for the whole burst ({probe.probes_run})")

# The same for the list, which is a different question about the same card.
inner2 = CountingProbe()
clock2 = Clock()
probe2 = cached(inner2, clock2)
threads = [threading.Thread(target=probe2.devices) for _ in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
check(probe2.probes_run == 1,
      f"and 8 concurrent devices() calls also run one enumeration "
      f"({probe2.probes_run}) -- the cache is on the port, so the newer "
      f"endpoint is covered too")

# ==========================================================================
print("\n-- the TTL, on an injected clock --")

inner3 = CountingProbe()
clock3 = Clock()
probe3 = cached(inner3, clock3)
probe3.report()
probe3.report()
probe3.report()
check(inner3.calls == 1,
      f"three sequential calls inside the TTL run one probe ({inner3.calls})")

clock3.advance(READINESS_CACHE_SECONDS - 1)
probe3.report()
check(inner3.calls == 1,
      f"still one, one second before expiry ({inner3.calls})")

clock3.advance(2)  # now past READINESS_CACHE_SECONDS
probe3.report()
check(inner3.calls == 2,
      f"and a second once the TTL has passed ({inner3.calls})")

# Exactly at the boundary. `<` not `<=`, so the TTL is a TTL.
inner4 = CountingProbe()
clock4 = Clock()
probe4 = cached(inner4, clock4)
probe4.report()
clock4.advance(READINESS_CACHE_SECONDS)
probe4.report()
check(inner4.calls == 2,
      f"the boundary is not inclusive ({inner4.calls}) -- a cache that "
      f"refreshes a hair early still costs an import per page load")

# ==========================================================================
print("\n-- refresh bypasses the cache, but is rate limited --")

inner5 = CountingProbe()
clock5 = Clock()
probe5 = cached(inner5, clock5)
probe5.report()
check(probe5.invalidate() is True,
      "refresh=true re-probes when the floor allows")
check(inner5.calls == 2, f"so the probe ran again ({inner5.calls})")

check(probe5.invalidate() is False,
      "a second refresh inside the floor is refused")
check(inner5.calls == 2,
      f"and runs nothing -- a Re-check button clicked three times must not "
      f"become the fan-out it replaced ({inner5.calls})")

clock5.advance(DEVICE_REFRESH_MIN_SECONDS + 0.1)
check(probe5.invalidate() is True,
      "and works again once the floor has passed")

# The floor and the TTL are different bounds, and conflating them would make
# the button either useless or unlimited.
check(READINESS_CACHE_SECONDS != DEVICE_REFRESH_MIN_SECONDS,
      f"the TTL ({READINESS_CACHE_SECONDS}) and the refresh floor "
      f"({DEVICE_REFRESH_MIN_SECONDS}) are separate bounds")

# ==========================================================================
print("\n-- a running training run is never disturbed --")

inner6 = CountingProbe()
clock6 = Clock()
probe6 = cached(inner6, clock6, busy=lambda: True)
report = probe6.report()
check(inner6.calls == 0,
      f"no probe runs while a run is active ({inner6.calls}) -- starting one "
      f"would take VRAM from the run to answer a wizard question")
check(not report.present,
      f"and the answer is 'not usable', not a fabricated success "
      f"({report.present})")
check(report.reason and "training run" in report.reason,
      f"with a reason that says a run is active ({report.reason!r})")
check("not a missing card" in (report.reason or ""),
      "and that the card was deliberately not asked, which is a different "
      "fact from the card being absent")

# The list endpoint has the same duty.
rows, reason = probe6.devices_with_reason()
check(rows == () and reason and "training run" in reason,
      f"and the list endpoint declines the same way ({reason!r})")

# With something cached, a busy machine reuses it rather than re-probing --
# the answer it already has is better than a refusal.
inner7 = CountingProbe()
clock7 = Clock()
state = {"busy": False}
probe7 = cached(inner7, clock7, busy=lambda: state["busy"])
probe7.report()
state["busy"] = True
clock7.advance(READINESS_CACHE_SECONDS + 1)  # TTL has expired
report7 = probe7.report()
check(inner7.calls == 1,
      f"a busy machine with a stale cache does not re-probe either "
      f"({inner7.calls})")

# And once the run is over, the next probe happens.
state["busy"] = False
probe7.report()
check(inner7.calls == 2,
      f"and probes again once the run is over ({inner7.calls})")

# ==========================================================================
print("\n-- the wrapper is transparent --")

inner8 = CountingProbe()
clock8 = Clock()
probe8 = cached(inner8, clock8)
check(probe8.enumerate_all is inner8.enumerate_all,
      "enumerate_all is the inner probe's, not a guess")
check(probe8.backend == "xpu", f"and the backend is forwarded ({probe8.backend})")
check(probe8.devices()[0].name == "Intel(R) Arc(TM) B580 Graphics",
      "and devices() returns what the inner probe found, uncached-by-accident")

# A probe that finds nothing must cache nothing-true: a second call inside
# the TTL reports absent without asking again, which is what lets the short
# circuit skip the device entirely.
inner9 = CountingProbe(present=False)
clock9 = Clock()
probe9 = cached(inner9, clock9)
first = probe9.report()
second = probe9.report()
check(first.present is False and second.present is False
      and inner9.calls == 1,
      f"an absent device is cached as absent, not re-asked ({inner9.calls})")

finish()