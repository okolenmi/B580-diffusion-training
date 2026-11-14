"""Device reservations -- what has already claimed a device's VRAM.

The mechanism underneath both ways a run can hold memory against others on the
same card. **Off by default and not wired into any trainer yet** -- this is the
registry and the two policies, deliberately separable from the thing that would
consume it, because the registry is the part with no opinions and the wiring is
the part with several.

The problem this exists for is not the one a per-run budget solves.
`ResourceBudget` (nodes/resource_budget.py) bounds *one* run against a number
that run was given, and has no idea anyone else is on the card. So two graphs on
one device each believe they are the only tenant, each stay under their own
ceiling, and together they overrun it -- which is the failure the supervisor's
own comments keep circling. A ceiling is not a claim on shared memory; a claim
has to be visible to the other claimants.

Two ways to hold a claim, both the user's call:

* **Observed** -- reserve the highest peak measured so far plus a small pillow,
  raising it as the run reveals a higher one. Needs no prediction: the number is
  what the allocator actually did. B580 measurements that make it concrete: a
  LoRA run's peak is 7,666 MB at batch 2 and 8,954 MB at batch 4, so an observed
  ceiling is *learned* within a few steps rather than guessed before the first
  one. Its weakness is exactly its first steps -- before a peak exists there is
  nothing to reserve, and that is where an overrun happens.
* **Stated** -- the caller names a number, it is held for the whole run, and the
  run is refused if it cannot have it. The estimate is the caller's, typically a
  previous run's reported peak. This is the mode that is safe on step 0, and the
  price is that it refuses a run that would have fit.

A caller that can state neither is **unknown**, not zero. Zero is a claim that
nothing is needed, and an unchecked claim is how two graphs end up sharing a card
believing neither of them is there -- the same argument
`check_comfy_conflicts` makes for installed-but-undeclared packages, where
unknown gets a strict pin rather than being read as safe.

Per *device*, not per graph and not a global setting: the device is the resource
being divided, and a global one could not tell two cards apart.

Not a singleton -- a caller constructs one per device and passes it where it is
needed, the same no-globals posture the rest of this codebase uses (root README,
Goals, goal 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.memory.device_reservations import (
    OBSERVED_PILLOW_MB,
    DeviceReservations,
    Reservation,
    ReservationRefused,
    admit,
    observe,
)

CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)
    print(f"    PASS: {message}")


# What DeviceContext.total_memory_mb() reports for the B580, read rather
# than converted: 11.93 GiB is 12,216 MB, and using the GiB figure here would
# put every expectation 2.4% low without looking wrong.
CARD_MB = 12_216.0


def main() -> None:
    print("[a claim is the larger of what was stated and what was seen]")
    r = Reservation(owner="a")
    check(r.held_mb == 0.0 and r.is_unknown,
          "a holder that has claimed and been seen nothing is unknown, not "
          "free -- zero would be a claim that nothing is needed")
    r = Reservation(owner="a", stated_mb=10_000.0)
    check(r.held_mb == 10_000.0 and not r.is_unknown,
          "a stated claim is held even with nothing observed")
    r = Reservation(owner="a", stated_mb=10_000.0, observed_mb=2_000.0)
    check(r.held_mb == 10_000.0,
          "and it is not lowered by a peak below it: stating means the "
          "number is available before it is needed")
    r = Reservation(owner="a", observed_mb=7_666.0)
    check(r.held_mb == 7_666.0, "observed alone holds the observed peak")

    print("\n[observed mode: the peak ratchets up and never back down]")
    d = DeviceReservations(device="xpu", total_mb=CARD_MB)
    observe(d, "run-a", 7_666.0)
    check(d.held_by("run-a") == 7_666.0 + OBSERVED_PILLOW_MB,
          f"a measured peak plus the {OBSERVED_PILLOW_MB:.0f} MB pillow "
          f"(got {d.held_mb('run-a'):.0f})")
    observe(d, "run-a", 3_000.0)
    check(d.held_by("run-a") == 7_666.0 + OBSERVED_PILLOW_MB,
          "a smaller later peak does not shrink the claim -- a peak is a "
          "high-water mark, and letting it fall is the silent overshoot this "
          "exists to prevent")
    observe(d, "run-a", 8_954.0)
    check(d.held_by("run-a") == 8_954.0 + OBSERVED_PILLOW_MB,
          "a higher one raises it, so batch 4's real peak is learnable")

    print("\n[two runs on one card see each other]")
    observe(d, "run-a", 8_954.0)
    observe(d, "run-b", 7_666.0)
    total = d.held_mb()
    check(total == (8_954.0 + OBSERVED_PILLOW_MB) + (7_666.0 + OBSERVED_PILLOW_MB),
          f"both claims are counted (got {total:.0f} MB)")
    check(d.held_by("run-a") == 8_954.0 + OBSERVED_PILLOW_MB,
          "and a run checking whether it fits does not count itself")
    check(d.unreserved_mb(exclude="run-c") == CARD_MB - total,
          f"what is left is the card minus both claims "
          f"(got {d.unreserved_mb(exclude='run-c'):.0f} MB of {CARD_MB:.0f})")

    print("\n[stated mode: a named number, or a refusal]")
    d2 = DeviceReservations(device="xpu", total_mb=CARD_MB)
    admit(d2, "run-a", stated_mb=8_000.0)
    check(d2.fits(3_000.0, exclude="run-b") is True,
          "a 3,000 MB run fits alongside an 8,000 MB claim")
    check(d2.fits(5_000.0, exclude="run-b") is False,
          "and a 5,000 MB one does not (12,216 - 8,000 = 4,216 free)")
    try:
        admit(d2, "run-b", stated_mb=5_000.0)
        check(False, "an admission that does not fit must raise")
    except ReservationRefused as exc:
        check("4216 MB is unreserved" in str(exc),
              f"the refusal names what was available and who holds it "
              f"(got {str(exc)[:56]!r})")
    check(d2.holders().get("run-b") is None,
          "and a refused run holds no claim, so a retry after freeing room "
          "is not blocked by the attempt")

    print("\n[an unmeasured run is refused, not admitted with a zero claim]")
    d3 = DeviceReservations(device="xpu", total_mb=CARD_MB)
    try:
        admit(d3, "run-a")
        check(False, "admitting a run with no stated reservation must raise")
    except ReservationRefused as exc:
        check("nothing is known about how much VRAM it needs" in str(exc),
              f"a run with no reservation is refused (got {str(exc)[:52]!r})")
    check(d3.holders() == {},
          "and holds nothing afterwards -- the hole this closes was a "
          "demonstration run being admitted with a 0 MB claim on its first "
          "ever configuration")

    print("\n[unknown device size is unanswerable, not permissive]")
    d4 = DeviceReservations(device="xpu", total_mb=None)
    check(d4.unreserved_mb() is None and d4.fits(1.0) is None,
          "an unknown-size device answers 'cannot tell', not 'yes' -- the "
          "check_comfy_conflicts rule applied to memory")
    reservation = admit(d4, "run-a", stated_mb=5_000.0)
    check(reservation.held_mb == 5_000.0,
          "and the claim is still recorded, so the caller can see it was not "
          "verified")

    print("\n[re-claiming and releasing]")
    d5 = DeviceReservations(device="xpu", total_mb=CARD_MB)
    admit(d5, "run-a", stated_mb=8_000.0)
    observe(d5, "run-a", 2_000.0)
    admit(d5, "run-a", stated_mb=9_000.0)
    check(d5.held_mb() == 9_000.0,
          "re-claiming replaces rather than accumulates, so a graph's own "
          "corrections cannot inflate the device total")
    check(d5.held_mb() == 9_000.0,
          "and the observed high-water mark survives the restatement")
    d5.release("run-a")
    check(d5.held_mb() == 0.0 and d5.holders() == {},
          "releasing frees the claim")
    d5.release("run-a")
    check(d5.held_mb() == 0.0, "and releasing twice is a no-op, so a failed "
          "start can clean up unconditionally")

    print()
    print("=" * 60)
    print(f"SMOKE TEST: ALL {CHECKS} CHECKS PASSED")


if __name__ == "__main__":
    main()
