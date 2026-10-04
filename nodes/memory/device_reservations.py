"""DeviceReservations -- what has already claimed a device's VRAM.

The mechanism underneath both ways a run can hold memory against others on the
same card. **Off by default; nothing here is wired into a trainer yet** --
see the module's own closing note on what is and is not built.

The problem this exists for, and it is not the one a per-run budget solves.
`ResourceBudget` (nodes/resource_budget.py) bounds *one* run against a number
that run was given, and it has no idea anyone else is on the card. So two
graphs on one device each believe they are the only tenant, each stay under
their own ceiling, and together they overrun it -- which is the failure the
supervisor's own comments keep circling. A ceiling is not a claim on shared
memory; a claim has to be visible to the other claimants.

Two ways to hold a claim, both the user's call:

* **Observed** -- reserve the highest peak measured so far, plus a pillow,
  and raise it as the run reveals a higher one. Needs no prediction: the
  number is what the allocator actually did. B580 measurements that make it
  concrete: a LoRA run's peak is 7,666 MB at batch 2 and 8,954 MB at batch 4,
  so an observed ceiling is *learned* within a few steps rather than guessed
  before the first one. Its weakness is exactly its first steps -- before a
  peak exists there is nothing to reserve, and that is where an overrun
  happens.
* **Stated** -- the caller names a number and it is held for the whole run,
  and the run fails if it needs more. The estimate is the caller's, typically
  a previous run's reported peak. This is the mode that is safe on step 0,
  and the price is that it fails a run that would have fit.

A caller that can state neither is **unknown**, not zero. Zero is a claim
that nothing is needed, and an unchecked claim is how two graphs end up
sharing a card believing neither of them is there -- the same argument
`check_comfy_conflicts` makes for installed-but-undeclared packages, where
unknown gets a strict pin rather than being read as safe.

Deliberately not a graph object and not a global setting. It is per *device*,
because that is the resource being divided, and a global one could not
distinguish two cards.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

#: Extra VRAM held above an observed peak, in MB.
#:
#: Small on purpose. The point of the observed mode is that the number is
#: real, so the pillow only has to cover the difference between a peak and
#: the next step's peak -- allocator jitter and a fragment or two -- not the
#: uncertainty of an estimate. Measured on the B580, the run-to-run spread of
#: a steady-state peak is a few MB (reserved drift 0-14 MB across runs), so
#: ~150 MB is generous by an order of magnitude and still small next to the
#: 7-9 GB peaks it sits on. A larger pillow quietly turns this into the
#: stated mode with extra steps.
OBSERVED_PILLOW_MB = 150.0


@dataclass(frozen=True, slots=True)
class Reservation:
    """One holder's claim on a device.

    `observed_mb` is what has actually been seen; `stated_mb` is what the
    caller promised. `held_mb` is the larger of the two, because a claim is a
    claim: a caller who stated 10 GB and has been observed at 2 GB still
    holds 10, since the whole content of stating is that the number is
    available *before* it is needed.

    `observed_mb` is None until something has been measured, which is the
    window the observed mode is weakest in and the reason it is not the only
    mode.
    """

    owner: str
    stated_mb: float | None = None
    observed_mb: float | None = None

    @property
    def held_mb(self) -> float:
        parts = [p for p in (self.stated_mb, self.observed_mb) if p is not None]
        return max(parts) if parts else 0.0

    @property
    def is_unknown(self) -> bool:
        """True when this holder has claimed nothing and been seen doing nothing.

        Not an error by itself -- a graph that has not run yet is legitimately
        unmeasured -- but a caller deciding whether a device can take on more
        work has to be able to tell "needs nothing" from "not known yet", and
        those are not the same answer.
        """
        return self.stated_mb is None and self.observed_mb is None


@dataclass
class DeviceReservations:
    """Every claim on one device, and what is left.

    Not a singleton: the device is the scope, so a caller constructs one per
    device and passes it where it is needed, the same no-globals posture the
    rest of this codebase uses (root README, Goals, goal 3).
    """

    device: str
    total_mb: float | None = None
    _by_owner: dict[str, Reservation] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def claim(self, owner: str, stated_mb: float | None = None) -> Reservation:
        """Record a holder's claim. Re-claiming by the same owner replaces it.

        Replace rather than accumulate, because a holder that re-claims is
        stating a new number for the same run -- a ratcheting observed peak,
        or a corrected estimate -- and summing those would let a single
        graph's own revisions inflate the device total.
        """
        with self._lock:
            previous = self._by_owner.get(owner)
            observed = previous.observed_mb if previous else None
            reservation = Reservation(owner=owner, stated_mb=stated_mb,
                                      observed_mb=observed)
            self._by_owner[owner] = reservation
            return reservation

    def observe(self, owner: str, peak_mb: float) -> Reservation:
        """Raise a holder's observed peak, never lower it.

        Monotonic on purpose. A peak is the high-water mark of what the
        allocator did, so a later smaller number is not evidence that less is
        needed -- it is a step that happened not to be the worst one. Letting
        it fall would let a reservation shrink while the run that holds it
        keeps a higher high-water mark, which is precisely the silent
        overshoot this exists to prevent.
        """
        with self._lock:
            previous = self._by_owner.get(owner)
            observed = peak_mb
            if previous is not None and previous.observed_mb is not None:
                observed = max(previous.observed_mb, peak_mb)
            stated = previous.stated_mb if previous else None
            reservation = Reservation(owner=owner, stated_mb=stated,
                                      observed_mb=observed)
            self._by_owner[owner] = reservation
            return reservation

    def release(self, owner: str) -> None:
        """Drop a holder's claim. Idempotent, so a failed start can clean up."""
        with self._lock:
            self._by_owner.pop(owner, None)

    def held_by(self, owner: str) -> float:
        """How much one holder claims. 0 for a holder that does not exist.

        Separate from `held_mb(exclude=...)` because that one answers
        "everyone *else*", and the two are asked constantly and confused
        constantly -- `held_mb(owner)` returning 0 reads exactly like a bug
        when what it means is "0 for everyone else, which is right".
        """
        with self._lock:
            reservation = self._by_owner.get(owner)
            return reservation.held_mb if reservation else 0.0

    def held_mb(self, exclude: str | None = None) -> float:
        """Total claimed, optionally ignoring one holder's own claim.

        The `exclude` is what makes this usable by the thing doing the asking:
        a run checking whether it fits must not count itself.
        """
        with self._lock:
            return sum(r.held_mb for o, r in self._by_owner.items()
                       if o != exclude)

    def unreserved_mb(self, exclude: str | None = None) -> float | None:
        """What is left on the device for `exclude` to take.

        None when the device's size is unknown -- which is not the same as
        "everything is free", and is the same unknown-the-caller-declared
        problem one level down.
        """
        if self.total_mb is None:
            return None
        return self.total_mb - self.held_mb(exclude)

    def fits(self, needed_mb: float, exclude: str | None = None) -> bool | None:
        """Whether `needed_mb` fits alongside everyone else.

        None when the device size is unknown -- an unanswerable question
        reported as unanswerable rather than as a permissive True, for the
        reason `check_comfy_conflicts` treats installed-but-undeclared as
        unknown rather than safe.
        """
        free = self.unreserved_mb(exclude)
        if free is None:
            return None
        return needed_mb <= free

    def holders(self) -> dict[str, Reservation]:
        with self._lock:
            return dict(self._by_owner)


class ReservationRefused(RuntimeError):
    """A run cannot be admitted against what the device has left.

    A refusal, not a warning. Under a stated reservation the whole content of
    stating is that the number is available before it is needed, so "it did
    not fit" has to stop the run at that point rather than be discovered as an
    out-of-memory error later -- which is what `check_comfy_conflicts` means
    by refusing what it cannot read, and what its docstring calls the
    alternative: treating unknown as safe is how a soft install quietly
    becomes the hard one.
    """


def admit(reservations: DeviceReservations, owner: str, *,
          stated_mb: float | None = None) -> Reservation:
    """Reserve for `owner`, or refuse.

    `stated_mb` is the number the run will hold for its whole life, and it is
    **required**: None means nobody knows what this run needs, and admitting
    it is admitting a run whose peak is nominally nothing. That was a real hole
    here, not a hypothetical -- an earlier version treated None as "observed
    mode, nothing to check yet" and a demonstration run duly got admitted with
    a 0 MB claim on its first ever configuration.

    **The observed and stated modes are not two mechanisms.** They are this one
    mechanism with the number coming from different places: a *stated*
    reservation is written by a person, an *observed* one is read out of
    `PeakRecord` (nodes/memory/peak_record.py) from what a previous run of the
    same configuration actually peaked at, plus a small pillow. Both are
    checked the same way, because the whole point of observing is that the
    observation happened *before* this run -- a peak learned while the run is
    already going can only be enforced by killing it, which is the worst place
    for that failure and the reason the record is persistent rather than
    in-memory.

    The asymmetry that remains is only in the first ever run of a
    configuration, where the record has nothing and admission fails. That is
    the cost of not guessing, and it is one refused run rather than an
    out-of-memory error at step 900.
    """
    if stated_mb is None:
        raise ReservationRefused(
            f"{owner} cannot be admitted: nothing is known about how much VRAM "
            f"it needs. Supply a stated reservation, or measure this "
            f"configuration once so PeakRecord can supply one -- an "
            f"unmeasured run is refused, not admitted with a zero claim."
        )
    fits = reservations.fits(stated_mb, exclude=owner)
    if fits is None:
        # Unknown device size: record the claim and say so by not pretending
        # it was checked. The caller gets the reservation and can see that
        # `unreserved_mb()` is None.
        return reservations.claim(owner, stated_mb=stated_mb)
    if not fits:
        raise ReservationRefused(
            f"{owner} asked for {stated_mb:.0f} MB but only "
            f"{reservations.unreserved_mb(exclude=owner):.0f} MB is unreserved "
            f"on {reservations.device} (held by "
            f"{', '.join(sorted(reservations.holders())) or 'nobody'}). "
            f"Lower the reservation, or free the other run's claim first."
        )
    return reservations.claim(owner, stated_mb=stated_mb)


def observe(reservations: DeviceReservations, owner: str,
            peak_mb: float) -> Reservation:
    """Raise `owner`'s claim to a peak just measured, plus the pillow.

    The observed mode's whole mechanism, and it is one line because the two
    properties it needs already live elsewhere: `DeviceReservations.observe`
    is monotonic, so the claim can only ratchet up, and `Reservation.held_mb`
    is the max of stated and observed, so a run that also stated a number
    keeps the larger one without this function knowing about it.

    No `stated_mb` parameter on purpose. A caller that stated a number already
    holds it; adding the peak here would either be redundant or would quietly
    lower a deliberate reservation to a measured one, and both are worse than
    doing nothing.
    """
    return reservations.observe(owner, peak_mb + OBSERVED_PILLOW_MB)

