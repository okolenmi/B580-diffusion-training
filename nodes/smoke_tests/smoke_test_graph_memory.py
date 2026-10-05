"""Checks nodes/memory/graph_memory.py's GraphMemory (MEM-05 #1, #2, #3).

The child's memory picture: two numbers, their unknowns named, and the
validation that keeps a NaN or a negative out of everything downstream.
The NaN case is the load-bearing one -- a NaN grant would make every
``free < grant`` comparison False, so the physical check that follows
this constructor would pass on garbage instead of refusing it.

Then the check itself (#2): what the pre-load comparison against the
real card decides in each of its three states -- ok, shortfall, and the
unknowns that must continue with a warning rather than pass on a zero.

And the backstop (#3): the allocator's own cap, ``budget / total``,
applied through the device when the installed torch build has the
attribute -- and the honest case name when it does not, so the
telemetry can say which happened.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.infrastructure.graph_task_worker import apply_allocator_backstop
from nodes.core import ExecutionContext
from nodes.memory.graph_memory import (
    CHECK_OK,
    CHECK_SHORTFALL,
    CHECK_UNKNOWN,
    GraphMemory,
)

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def check_stores_both_numbers():
    print("[both numbers are stored, and an int arrives as a float]")
    memory = GraphMemory(grant_mb=6300.0, budget_mb=6000)
    record(memory.grant_mb == 6300.0, "grant_mb stored", detail=repr(memory.grant_mb))
    record(memory.budget_mb == 6000.0, "budget_mb stored", detail=repr(memory.budget_mb))


def check_absent_is_unknown_not_zero():
    print("[no numbers supplied: explicit unknown, never an invented zero]")
    memory = GraphMemory()
    record(memory.grant_mb is None, "grant_mb is None (unknown)", detail=repr(memory.grant_mb))
    record(memory.budget_mb is None, "budget_mb is None (unknown)", detail=repr(memory.budget_mb))


def _refuses(name: str, value, must_mention: str) -> bool:
    """Assert one bad value raises ValueError naming the field."""
    try:
        GraphMemory(**{name: value})
    except ValueError as exc:
        return must_mention in str(exc)
    return False


def check_rejects_bad_numbers():
    print("[what a device size cannot be: NaN, inf, overflow, negative, "
          "zero, bool, string -- each refused with the field named]")
    cases = [
        ("grant_mb", float("nan"), "grant_mb"),
        ("grant_mb", float("inf"), "grant_mb"),
        ("grant_mb", 10**400, "grant_mb"),
        ("grant_mb", -5.0, "grant_mb"),
        ("budget_mb", float("nan"), "budget_mb"),
        ("budget_mb", 0.0, "budget_mb"),
        ("budget_mb", True, "budget_mb"),
        ("budget_mb", "6000", "budget_mb"),
    ]
    for name, value, mention in cases:
        ok = _refuses(name, value, mention)
        record(ok, f"refuses {name}={value!r}", detail="no ValueError" if not ok else "")


def check_nan_cannot_smuggle_through():
    print("[the comparison-poisoning case, stated directly: a NaN grant "
          "is refused, so 'free < grant' can never see one]")
    refused = _refuses("grant_mb", math.nan, "grant_mb")
    record(refused, "NaN grant refused at construction")


def check_exec_context_carries_memory():
    print("[ExecutionContext carries what it is given, defaults to None]")
    memory = GraphMemory(grant_mb=6300.0, budget_mb=6000.0)
    with_memory = ExecutionContext(memory=memory)
    record(with_memory.memory is memory, "memory survives construction",
           detail=repr(with_memory.memory))
    record(ExecutionContext().memory is None,
           "no memory passed: None, like monitor_bus",
           detail=repr(ExecutionContext().memory))


#: Distinguishes "answer from the arithmetic" (default) from a backend
#: that explicitly answers None.
_NO_OVERRIDE = object()


class FakeDevice:
    """A driver's-eye view of a card, with room foreign users can take
    while the test watches -- the gap MEM-05 #2 exists to close.

    Free is what the driver would report: everything, minus what the
    allocator holds, minus this process's other overhead, minus whatever
    a foreign process (ComfyUI, the desktop) has claimed and admission
    knows nothing about.
    """

    def __init__(self, *, capacity_mb, allocation_mb=0.0, overhead_mb=0.0,
                 free_override=_NO_OVERRIDE, total_override=_NO_OVERRIDE):
        self.capacity_mb = capacity_mb
        self.allocation_mb = allocation_mb
        self.overhead_mb = overhead_mb
        self.foreign_mb = 0.0
        self._override = free_override
        self._total_override = total_override
        self.fraction = None

    def appears(self, foreign_mb):
        """A foreign process claims room the ledger cannot see."""
        self.foreign_mb = foreign_mb

    def free_memory_mb(self):
        if self._override is not _NO_OVERRIDE:
            return self._override
        return (self.capacity_mb - self.allocation_mb
                - self.overhead_mb - self.foreign_mb)

    def total_memory_mb(self):
        if self._total_override is not _NO_OVERRIDE:
            return self._total_override
        return self.capacity_mb

    def set_per_process_memory_fraction(self, fraction):
        """The allocator's cap, recorded so the test can read it back."""
        self.fraction = fraction
        return True


class _BareDevice:
    """A device with no allocator-fraction notion: the method is simply
    absent, which is how a CPU backend or an older torch build looks
    to the backstop."""

    def __init__(self, capacity_mb):
        self.capacity_mb = capacity_mb

    def total_memory_mb(self):
        return self.capacity_mb


class _RefusingDevice(_BareDevice):
    """A build where the context has the method but the torch build
    answers False: the cap cannot be applied."""

    def set_per_process_memory_fraction(self, fraction):
        return False


def check_physical_ok_when_the_card_has_room():
    print("[physical check: room for the grant -> ok, not refused]")
    device = FakeDevice(capacity_mb=12216.0, allocation_mb=500.0,
                        overhead_mb=400.0)
    result = GraphMemory(grant_mb=6300.0).physical_check(device)
    record(result.status == CHECK_OK, "status ok",
           detail=f"{result.status} ({result.reason})")
    record(not result.refused, "not refused")
    record(result.free_mb is not None and result.free_mb == 11316.0,
           "the free number is carried", detail=repr(result.free_mb))


def check_physical_refuses_when_foreign_holds_the_room():
    print("[physical check: foreign users the ledger cannot see take the "
          "room -> shortfall, and the message quotes both numbers]")
    device = FakeDevice(capacity_mb=7500.0, allocation_mb=500.0,
                        overhead_mb=500.0)
    device.appears(foreign_mb=400.0)  # free: 7500 - 500 - 500 - 400 = 6100
    result = GraphMemory(grant_mb=6300.0).physical_check(device)
    record(result.status == CHECK_SHORTFALL, "status shortfall",
           detail=f"{result.status} ({result.reason})")
    record(result.refused, "refused is True")
    message = result.refusal_message()
    record("6300" in message, "message quotes the grant", detail=message)
    record("6100" in message, "message quotes what is free", detail=message)
    record("foreign" in message, "message names the party outside the ledger",
           detail=message)


def check_physical_boundary_free_equals_grant_passes():
    print("[physical check: free == grant is the boundary -- exactly "
          "enough is enough]")
    device = FakeDevice(capacity_mb=6300.0)
    result = GraphMemory(grant_mb=6300.0).physical_check(device)
    record(result.status == CHECK_OK, "exactly enough: ok",
           detail=f"{result.status} ({result.reason})")


def check_physical_unknowns_never_pass_or_refuse_silently():
    print("[physical check: no grant, no answer, or a garbage answer -> "
          "unknown with a reason -- never an ok that pretended]")
    # No grant: the run's numbers never travelled (argv pairs absent).
    unanswered = GraphMemory().physical_check(
        FakeDevice(capacity_mb=12216.0))
    record(unanswered.status == CHECK_UNKNOWN, "no grant: unknown",
           detail=f"{unanswered.status} ({unanswered.reason})")
    record(unanswered.reason != "", "the unknown carries a reason",
           detail=repr(unanswered.reason))
    # Backend with no such notion (CPU) or a failed query.
    silent = GraphMemory(grant_mb=6300.0).physical_check(
        FakeDevice(capacity_mb=12216.0, free_override=None))
    record(silent.status == CHECK_UNKNOWN, "free is None: unknown",
           detail=f"{silent.status} ({silent.reason})")
    # A NaN reading would make free < grant False, i.e. "ok" on garbage.
    garbage = GraphMemory(grant_mb=6300.0).physical_check(
        FakeDevice(capacity_mb=12216.0, free_override=float("nan")))
    record(garbage.status == CHECK_UNKNOWN, "NaN free: unknown, not ok",
           detail=f"{garbage.status} ({garbage.reason})")
    for name, result in (("no grant", unanswered), ("free None", silent),
                         ("NaN free", garbage)):
        record(not result.refused,
               f"{name}: not refused (refusal needs both numbers)")


def check_backstop_enforced_when_the_build_has_the_attribute():
    print("[backstop: budget/total is set on the device, and the case "
          "says so]")
    device = FakeDevice(capacity_mb=12216.0)
    case = apply_allocator_backstop(GraphMemory(budget_mb=6000.0), device)
    record(case == "enforced", "case is enforced", detail=case)
    record(
        abs(device.fraction - 6000.0 / 12216.0) < 1e-9,
        "the fraction is budget/total",
        detail=repr(device.fraction),
    )


def check_backstop_unavailable_without_the_attribute():
    print("[backstop: no set_per_process_memory_fraction on this build "
          "-> unavailable, not a refusal]")
    case = apply_allocator_backstop(
        GraphMemory(budget_mb=6000.0), _BareDevice(12216.0))
    record(case == "unavailable", "case is unavailable", detail=case)
    case = apply_allocator_backstop(
        GraphMemory(budget_mb=6000.0), _RefusingDevice(12216.0))
    record(case == "unavailable",
           "a False from the device is the same case", detail=case)


def check_backstop_unknowns_name_the_missing_number():
    print("[backstop: no budget or no total -> the case names which]")
    device = FakeDevice(capacity_mb=12216.0)
    case = apply_allocator_backstop(GraphMemory(grant_mb=6300.0), device)
    record(case == "unknown_budget", "no budget stated: unknown_budget",
           detail=case)
    case = apply_allocator_backstop(
        GraphMemory(budget_mb=6000.0),
        FakeDevice(capacity_mb=12216.0, total_override=None),
    )
    record(case == "unknown_total", "no total to divide by: unknown_total",
           detail=case)


def check_backstop_clamps_a_budget_above_the_total():
    print("[backstop: a budget above the total is capped at 1.0, not "
          "passed through as a license to over-subscribe]")
    device = FakeDevice(capacity_mb=12216.0)
    case = apply_allocator_backstop(GraphMemory(budget_mb=20000.0), device)
    record(case == "enforced", "still enforced (capped)", detail=case)
    record(device.fraction == 1.0, "the fraction is capped at 1.0",
           detail=repr(device.fraction))


def main():
    check_stores_both_numbers()
    check_absent_is_unknown_not_zero()
    check_rejects_bad_numbers()
    check_nan_cannot_smuggle_through()
    check_exec_context_carries_memory()
    check_physical_ok_when_the_card_has_room()
    check_physical_refuses_when_foreign_holds_the_room()
    check_physical_boundary_free_equals_grant_passes()
    check_physical_unknowns_never_pass_or_refuse_silently()
    check_backstop_enforced_when_the_build_has_the_attribute()
    check_backstop_unavailable_without_the_attribute()
    check_backstop_unknowns_name_the_missing_number()
    check_backstop_clamps_a_budget_above_the_total()

    print()
    print("=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
