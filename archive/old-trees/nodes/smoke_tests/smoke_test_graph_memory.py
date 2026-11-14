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
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.infrastructure.graph_task_worker import apply_allocator_backstop
from nodes.core import ExecutionContext
from nodes.memory.graph_memory import (
    CHECK_OK,
    CHECK_SHORTFALL,
    CHECK_UNKNOWN,
    GraphMemory,
    MemoryRequestDenied,
)
from nodes.memory.handle import DeviceResident

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

    def memory_stats(self):
        """None: no allocator reading, so the handle's pressure loop is
        a cheap no-op and what is under test is the shared ordering."""
        return None

    def synchronize(self):
        pass


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


def check_memory_events_reach_the_listener():
    print("[memory events: a lease/eviction decision reaches the wired "
          "listener]")
    memory = GraphMemory(grant_mb=6300.0, budget_mb=6000.0)
    seen = []
    memory.on_memory_event = lambda event, mb, cost_ms: seen.append(
        (event, mb, cost_ms)
    )
    memory.report_memory_event("eviction", 1234.0, 56.7)
    record(seen == [("eviction", 1234.0, 56.7)],
           "the listener got the decision with MB and cost",
           detail=repr(seen))


def check_memory_events_without_a_listener_are_not_an_error():
    print("[memory events: no listener is a no-op, not an error]")
    memory = GraphMemory(grant_mb=6300.0, budget_mb=6000.0)
    memory.report_memory_event("lease_granted", 500.0, 0.0)
    record(memory.on_memory_event is None,
           "nothing was wired, nothing broke")


# -- MEM-06: the lease policy -----------------------------------------------


class _FakeResident(DeviceResident):
    """A resident whose size and reload cost are whatever the test says.

    ``reload_ms`` is really slept, so the cost GraphMemory measures
    around ``reload()`` is a real measurement rather than a scripted
    one -- the eviction policy orders by that measured number, and a
    test that faked it would be testing the fake.
    """

    def __init__(self, name: str, size_mb: float, reload_ms: float = 0.0):
        self.name = name
        self.size_mb = size_mb
        self.reload_ms = reload_ms
        self.offloaded = False
        self.offload_calls = 0
        self.reload_calls = 0

    def footprint_bytes(self) -> int:
        return 0 if self.offloaded else int(self.size_mb * (1024 ** 2))

    def offload(self) -> None:
        self.offloaded = True
        self.offload_calls += 1

    def reload(self, device=None) -> None:
        self.offloaded = False
        self.reload_calls += 1
        if self.reload_ms:
            time.sleep(self.reload_ms / 1000.0)

    def release(self) -> None:
        pass


def _memory_with(*residents, budget_mb):
    memory = GraphMemory(budget_mb=budget_mb)
    for resident, kwargs in residents:
        memory.register_resident(resident.name, resident, **kwargs)
    return memory


def check_lease_is_granted_when_there_is_room():
    print("[lease: room inside the budget is granted without evicting "
          "anyone]")
    resident = _FakeResident("model", 2000.0)
    memory = _memory_with((resident, {}), budget_mb=10000.0)
    with memory.request(3000.0, why="vae decode") as lease:
        record(lease.mb == 3000.0, "the lease holds what was asked",
               detail=repr(lease.mb))
        record(resident.offload_calls == 0,
               "nobody was evicted -- there was room",
               detail=repr(resident.offload_calls))
        record(memory.in_use_mb() == 5000.0,
               "in-use counts the resident and the live lease",
               detail=repr(memory.in_use_mb()))
    record(memory.in_use_mb() == 2000.0,
           "and the lease released on the way out",
           detail=repr(memory.in_use_mb()))


def check_lease_evicts_the_least_important_first():
    print("[lease: pressure evicts by priority -- least important "
          "first -- and pinned never]")
    model = _FakeResident("model", 2000.0)
    optimizer = _FakeResident("optimizer", 2000.0)
    pinned = _FakeResident("pinned", 1000.0)
    memory = _memory_with(
        (model, {"priority": 10}),        # important: goes last
        (optimizer, {"priority": 0}),     # least important: goes first
        (pinned, {"pinned": True}),
        budget_mb=6000.0,
    )
    with memory.request(2000.0, why="vae decode"):
        record(optimizer.offloaded, "the lowest-priority resident moved",
               detail=f"optimizer.offloaded={optimizer.offloaded}")
        record(not model.offloaded, "the important one did not")
        record(not pinned.offloaded, "and the pinned one never does")
        record(memory.in_use_mb() <= 6000.0,
               "a grant never leaves in-use above the budget",
               detail=repr(memory.in_use_mb()))


def check_the_handle_delegates_its_ordering_to_the_graph_memory():
    print("[the handle adapter: eviction order is GraphMemory's, measured "
          "once and shared]")
    from nodes.memory.control_handle import BudgetedResourceControlHandle
    from nodes.resource_budget import ResourceBudget

    cheap = _FakeResident("cheap", 800.0, reload_ms=2.0)
    dear = _FakeResident("dear", 800.0, reload_ms=40.0)
    handle = BudgetedResourceControlHandle(
        ResourceBudget(vram_budget_mb=1600.0, vram_reserve_mb=0.0),
        device="cpu", device_ctx=_BareDevice(12216.0),
    )
    handle.register("cheap", cheap, offloadable=True)
    handle.register("dear", dear, offloadable=True)
    memory = handle._memory
    record(isinstance(memory, GraphMemory),
           "the handle's own authority is a GraphMemory")

    # Move both through it, which times both real reloads -- the same
    # measurement the lease API's ordering uses.
    memory.move("cheap", offload=True)
    memory.move("dear", offload=True)
    memory.move("cheap", offload=False)
    memory.move("dear", offload=False)
    measured = memory._residents
    record(measured["cheap"].reload_cost_ms is not None
           and measured["dear"].reload_cost_ms is not None,
           "both residents have a measured reload cost now")
    record(measured["cheap"].reload_cost_ms < measured["dear"].reload_cost_ms,
           "and the cheap one really measured cheaper",
           detail=f"cheap={measured['cheap'].reload_cost_ms:.1f}ms "
           f"dear={measured['dear'].reload_cost_ms:.1f}ms")

    first = memory.next_to_evict()
    record(first is not None and first.name == "cheap",
           "so the handle's eviction picks the cheap one, by the policy "
           "order rather than registration order",
           detail=repr(first.name if first else None))

    # And the states map as documented: offloadable outranks
    # sacrificable, and "never" is not a candidate at all.
    pinned = _FakeResident("pinned", 800.0)
    handle.register("pinned", pinned)
    record(memory._residents["pinned"].pinned,
           "a resident registered with neither flag is pinned")
    sacrificable = _FakeResident("sac", 800.0)
    handle.register("sac", sacrificable, sacrificable=True)
    record(not memory._residents["sac"].pinned
           and memory._residents["sac"].priority
           > memory._residents["cheap"].priority,
           "a sacrificable one is evictable but sorts after an "
           "offloadable one")
    record(handle.release("cheap") is None,
           "release() still moves an offloadable resident on demand")


def check_a_denied_request_changes_nothing():
    print("[lease: a request that cannot be satisfied is denied, and "
          "everything it moved is put back]")
    a = _FakeResident("a", 2000.0)
    b = _FakeResident("b", 2000.0)
    memory = _memory_with((a, {}), (b, {}), budget_mb=4000.0)
    before = memory.in_use_mb()
    denied = None
    try:
        with memory.request(5000.0, why="vae decode"):
            record(False, "the block must not run")
    except MemoryRequestDenied as exc:
        denied = exc
    record(denied is not None, "an impossible request is refused")
    record(denied is not None and denied.needed_mb == 5000.0,
           "the refusal carries what was needed")
    record(denied is not None and denied.available_mb == 0.0,
           "and what was available",
           detail=repr(denied.available_mb if denied else None))
    record(denied is not None and denied.evictable == ["a", "b"],
           "and who was movable, by policy order",
           detail=repr(denied.evictable if denied else None))
    record(memory.in_use_mb() == before,
           "in-use is exactly as it was", detail=repr(memory.in_use_mb()))
    record(not a.offloaded and not b.offloaded,
           "every evicted resident was reloaded before the refusal")
    record(a.reload_calls == a.offload_calls
           and b.reload_calls == b.offload_calls,
           "each move was undone", detail=f"a={a.offload_calls}/"
           f"{a.reload_calls} b={b.offload_calls}/{b.reload_calls}")


def check_a_lease_restores_what_it_moved_even_on_an_exception():
    print("[lease: the block's exit restores what it moved, including "
          "when the body raises]")
    resident = _FakeResident("model", 3000.0)
    memory = _memory_with((resident, {}), budget_mb=3000.0)
    raised = False
    try:
        with memory.request(3000.0, why="vae decode"):
            record(resident.offloaded, "the resident made way for the lease")
            raise ValueError("the body blew up")
    except ValueError:
        raised = True
    record(raised, "the body's own exception still propagates")
    record(resident.reload_calls == 1,
           "and the resident was still put back", detail=repr(resident.reload_calls))
    record(not resident.offloaded and memory.in_use_mb() == 3000.0,
           "state restored", detail=repr(memory.in_use_mb()))


def check_concurrent_requests_serialise():
    print("[lease: two nodes asking at once are serialised, so together "
          "they cannot overgrant]")
    resident = _FakeResident("model", 3000.0)
    memory = _memory_with((resident, {}), budget_mb=5000.0)
    granted: list[str] = []
    denied: list[str] = []
    barrier = threading.Barrier(2)

    def ask(who: str) -> None:
        barrier.wait()
        try:
            with memory.request(3000.0, why=who):
                granted.append(who)
                # Observe in-use at the moment both could be live.
                time.sleep(0.02)
        except MemoryRequestDenied:
            denied.append(who)

    threads = [threading.Thread(target=ask, args=(who,))
               for who in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    record(len(granted) == 1,
           "exactly one of two 3000 MB requests into a 5000 MB budget is "
           "granted", detail=f"granted={granted} denied={denied}")
    record(len(denied) == 1,
           "the other is denied, not quietly overgranted",
           detail=f"granted={granted} denied={denied}")
    record(memory.in_use_mb() == 3000.0,
           "and the graph is whole again afterwards",
           detail=repr(memory.in_use_mb()))


def check_eviction_prefers_the_cheapest_resident_to_restore():
    print("[lease: once reload costs are measured, the cheapest to "
          "restore is given up first]")
    cheap = _FakeResident("cheap", 800.0, reload_ms=2.0)
    dear = _FakeResident("dear", 800.0, reload_ms=40.0)
    memory = _memory_with((cheap, {}), (dear, {}), budget_mb=1600.0)
    # Round one is spent measuring: a request big enough to move both
    # residents, so each one's real reload cost gets timed and stored.
    with memory.request(1600.0, why="measuring round"):
        pass
    record(cheap.reload_calls == 1 and dear.reload_calls == 1,
           "both residents were moved and measured once",
           detail=f"cheap={cheap.reload_calls} dear={dear.reload_calls}")
    # Round two needs room for exactly one of them. Whichever is
    # cheapest to restore is the one that should go, and the other
    # should be left alone.
    with memory.request(800.0, why="second"):
        record(cheap.offloaded,
               "the cheapest-to-restore resident is evicted first",
               detail=f"cheap.offload_calls={cheap.offload_calls}")
        record(not dear.offloaded,
               "the expensive one is kept", detail=f"dear="
               f"{dear.offload_calls}")


def check_an_unknown_budget_is_not_a_zero_budget():
    print("[lease: with no budget stated the request is granted and said, "
          "not refused]")
    resident = _FakeResident("model", 9000.0)
    memory = _memory_with((resident, {}), budget_mb=None)
    with memory.request(500.0, why="vae decode"):
        record(True, "granted without a ceiling to check against")
        record(resident.offload_calls == 0, "and nothing evicted")
    record(memory.in_use_mb() > 0.0,
           "in-use is still accounted", detail=repr(memory.in_use_mb()))


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
    check_memory_events_reach_the_listener()
    check_memory_events_without_a_listener_are_not_an_error()
    check_lease_is_granted_when_there_is_room()
    check_lease_evicts_the_least_important_first()
    check_a_denied_request_changes_nothing()
    check_a_lease_restores_what_it_moved_even_on_an_exception()
    check_concurrent_requests_serialise()
    check_eviction_prefers_the_cheapest_resident_to_restore()
    check_the_handle_delegates_its_ordering_to_the_graph_memory()
    check_an_unknown_budget_is_not_a_zero_budget()

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
