"""Tests for MemoryLedger: the server-side admission controller.

Level 1 of the two-level design: admission between processes. The server
owns it. The ledger holds no state of its own beyond the lock; on startup
it is rebuilt from rows of adopted/running children.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import logging
import math
import threading

from hypothesis import given, settings
from hypothesis import strategies as st

from backend.application.memory_ledger import (
    DEFAULT_FOREIGN_RESERVE_MB,
    Grant,
    MemoryLedger,
    Refusal,
)


def _ledger(total_mb: float = 12216.0) -> MemoryLedger:
    return MemoryLedger(total_mb=total_mb)


def _check(ledger: MemoryLedger) -> None:
    """Inspect the ledger after an operation (MEM-03H-01).

    The two facts that must hold after *every* operation, before any
    assertion about that operation's own result looks at anything else:
    claims never exceed the capacity, and every total is a real number.
    The seed bug was exactly their violation -- a NaN held total made
    ``free_mb()`` NaN, and ``demand > NaN`` is False, so the ledger then
    reported 19,592 MB free on an 11,192 MB card and granted it.
    """
    held = ledger.held_mb()
    capacity = ledger.capacity_mb
    free = ledger.free_mb()
    assert math.isfinite(held) and math.isfinite(free), (
        f"non-finite total after the operation: held={held} free={free}"
    )
    assert held <= capacity, f"held {held} exceeds capacity {capacity}"
    assert free == capacity - held, f"free {free} != capacity {capacity} - held {held}"


# -- capacity ---------------------------------------------------------------


def test_capacity_is_total_minus_foreign():
    """Capacity = total - foreign reserve."""
    ledger = _ledger(total_mb=12216.0)
    assert ledger.capacity_mb == 12216.0 - DEFAULT_FOREIGN_RESERVE_MB


def test_total_mb():
    """Total is the raw card size."""
    ledger = _ledger(total_mb=12216.0)
    assert ledger.total_mb == 12216.0


# -- reserve / release ------------------------------------------------------


def test_reserve_success():
    """A demand that fits is granted."""
    ledger = _ledger()
    result = ledger.reserve("graph_1", 7000.0)
    assert isinstance(result, Grant)
    assert result.owner == "graph_1"
    assert result.mb == 7000.0


def test_reserve_refusal_when_full():
    """A demand that doesn't fit is refused with a breakdown."""
    ledger = _ledger(total_mb=8000.0)
    r1 = ledger.reserve("graph_1", 5000.0)
    assert isinstance(r1, Grant)
    r2 = ledger.reserve("graph_2", 4000.0)
    assert isinstance(r2, Refusal)
    assert r2.requested_mb == 4000.0
    assert r2.free_mb < 4000.0
    assert "graph_1" in r2.holders


def test_reserve_explosive_only_when_empty():
    """An exploratory run is admitted only if nothing else holds the card."""
    ledger = _ledger()
    # The demand a real exploratory start passes is the capacity claim
    # (a zero demand is refused like any other unusable number,
    # MEM-03H-01); what this test pins is the exclusivity, not that.
    r1 = ledger.reserve("graph_1", ledger.capacity_mb, exploratory=True)
    assert isinstance(r1, Grant)
    assert r1.exploratory is True
    assert r1.mb == ledger.capacity_mb  # claims all free capacity


def test_reserve_explosive_refused_when_occupied():
    """An exploratory run is refused when another holder occupies the card."""
    ledger = _ledger()
    r1 = ledger.reserve("graph_1", 5000.0)
    assert isinstance(r1, Grant)
    r2 = ledger.reserve("graph_2", ledger.capacity_mb, exploratory=True)
    assert isinstance(r2, Refusal)
    assert "other holders" in r2.reason


def test_release_idempotent():
    """Releasing twice is safe."""
    ledger = _ledger()
    ledger.reserve("graph_1", 5000.0)
    ledger.release("graph_1")
    ledger.release("graph_1")  # no error
    assert ledger.held_mb() == 0.0


def test_release_frees_capacity():
    """Releasing a holder frees its claim."""
    ledger = _ledger(total_mb=8000.0)
    ledger.reserve("graph_1", 5000.0)
    assert ledger.free_mb() < 3000.0
    ledger.release("graph_1")
    assert ledger.free_mb() == ledger.capacity_mb


def test_rename_moves_the_claim():
    """rename re-keys a claim without changing what it holds.

    The start path takes the claim before the row exists (a refusal
    must not persist a row) and renames it to the row-derived owner
    once the id is bound -- size and bookkeeping must survive that.
    """
    ledger = _ledger()
    grant = ledger.reserve("graph:pending:abc", 4096.0)
    assert isinstance(grant, Grant)
    ledger.rename("graph:pending:abc", "graph:7")
    assert ledger.held_by("graph:pending:abc") == 0.0
    assert ledger.held_by("graph:7") == 4096.0
    assert ledger.held_mb() == 4096.0
    assert "graph:7" in ledger.snapshot()["holders"]


def test_rename_keeps_exploratory_flag():
    """An exclusive claim stays exclusive across the rename."""
    ledger = _ledger()
    grant = ledger.reserve(
        "task:pending:xyz", ledger.capacity_mb, exploratory=True
    )
    assert isinstance(grant, Grant) and grant.exploratory
    ledger.rename("task:pending:xyz", "task:3")
    holders = ledger.snapshot()["holders"]
    assert holders["task:3"]["exploratory"] is True
    refused = ledger.reserve("graph:1", 1.0)
    assert isinstance(refused, Refusal)


def test_rename_unknown_owner_is_a_noop():
    """Renaming a claim that was never taken changes nothing."""
    ledger = _ledger()
    ledger.reserve("a", 100.0)
    ledger.rename("nobody", "graph:1")
    assert ledger.held_mb() == 100.0
    assert ledger.held_by("graph:1") == 0.0
    assert ledger.held_by("a") == 100.0


# -- held / free ------------------------------------------------------------


def test_held_mb_sums_holders():
    """held_mb sums all holders."""
    ledger = _ledger()
    ledger.reserve("a", 1000.0)
    ledger.reserve("b", 2000.0)
    assert ledger.held_mb() == 3000.0


def test_held_by_unknown_is_zero():
    """held_by returns 0 for a holder that does not exist."""
    ledger = _ledger()
    assert ledger.held_by("nonexistent") == 0.0


def test_free_mb_is_capacity_minus_held():
    """free_mb = capacity - held."""
    ledger = _ledger(total_mb=10000.0)
    ledger.reserve("a", 2000.0)
    assert ledger.free_mb() == ledger.capacity_mb - 2000.0


# -- snapshot ---------------------------------------------------------------


def test_snapshot_shows_holders():
    """The snapshot names every holder."""
    ledger = _ledger()
    ledger.reserve("graph_1", 5000.0)
    snap = ledger.snapshot()
    assert "graph_1" in snap["holders"]
    assert snap["holders"]["graph_1"]["mb"] == 5000.0


def test_snapshot_free_and_held():
    """The snapshot reports free and held."""
    ledger = _ledger(total_mb=10000.0)
    ledger.reserve("a", 2000.0)
    snap = ledger.snapshot()
    assert snap["held_mb"] == 2000.0
    assert snap["free_mb"] == snap["capacity_mb"] - 2000.0


# -- rebuild_from_rows ------------------------------------------------------


def test_rebuild_from_rows():
    """On startup the ledger is rebuilt from rows."""
    ledger = _ledger()
    rows = [
        {"owner": "graph_1", "mb": 5000.0},
        {"owner": "graph_2", "mb": 3000.0},
    ]
    ledger.rebuild_from_rows(rows)
    assert ledger.held_mb() == 8000.0
    assert ledger.held_by("graph_1") == 5000.0
    assert ledger.held_by("graph_2") == 3000.0


def test_rebuild_clears_previous():
    """Rebuilding clears any previous state."""
    ledger = _ledger()
    ledger.reserve("old", 5000.0)
    ledger.rebuild_from_rows([{"owner": "new", "mb": 1000.0}])
    assert ledger.held_by("old") == 0.0
    assert ledger.held_by("new") == 1000.0


# -- thread safety ----------------------------------------------------------


def test_concurrent_reserves():
    """Concurrent reserves are serialized by the lock."""
    ledger = _ledger(total_mb=20000.0)
    results: list[Grant | Refusal] = []
    barrier = threading.Barrier(10)

    def worker(i: int):
        barrier.wait()
        r = ledger.reserve(f"worker_{i}", 2000.0)
        results.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly 10 holders, sum of grants <= capacity
    grants = [r for r in results if isinstance(r, Grant)]
    refusals = [r for r in results if isinstance(r, Refusal)]
    total_granted = sum(g.mb for g in grants)
    assert total_granted <= ledger.capacity_mb
    assert len(grants) + len(refusals) == 10


def test_concurrent_reserves_exact_fit():
    """N parallel starts that fit exactly K -> exactly K admitted."""
    total = 12216.0
    ledger = MemoryLedger(total_mb=total, foreign_reserve_mb=0.0)
    capacity = ledger.capacity_mb
    demand = capacity / 5.0  # 5 fit exactly

    results: list[Grant | Refusal] = []
    barrier = threading.Barrier(10)

    def worker(i: int):
        barrier.wait()
        r = ledger.reserve(f"worker_{i}", demand)
        results.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    grants = [r for r in results if isinstance(r, Grant)]
    refusals = [r for r in results if isinstance(r, Refusal)]
    assert len(grants) == 5
    assert len(refusals) == 5
    assert sum(g.mb for g in grants) <= capacity


# -- Grant / Refusal types ---------------------------------------------------


def test_grant_is_grant():
    """Grant has owner and mb."""
    g = Grant(owner="test", mb=1000.0)
    assert g.owner == "test"
    assert g.mb == 1000.0
    assert g.exploratory is False


def test_refusal_breakdown():
    """Refusal breakdown has all fields."""
    r = Refusal(
        owner="test",
        requested_mb=5000.0,
        capacity_mb=10000.0,
        foreign_reserve_mb=1024.0,
        free_mb=2000.0,
        holders={"other": 8000.0},
        reason="not enough",
    )
    bd = r.breakdown()
    assert bd["owner"] == "test"
    assert bd["requested_mb"] == 5000.0
    assert bd["free_mb"] == 2000.0
    assert "other" in bd["holders"]
    assert bd["what_would_fit"] == 2000.0


# -- MEM-03H-01: unusable demands -------------------------------------------
#
# Every value that must never become a claim, at the ledger -- the third
# of three independent checks (the API refuses with a 422, the domain
# raises, and here a Refusal names the value). The seeds came from the
# reproduction: nan was *granted* and made every total NaN, after which
# "demand > NaN" was False and all later claims were granted too; -9000
# was granted as if it freed space; zero was accepted as a claim.

#: (value, one more that keeps the ledger honest): a boolean and a
#: non-number ride along because Python compares them like numbers --
#: True is the int 1, and "lots" > 99999 is simply a TypeError the
#: caller should never have been able to trigger from a stored value.
_UNUSABLE_DEMANDS = [
    -1,
    0,
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    "lots",
    10**400,  # too big to ever be a device size: refuses, not OverflowError
]


def test_reserve_refuses_unusable_demands():
    """Each unusable demand is refused, named, and changes nothing."""
    for value in _UNUSABLE_DEMANDS:
        ledger = _ledger()
        before_free = ledger.free_mb()
        refusal = ledger.reserve("graph:1", value)
        assert isinstance(refusal, Refusal), f"{value!r} was granted"
        assert str(value) in refusal.reason, (
            f"the reason must name the value: {refusal.reason!r}"
        )
        assert ledger.held_mb() == 0.0, f"{value!r} left held={ledger.held_mb()}"
        assert ledger.free_mb() == before_free, (
            f"{value!r} changed free: {ledger.free_mb()} != {before_free}"
        )
        assert ledger.snapshot()["holders"] == {}
        _check(ledger)


def test_bad_demand_refusal_breakdown_is_json():
    """A refusal over an unusable demand still serializes as JSON.

    The breakdown goes straight into the 409 response, and Starlette
    serializes with ``allow_nan=False`` -- a ``requested_mb`` carrying
    NaN would raise where this refusal owes an answer (the same 500
    ``presentation/responses.py`` documents for a frame that carries
    one). ``allow_nan=False`` here is that constraint, failing if the
    sanitizing stops happening.
    """
    ledger = _ledger()
    for value in (float("nan"), float("inf"), float("-inf"), "lots", 10**400):
        refusal = ledger.reserve("graph:1", value)
        assert isinstance(refusal, Refusal)
        body = json.dumps(refusal.breakdown(), allow_nan=False)
        assert "requested_mb" in body
        _check(ledger)


def test_exploratory_also_refuses_unusable_demands():
    """The demand check runs whatever the mode (MEM-03H-01 says *any*).

    An exploratory claim would have been the card's free space, so the
    number itself could not have poisoned the totals -- but it is still
    what the refusal reports, and one rule is easier to reason about
    than a rule with a mode-shaped hole in it.
    """
    ledger = _ledger()
    for value in (float("nan"), 0, -1, "lots"):
        refusal = ledger.reserve("probe", value, exploratory=True)
        assert isinstance(refusal, Refusal), f"exploratory {value!r} was granted"
        assert str(value) in refusal.reason, refusal.reason
        assert ledger.held_mb() == 0.0
        _check(ledger)
    # The demand production actually sends for an exploratory start.
    grant = ledger.reserve("probe", ledger.capacity_mb, exploratory=True)
    assert isinstance(grant, Grant) and grant.exploratory
    _check(ledger)


def test_rebuild_skips_unusable_claims():
    """A row with an unusable claim is skipped *and logged* (MEM-03H-01).

    One NaN row would poison the rebuilt totals -- exactly the ledger
    the startup code exists to reproduce.
    """
    records: list[str] = []

    class _Recorder(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Recorder()
    module_logger = logging.getLogger("backend.application.memory_ledger")
    module_logger.addHandler(handler)
    try:
        ledger = _ledger()
        ledger.rebuild_from_rows([
            {"owner": "graph:1", "mb": 5000.0},
            {"owner": "graph:nan", "mb": float("nan")},
            {"owner": "graph:neg", "mb": -5.0},
            {"owner": "graph:zero", "mb": 0.0},
            {"owner": "graph:lots", "mb": "lots"},
            {"owner": "graph:2", "mb": 2000.0},
        ])
    finally:
        module_logger.removeHandler(handler)
    assert ledger.held_mb() == 7000.0, ledger.held_mb()
    for skipped in ("graph:nan", "graph:neg", "graph:zero", "graph:lots"):
        assert ledger.held_by(skipped) == 0.0, skipped
        assert skipped in "\n".join(records), f"{skipped} was skipped silently"
    assert "graph:1" not in "\n".join(records), "a good row was complained about"
    _check(ledger)


#: Every operation shape, over arbitrary floats: the invariant this
#: property defends (held never exceeds capacity, totals stay finite)
#: is the whole admission contract -- the unit tests above pin the
#: known-bad values, this pins all the others at once.
_operations = st.lists(
    st.tuples(
        st.sampled_from(("reserve", "exploratory", "release", "rename")),
        st.sampled_from(("graph:1", "graph:2", "graph:pending:x")),
        st.sampled_from(("graph:1", "graph:2", "task:9")),
        st.floats(allow_nan=True, allow_infinity=True),
    ),
    min_size=1,
    max_size=30,
)


@given(_operations)
@settings(max_examples=200, deadline=None)
def test_random_sequences_stay_within_capacity(ops) -> None:
    """Arbitrary reserve/release/rename sequences never overcommit."""
    ledger = _ledger()
    for kind, owner, other, demand in ops:
        if kind == "reserve":
            ledger.reserve(owner, demand)
        elif kind == "exploratory":
            ledger.reserve(owner, demand, exploratory=True)
        elif kind == "release":
            ledger.release(owner)
        else:
            ledger.rename(owner, other)
        _check(ledger)



def main() -> None:
    """Run every test in this file, listed by name.

    Listed, not discovered: a `def test_*` nothing calls is a comment
    shaped like a safety net, and `scripts/check_test_wiring.py` fails
    this file when one is defined and left out here -- all 19 of
    these were, and the file exited 0 having run nothing, before that
    check caught it.
    """
    tests = [
        test_capacity_is_total_minus_foreign,
        test_total_mb,
        test_reserve_success,
        test_reserve_refusal_when_full,
        test_reserve_explosive_only_when_empty,
        test_reserve_explosive_refused_when_occupied,
        test_release_idempotent,
        test_release_frees_capacity,
        test_rename_moves_the_claim,
        test_rename_keeps_exploratory_flag,
        test_rename_unknown_owner_is_a_noop,
        test_held_mb_sums_holders,
        test_held_by_unknown_is_zero,
        test_free_mb_is_capacity_minus_held,
        test_snapshot_shows_holders,
        test_snapshot_free_and_held,
        test_rebuild_from_rows,
        test_rebuild_clears_previous,
        test_concurrent_reserves,
        test_concurrent_reserves_exact_fit,
        test_grant_is_grant,
        test_refusal_breakdown,
        test_reserve_refuses_unusable_demands,
        test_bad_demand_refusal_breakdown_is_json,
        test_exploratory_also_refuses_unusable_demands,
        test_rebuild_skips_unusable_claims,
        test_random_sequences_stay_within_capacity,
    ]
    for test in tests:
        test()
    print()
    print("=" * 60)
    print(f"SMOKE TEST: ALL {len(tests)} CHECKS PASSED")


if __name__ == "__main__":
    main()
