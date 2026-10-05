"""Checks nodes/memory/graph_memory.py's GraphMemory (MEM-05 #1).

The child's memory picture: two numbers, their unknowns named, and the
validation that keeps a NaN or a negative out of everything downstream.
The NaN case is the load-bearing one -- a NaN grant would make every
``free < grant`` comparison False, so the physical check that follows
this constructor would pass on garbage instead of refusing it.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.core import ExecutionContext
from nodes.memory.graph_memory import GraphMemory

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


def main():
    check_stores_both_numbers()
    check_absent_is_unknown_not_zero()
    check_rejects_bad_numbers()
    check_nan_cannot_smuggle_through()
    check_exec_context_carries_memory()

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
