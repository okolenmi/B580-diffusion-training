"""Run the RunRepository contract against every implementation (Q9).

    python backend/tests/test_repository_contract.py

Why this exists in the shape it does: a repository contract written only
against the in-memory fake proves nothing about the SQLite adapter, and
one written only against SQLite makes the common case slow and awkward
to write. So the contract lives in ``contracts/`` and names no
implementation, and this runner hands it each one.

**The contract found a real difference and it has been fixed.** The fake
used to store the caller's object by reference, so a mutation that was
never written came back out of ``get``/``find_active``/``list_runs`` --
an in-flight change looked like a committed one. It carried a parallel
``_statuses`` dict to stop the compare-and-swap from being fooled, which
fixed the CAS and left the reads lying: ``find_active()`` filtered on the
mirror and then returned an entity whose own ``.status`` said something
else.

The fake now stores a snapshot on write and returns a fresh copy on read,
exactly as SQLite necessarily does, and ``_statuses`` is gone. The check
below is the *old* difference asserted in its new direction: both
implementations must now hand back an object the caller does not already
hold, and neither may leak an un-written mutation into a read.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.domain.entities.run import Run
from backend.domain.value_objects import RunStatus
from backend.infrastructure.persistence.run_repository import SqliteRunRepository
from backend.infrastructure.persistence.sqlite import SqliteDatabase
from backend.tests.contracts.run_repository_contract import run_contract
from backend.tests.support import FakeClock, InMemoryRunRepository, check, finish


def _sqlite_factory() -> SqliteRunRepository:
    """A fresh temp DB per repository, so each block starts empty."""
    tmp = tempfile.mkdtemp(prefix="run-contract-")
    db = SqliteDatabase(Path(tmp) / "runs.db")
    db.initialize()
    return SqliteRunRepository(db)


def _fresh() -> Run:
    clock = FakeClock()
    return Run.create(
        config_path="configs/a.toml", mode="distillation",
        total_steps=100, created_at=clock.now(),
    )


def test_contract_holds_for_both_implementations() -> None:
    print("\n== the RunRepository contract, run against every implementation ==")
    for name, factory in (
        ("InMemoryRunRepository", InMemoryRunRepository),
        ("SqliteRunRepository", _sqlite_factory),
    ):
        print(f"\n-- {name} --")
        run_contract(factory, name)


def test_the_difference_that_was_found_is_gone() -> None:
    """The fake must now behave like SQLite where it used to differ.

    Asserted explicitly rather than left to the contract, because these
    are the exact two assertions that used to fail -- and "we fixed it"
    is only worth anything while something keeps checking.
    """
    print("\n== the difference the contract found no longer exists ==")

    for name, repo in (
        ("InMemoryRunRepository", InMemoryRunRepository()),
        ("SqliteRunRepository", _sqlite_factory()),
    ):
        run = repo.add(_fresh())
        fetched = repo.get(run.require_id())
        check(fetched is not run,
              f"[{name}] a read returns an object the caller does not hold")

        # The case that used to differ: mutate without writing.
        run.mark_started(pid=99, at=run.created_at)
        check(repo.get(run.require_id()).status is RunStatus.CREATED,
              f"[{name}] an un-written mutation does not come back out of a read "
              f"(got {repo.get(run.require_id()).status})")

        # ...and the write does land.
        check(repo.update(run) is True, f"[{name}] the write is accepted")
        check(repo.get(run.require_id()).status is RunStatus.RUNNING,
              f"[{name}] and is visible afterwards")

    check(not hasattr(InMemoryRunRepository(), "_statuses"),
          "the workaround dict is gone -- the snapshot replaced the need for it")


def main() -> int:
    try:
        test_contract_holds_for_both_implementations()
        test_the_difference_that_was_found_is_gone()
    except AssertionError as exc:
        print(f"\nCONTRACT: FAILED -- {exc}")
        return 1
    finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())