"""The RunRepository contract, written once (docs 08 Q9).

``InMemoryRunRepository`` and ``SqliteRunRepository`` implement the same
port and are both used -- by the test suite and by production
respectively -- so a behaviour that holds for one and not the other is a
bug that surfaces only once the tests are swapped for the real thing.
That is the failure mode this file exists to prevent: the fake is
*easier* to believe, so drift in it is a lie the suite tells with a
straight face.

Nothing here knows which implementation it is talking to. Each check
names a behaviour of the *port*, and the runner executes the whole file
against every implementation. A check that passes for one and fails for
the other is the entire point of the file.

Deliberately *not* asserted here: identity and copy semantics. SQLite
necessarily stores a copy; the fake returns the object it was handed,
which is what its ``_statuses`` mirror exists to compensate for. That
divergence is real, and ``test_repository_contract.py`` asserts it
separately so it stays a stated fact rather than a surprise.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from backend.domain.entities.run import Run
from backend.domain.exceptions import DomainError
from backend.domain.value_objects import RunStatus
from backend.tests.support import FakeClock, check

# A factory that builds one empty repository.
RepoFactory = Callable[[], "RunRepositoryLike"]


class RunRepositoryLike(Protocol):
    """The surface under test, restated so this file needs no import of
    the port's ABC (which would pin it to one spelling)."""

    def add(self, run: Run) -> Run: ...
    def get(self, run_id: int) -> Run | None: ...
    def list_runs(self, *, limit: int = 50,
                  status: RunStatus | None = None) -> list[Run]: ...
    def update(self, run: Run) -> bool: ...
    def update_if_status(self, run: Run, expected: RunStatus) -> bool: ...
    def continue_ids_above(self, run_id: int) -> None: ...
    def find_active(self) -> Run | None: ...
    def list_unfinished(self) -> list[Run]: ...
    def delete_all(self) -> int: ...


_CLOCK = FakeClock()

CREATED = "created"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"


def _fresh(*, path: str = "configs/a.toml", total: int = 100) -> Run:
    """An unpersisted run. Any transition needs an id first, so a run
    cannot be built in a terminal state before it is stored."""
    return Run.create(
        config_path=path, mode="distillation", total_steps=total,
        created_at=_CLOCK.now(),
    )


def _drive(run: Run, status: str) -> Run:
    """Take a *persisted* run to `status`. Raises for anything the
    lifecycle forbids, which is the point: the contract builds only
    reachable states."""
    at = run.created_at
    if status == CREATED:
        return run
    if status == RUNNING:
        run.mark_started(pid=4242, at=at)
    elif status == COMPLETED:
        run.mark_started(pid=4242, at=at)
        run.mark_completed(at=at)
    elif status == FAILED:
        run.mark_started(pid=4242, at=at)
        run.mark_failed(at=at, error="boom", exit_code=1)
    elif status == CANCELLED:
        run.mark_started(pid=4242, at=at)
        run.cancel(at=at, reason="stop requested")
    else:
        raise AssertionError(f"unknown status {status!r}")
    return run


def _added(repo: RunRepositoryLike, status: str = CREATED, *,
           path: str = "configs/a.toml") -> Run:
    """A stored run already written through to `status`."""
    run = repo.add(_fresh(path=path))
    _drive(run, status)
    repo.update(run)
    return repo.get(run.require_id()) or run


def run_contract(make: RepoFactory, name: str) -> None:
    """Execute every contract check against one implementation.

    `make` builds one empty repository; `name` says which implementation
    this is, and appears in every failure message -- a contract check
    that fails without naming the implementation is half a report.
    """
    def fail(message: str) -> None:
        check(False, f"[{name}] {message}")

    # -- add ---------------------------------------------------------------
    repo = make()
    first = repo.add(_fresh())
    second = repo.add(_fresh())
    check(first.require_id() == 1, f"[{name}] first add gets id 1 (got {first.id})")
    check(second.require_id() == 2, f"[{name}] second add gets id 2 (got {second.id})")

    repo = make()
    run = _fresh()
    check(repo.add(run) is run, f"[{name}] add returns its argument")

    repo = make()
    persisted = repo.add(_fresh())
    try:
        repo.add(persisted)
    except DomainError:
        pass
    else:
        fail("re-adding a persisted run must raise DomainError")

    repo = make()
    repo.add(_fresh())
    try:
        repo.add(repo.get(1))
    except DomainError:
        pass
    nxt = repo.add(_fresh())
    check(nxt.require_id() == 2,
          f"[{name}] a refused add consumed no id (got {nxt.id})")

    # -- get ---------------------------------------------------------------
    # A read-back never carries pending events. SQLite gets this for free
    # (restore() rebuilds with an empty buffer); a fake that stores the
    # caller's object, or a plain copy of it, does not -- and the cost is
    # a *duplicate* on the stream, because StartTraining repairs a failed
    # start by committing a run it read back, so a stale RunCreated in
    # that copy gets published a second time.
    repo = make()
    run = repo.add(_fresh())
    repo.get(run.require_id()).collect_events()  # drain the writer's copy
    check(repo.get(run.require_id()).collect_events() == [],
          f"[{name}] a read-back carries no pending events")

    repo = make()
    stored = _added(repo, RUNNING)
    fetched = repo.get(stored.require_id())
    check(fetched is not None, f"[{name}] get finds an added run")
    check(fetched.status is RUNNING or fetched.status is RunStatus.RUNNING,
          f"[{name}] get returns the persisted status (got {fetched.status})")
    check(fetched.total_steps == 100, f"[{name}] get returns total_steps")
    check(fetched.config_path == "configs/a.toml",
          f"[{name}] get returns config_path")

    check(make().get(999) is None, f"[{name}] get on an unknown id -> None")

    # -- list_runs ---------------------------------------------------------
    repo = make()
    added = [repo.add(_fresh(path=f"configs/{i}.toml")) for i in range(4)]
    listed = [x.require_id() for x in repo.list_runs(limit=50)]
    check(listed == [x.require_id() for x in reversed(added)],
          f"[{name}] list_runs is newest first (got {listed})")

    limited = [x.require_id() for x in repo.list_runs(limit=2)]
    check(limited == [added[3].require_id(), added[2].require_id()],
          f"[{name}] limit keeps the newest two (got {limited})")

    repo = make()
    a = _added(repo, RUNNING, path="configs/a.toml")
    b = _added(repo, FAILED, path="configs/b.toml")
    c = _added(repo, RUNNING, path="configs/c.toml")
    running = [x.require_id() for x in repo.list_runs(status=RunStatus.RUNNING)]
    check(running == [c.require_id(), a.require_id()],
          f"[{name}] status filter, then newest first (got {running})")
    failed = [x.require_id() for x in repo.list_runs(status=RunStatus.FAILED)]
    check(failed == [b.require_id()], f"[{name}] failed filter (got {failed})")
    check(repo.list_runs(status=RunStatus.CANCELLED) == [],
          f"[{name}] a status nobody has -> []")
    check(repo.list_runs(limit=0) == [],
          f"[{name}] limit 0 -> [] (not every row)")

    # -- update ------------------------------------------------------------
    # Both implementations raise for a run with no id. That was checked
    # as a possible *difference* first (the port does not say), and it is
    # not one -- so it is now pinned here as agreement. The alternative
    # was returning False, which would make an unpersisted write
    # indistinguishable from a lost race.
    repo = make()
    try:
        repo.update(_fresh())
    except DomainError:
        pass
    else:
        fail("update of a run with no id must raise DomainError, not return")

    try:
        repo.update_if_status(_fresh(), RunStatus.CREATED)
    except DomainError:
        pass
    else:
        fail("CAS on a run with no id must raise DomainError, not return")

    # A run that *has* an id but was never stored: the row is missing, so
    # this is a lost race and returns False rather than raising.
    repo = make()
    orphan = _fresh()
    orphan.assign_id(999)
    check(repo.update(orphan) is False,
          f"[{name}] update of an id no row holds -> False")
    check(repo.update_if_status(orphan, RunStatus.CREATED) is False,
          f"[{name}] CAS against an id no row holds -> False")

    repo = make()
    run = repo.add(_fresh())
    run.mark_started(pid=7, at=run.created_at)
    check(repo.update(run) is True, f"[{name}] update of a stored row -> True")
    reloaded = repo.get(run.require_id())
    check(reloaded.status is RunStatus.RUNNING,
          f"[{name}] the update is visible on read-back (got {reloaded.status})")

    # -- update_if_status: the compare-and-swap ---------------------------
    repo = make()
    run = repo.add(_fresh())
    run.mark_started(pid=7, at=run.created_at)
    check(repo.update_if_status(run, RunStatus.CREATED) is True,
          f"[{name}] CAS wins when the persisted status matches")
    check(repo.get(run.require_id()).status is RunStatus.RUNNING,
          f"[{name}] and the row now says running")

    repo = make()
    run = repo.add(_fresh())
    check(repo.update_if_status(run, RunStatus.RUNNING) is False,
          f"[{name}] CAS loses on a status the row is not in")
    check(repo.get(run.require_id()).status is RunStatus.CREATED,
          f"[{name}] and the losing write did not land")

    # The CAS must decide against the *row*, not against the caller's copy.
    # A caller that mutates its entity to `failed` and then CASes on
    # `created` is asking "is the row still created?" -- yes -- so the
    # write must land, and the row must end up `failed`.
    repo = make()
    run = repo.add(_fresh())
    run.mark_started(pid=7, at=run.created_at)
    run.mark_failed(at=run.created_at, error="boom", exit_code=1)
    check(repo.update_if_status(run, RunStatus.CREATED) is True,
          f"[{name}] CAS reads the row, not the caller's in-memory copy")
    check(repo.get(run.require_id()).status is RunStatus.FAILED,
          f"[{name}] and the caller's state is what got written")


    # -- find_active / list_unfinished ------------------------------------
    repo = make()
    _added(repo, CREATED, path="configs/a.toml")
    active = repo.find_active()
    check(active is not None and active.status is RunStatus.CREATED,
          f"[{name}] find_active counts `created` as active "
          f"(got {active and active.status})")

    repo = make()
    _added(repo, COMPLETED, path="configs/a.toml")
    _added(repo, RUNNING, path="configs/b.toml")
    newest = _added(repo, CREATED, path="configs/c.toml")
    active = repo.find_active()
    check(active is not None and active.require_id() == newest.require_id(),
          f"[{name}] find_active is the newest unfinished run")

    repo = make()
    _added(repo, COMPLETED, path="configs/a.toml")
    _added(repo, FAILED, path="configs/b.toml")
    _added(repo, CANCELLED, path="configs/c.toml")
    check(repo.find_active() is None,
          f"[{name}] find_active ignores every terminal status")

    repo = make()
    _added(repo, COMPLETED, path="configs/a.toml")
    one = _added(repo, CREATED, path="configs/b.toml")
    two = _added(repo, RUNNING, path="configs/c.toml")
    got = [x.require_id() for x in repo.list_unfinished()]
    check(got == [two.require_id(), one.require_id()],
          f"[{name}] list_unfinished: created+running, newest first (got {got})")

    # An un-written mutation must not end the run. This is the single
    # behaviour the fake's `_statuses` mirror exists to preserve, and the
    # one most likely to drift, because in the fake the entity object
    # itself has already moved.
    repo = make()
    run = _added(repo, RUNNING, path="configs/a.toml")
    run.mark_completed(at=run.created_at)  # in memory only
    active = repo.find_active()
    check(active is not None and active.status is RunStatus.RUNNING,
          f"[{name}] an un-written completion does not end the run "
          f"(got {active and active.status})")

    # -- continue_ids_above ------------------------------------------------
    repo = make()
    repo.add(_fresh())
    repo.continue_ids_above(500)
    check(repo.add(_fresh()).require_id() == 501,
          f"[{name}] continue_ids_above moves the next id above the given one")

    repo = make()
    for _ in range(3):
        repo.add(_fresh())
    repo.continue_ids_above(1)
    check(repo.add(_fresh()).require_id() == 4,
          f"[{name}] a lower id never rewinds the counter")

    # -- delete_all --------------------------------------------------------
    repo = make()
    _added(repo, CREATED, path="configs/a.toml")
    _added(repo, RUNNING, path="configs/b.toml")
    _added(repo, COMPLETED, path="configs/c.toml")
    check(repo.delete_all() == 3, f"[{name}] delete_all counts every row")
    check(repo.list_runs(limit=50) == [], f"[{name}] nothing listed afterwards")
    check(repo.list_unfinished() == [], f"[{name}] nothing unfinished afterwards")
    check(repo.find_active() is None, f"[{name}] nothing active afterwards")
    check(repo.delete_all() == 0, f"[{name}] deleting nothing reports 0")
    repo.continue_ids_above(10)
    check(repo.add(_fresh()).require_id() == 11,
          f"[{name}] ids keep going after delete_all")