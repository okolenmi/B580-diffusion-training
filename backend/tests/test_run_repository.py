"""Integration tests -- SqliteRunRepository against a real (temp) file.

Run directly: python backend/tests/test_run_repository.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.domain.entities.run import Run
from backend.domain.exceptions import DomainError
from backend.domain.value_objects import RunStatus
from backend.infrastructure.persistence.run_repository import SqliteRunRepository
from backend.infrastructure.persistence.sqlite import SqliteDatabase
from backend.tests.support import FakeClock, check, finish


def _open(tmp: str) -> SqliteDatabase:
    db = SqliteDatabase(Path(tmp) / "runs.db")
    db.initialize()
    return db


def test_roundtrip() -> None:
    print("\n== add / update / get roundtrip ==")
    with tempfile.TemporaryDirectory() as tmp:
        repo = SqliteRunRepository(_open(tmp))
        clock = FakeClock()
        run = Run.create(
            config_path="configs/a.toml",
            mode="distillation",
            total_steps=100,
            created_at=clock.now(),
        )
        repo.add(run)
        check(run.id == 1, f"add assigns id (got {run.id})")
        check(
            [e.event_type for e in run.collect_events()] == ["run_created"],
            "add emitted run_created",
        )

        run.mark_started(pid=99, at=clock.now())
        clock.advance(3)
        run.record_progress(
            done_steps=12, at=clock.now(), current_loss=1.25, avg_loss=2.5, phase="training"
        )
        check(repo.update(run), "update persists")

        fetched = repo.get(run.id)
        check(fetched is not None, "get returns the run")
        check(
            fetched.status is RunStatus.RUNNING and fetched.pid == 99,
            "status + pid roundtrip",
        )
        check(
            (fetched.done_steps, fetched.current_loss, fetched.avg_loss)
            == (12, 1.25, 2.5),
            "progress + loss roundtrip",
        )
        check(fetched.phase == "training", "phase roundtrip")
        check(
            fetched.cache_done is None and fetched.cache_total is None,
            "cache columns default to NULL",
        )
        check(
            fetched.created_at == run.created_at and fetched.started_at == run.started_at,
            "timestamps roundtrip (tz-aware equality)",
        )
        check(
            fetched.log_path is None
            and fetched.error is None
            and fetched.exit_code is None
            and fetched.finished_at is None,
            "NULL columns roundtrip as None",
        )
        check(
            fetched.collect_events() == [],
            "reconstructed run starts with an empty event buffer",
        )


def test_listing_and_filters() -> None:
    print("\n== list ordering, limit, status filter ==")
    with tempfile.TemporaryDirectory() as tmp:
        repo = SqliteRunRepository(_open(tmp))
        clock = FakeClock()
        runs = []
        for index in range(3):
            runs.append(
                Run.create(
                    config_path=f"configs/{index}.toml",
                    mode="distillation",
                    total_steps=10 * (index + 1),
                    created_at=clock.now(),
                )
            )
            repo.add(runs[-1])
            clock.advance(1)
        runs[1].mark_started(pid=5, at=clock.now())
        repo.update(runs[1])

        listed = repo.list()
        check(
            [r.id for r in listed] == [3, 2, 1],
            f"newest first (got {[r.id for r in listed]})",
        )
        check(len(repo.list(limit=2)) == 2, "limit caps the page")
        only_running = repo.list(status=RunStatus.RUNNING)
        check(
            len(only_running) == 1 and only_running[0].id == 2,
            "status filter narrows correctly",
        )
        check(repo.list(status=RunStatus.FAILED) == [], "empty filter result")


def test_find_active_and_delete() -> None:
    print("\n== find_active + delete_all ==")
    with tempfile.TemporaryDirectory() as tmp:
        repo = SqliteRunRepository(_open(tmp))
        clock = FakeClock()
        run = Run.create(
            config_path="c", mode="m", total_steps=1, created_at=clock.now()
        )
        repo.add(run)
        active = repo.find_active()
        check(
            active is not None and active.id == run.id,
            "created run counts as active (blocks a second start)",
        )

        run.mark_started(pid=1, at=clock.now())
        repo.update(run)
        active = repo.find_active()
        check(active is not None and active.id == run.id, "active run found while running")

        run.mark_completed(at=clock.now())
        repo.update(run)
        check(repo.find_active() is None, "completed run is no longer active")

        check(repo.delete_all() == 1, "delete_all returns rows removed")
        check(repo.get(run.id) is None, "runs really gone")
        check(repo.delete_all() == 0, "delete_all on empty table returns 0")


def test_update_edge_cases() -> None:
    print("\n== update edge cases ==")
    with tempfile.TemporaryDirectory() as tmp:
        repo = SqliteRunRepository(_open(tmp))
        clock = FakeClock()
        orphan = Run.create(
            config_path="c", mode="m", total_steps=1, created_at=clock.now()
        )
        orphan.assign_id(999)
        check(repo.update(orphan) is False, "update of a missing id returns False")

        fresh = Run.create(
            config_path="c", mode="m", total_steps=1, created_at=clock.now()
        )
        try:
            repo.update(fresh)
            check(False, "update of an unpersisted run must be rejected")
        except DomainError:
            check(True, "update of an unpersisted run rejected")


def test_persistence_across_instances() -> None:
    print("\n== data survives a new connection; migrations idempotent ==")
    with tempfile.TemporaryDirectory() as tmp:
        db1 = _open(tmp)
        repo1 = SqliteRunRepository(db1)
        clock = FakeClock()
        run = Run.create(
            config_path="c", mode="m", total_steps=3, created_at=clock.now()
        )
        repo1.add(run)

        db2 = SqliteDatabase(Path(tmp) / "runs.db")
        db2.initialize()  # second run of migrations must be a no-op
        repo2 = SqliteRunRepository(db2)
        check(repo2.get(run.id) is not None, "second database instance sees the row")

        with db2.connection() as conn:
            applied = [
                row["version"]
                for row in conn.execute("SELECT version FROM schema_migrations")
            ]
        # The invariant: every migration file on disk is recorded, in order.
        migrations_dir = (
            Path(__file__).resolve().parents[1]
            / "infrastructure"
            / "persistence"
            / "migrations"
        )
        expected = sorted(p.stem for p in migrations_dir.glob("*.sql"))
        check(
            applied == expected,
            f"schema_migrations records every migration file (got {applied}, "
            f"expected {expected})",
        )


def test_update_if_status() -> None:
    print("\n== update_if_status (compare-and-swap) ==")
    with tempfile.TemporaryDirectory() as tmp:
        repo = SqliteRunRepository(_open(tmp))
        clock = FakeClock()
        run = Run.create(
            config_path="c", mode="m", total_steps=1, created_at=clock.now()
        )
        repo.add(run)

        run.mark_started(pid=7, at=clock.now())
        check(
            repo.update_if_status(run, expected=RunStatus.CREATED),
            "CAS succeeds when the row still has the expected status",
        )
        check(
            repo.get(run.id).status is RunStatus.RUNNING,
            "row took the new status",
        )

        # A racing writer that already transitioned the row: the loser's
        # write (stale entity copy still believing it is running) must
        # not land. Capture the loser BEFORE the winner transitions.
        loser = repo.get(run.id)  # independent copy (fresh row -> Run)
        winner = repo.get(run.id)
        winner.mark_completed(at=clock.now())
        check(
            repo.update_if_status(winner, expected=RunStatus.RUNNING),
            "first terminal writer wins",
        )

        loser.mark_failed(at=clock.now(), error="late loser")
        check(
            repo.update_if_status(loser, expected=RunStatus.RUNNING) is False,
            "second terminal writer sees False",
        )
        check(
            repo.get(run.id).status is RunStatus.COMPLETED,
            "loser did not overwrite the winner's status",
        )

        check(
            repo.update_if_status(run, expected=RunStatus.CREATED) is False,
            "stale expected status fails the CAS",
        )
        try:
            repo.update_if_status(
                Run.create(config_path="c", mode="m", total_steps=1, created_at=clock.now()),
                expected=RunStatus.CREATED,
            )
            check(False, "CAS of an unpersisted run must be rejected")
        except DomainError:
            check(True, "CAS of an unpersisted run rejected")


def test_list_unfinished() -> None:
    print("\n== list_unfinished ==")
    with tempfile.TemporaryDirectory() as tmp:
        repo = SqliteRunRepository(_open(tmp))
        clock = FakeClock()
        created = Run.create(
            config_path="a", mode="m", total_steps=1, created_at=clock.now()
        )
        repo.add(created)
        running = Run.create(
            config_path="b", mode="m", total_steps=1, created_at=clock.now()
        )
        repo.add(running)
        running.mark_started(pid=3, at=clock.now())
        repo.update(running)
        done = Run.create(
            config_path="c", mode="m", total_steps=1, created_at=clock.now()
        )
        repo.add(done)
        done.mark_started(pid=4, at=clock.now())
        done.mark_completed(at=clock.now())
        repo.update(done)

        unfinished = repo.list_unfinished()
        check(
            [r.id for r in unfinished] == [2, 1],
            f"unfinished, newest first (got {[r.id for r in unfinished]})",
        )
        check(
            all(r.id != 3 for r in unfinished),
            "terminal rows never appear in unfinished",
        )


def main() -> None:
    test_roundtrip()
    test_listing_and_filters()
    test_find_active_and_delete()
    test_update_edge_cases()
    test_persistence_across_instances()
    test_update_if_status()
    test_list_unfinished()
    finish()


if __name__ == "__main__":
    main()
