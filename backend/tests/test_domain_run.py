"""Domain tests -- the Run state machine owns every lifecycle rule.

Run directly: python backend/tests/test_domain_run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.domain.entities.run import Run
from backend.domain.exceptions import DomainError, InvalidTransitionError
from backend.domain.value_objects import RunStatus
from backend.tests.support import FakeClock, check, finish


def _persisted(clock: FakeClock, **kwargs) -> Run:
    run = Run.create(
        config_path=kwargs.get("config_path", "configs/x.toml"),
        mode="distillation",
        total_steps=5,
        created_at=clock.now(),
    )
    run.assign_id(7)
    run.collect_events()  # drop run_created noise for the assertions below
    return run


def test_happy_path() -> None:
    print("\n== happy path: create -> start -> progress -> complete ==")
    clock = FakeClock()
    run = Run.create(
        config_path="configs/legacy_check.toml",
        mode="distillation",
        total_steps=10,
        created_at=clock.now(),
    )
    check(run.status is RunStatus.CREATED, "new run starts in 'created'")
    check(run.id is None, "new run has no id until persisted")
    check(run.collect_events() == [], "no events before an id exists")
    check(run.updated_at == clock.now(), "updated_at initialised to created_at")

    run.assign_id(1)
    events = run.collect_events()
    check(
        len(events) == 1 and events[0].event_type == "run_created",
        f"assign_id emits run_created (got {[e.event_type for e in events]})",
    )
    check(
        events[0].run_id == 1 and events[0].total_steps == 10,
        "run_created carries id + config facts",
    )

    clock.advance(2)
    run.mark_started(pid=4242, at=clock.now())
    events = run.collect_events()
    check(
        run.status is RunStatus.RUNNING and run.pid == 4242,
        "mark_started -> running with pid",
    )
    check(run.started_at == clock.now(), "started_at stamped from the injected clock")
    check([e.event_type for e in events] == ["run_started"], "run_started emitted")

    clock.advance(5)
    run.record_progress(
        done_steps=4, at=clock.now(), current_loss=2.5, avg_loss=3.0, phase="training"
    )
    check(
        (run.done_steps, run.current_loss, run.avg_loss, run.phase)
        == (4, 2.5, 3.0, "training"),
        "progress fields recorded",
    )
    check(run.updated_at == clock.now(), "progress touches updated_at")
    check(run.collect_events() == [], "progress emits no domain event (telemetry)")

    clock.advance(1)
    run.mark_completed(at=clock.now())
    events = run.collect_events()
    check(
        run.status is RunStatus.COMPLETED and run.finished_at == clock.now(),
        "mark_completed -> completed + finished_at",
    )
    check(
        len(events) == 1
        and events[0].event_type == "run_completed"
        and events[0].done_steps == 4,
        "run_completed emitted with final done_steps",
    )
    check(run.collect_events() == [], "event buffer drains exactly once")


def test_illegal_transitions() -> None:
    print("\n== illegal transitions leave the entity untouched ==")
    clock = FakeClock()

    run = _persisted(clock)
    try:
        run.mark_completed(at=clock.now())
        check(False, "created -> completed must be rejected")
    except InvalidTransitionError:
        check(run.status is RunStatus.CREATED, "rejected transition changed nothing")
    try:
        run.mark_failed(at=clock.now())
        check(False, "created -> failed must be rejected")
    except InvalidTransitionError:
        check(True, "created -> failed rejected")
    try:
        run.record_progress(done_steps=1, at=clock.now())
        check(False, "progress while 'created' must be rejected")
    except InvalidTransitionError:
        check(True, "progress outside 'running' rejected")

    started = _persisted(clock)
    started.mark_started(pid=1, at=clock.now())
    started.collect_events()
    try:
        started.mark_started(pid=2, at=clock.now())
        check(False, "double start must be rejected")
    except InvalidTransitionError:
        check(True, "double start rejected")
    started.mark_completed(at=clock.now())
    started.collect_events()
    for name, action in (
        ("restart", lambda: started.mark_started(pid=3, at=clock.now())),
        ("cancel", lambda: started.cancel(at=clock.now())),
        ("fail", lambda: started.mark_failed(at=clock.now())),
    ):
        try:
            action()
            check(False, f"{name} after completion must be rejected")
        except InvalidTransitionError:
            check(True, f"{name} after completion rejected")
    check(started.status is RunStatus.COMPLETED, "status stays terminal")

    cancelled = _persisted(clock)
    cancelled.cancel(at=clock.now())
    check(cancelled.status is RunStatus.CANCELLED, "created -> cancelled allowed")
    try:
        cancelled.mark_started(pid=1, at=clock.now())
        check(False, "start after cancel must be rejected")
    except InvalidTransitionError:
        check(True, "start after cancel rejected")


def test_invariants() -> None:
    print("\n== construction and identity invariants ==")
    clock = FakeClock()

    for kwargs, label in (
        (dict(config_path="", mode="m", total_steps=1, created_at=clock.now()), "empty config_path"),
        (dict(config_path="c", mode="", total_steps=1, created_at=clock.now()), "empty mode"),
        (dict(config_path="c", mode="m", total_steps=-1, created_at=clock.now()), "negative total_steps"),
    ):
        try:
            Run.create(**kwargs)
            check(False, f"create with {label} must be rejected")
        except DomainError:
            check(True, f"create with {label} rejected")

    try:
        Run(
            status=RunStatus.CREATED,
            config_path="c",
            mode="m",
            total_steps=1,
            done_steps=-2,
            created_at=clock.now(),
        )
        check(False, "create with negative done_steps must be rejected")
    except DomainError:
        check(True, "create with negative done_steps rejected")

    unpersisted = Run.create(
        config_path="c", mode="m", total_steps=1, created_at=clock.now()
    )
    try:
        unpersisted.mark_started(pid=1, at=clock.now())
        check(False, "start before persisting must be rejected")
    except DomainError:
        check(
            unpersisted.status is RunStatus.CREATED,
            "start before persisting rejected without mutating status",
        )

    run = _persisted(clock)
    try:
        run.assign_id(8)
        check(False, "re-binding an id must be rejected")
    except DomainError:
        check(True, "assign_id twice rejected")
    try:
        run.assign_id(0)
        check(False, "id 0 must be rejected")
    except DomainError:
        check(True, "assign_id(0) rejected")

    run.mark_started(pid=1, at=clock.now())
    try:
        run.record_progress(done_steps=-1, at=clock.now())
        check(False, "negative progress must be rejected")
    except DomainError:
        check(run.done_steps == 0, "negative progress rejected without mutation")


def main() -> None:
    test_happy_path()
    test_illegal_transitions()
    test_invariants()
    finish()


if __name__ == "__main__":
    main()
