"""Unit tests -- use cases against the in-memory fake repository.

Run directly: python backend/tests/test_use_cases.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.dto import ListRunsQuery
from backend.application.errors import InvalidQueryError, RunNotFoundError
from backend.domain.events import RunsDeleted
from backend.domain.value_objects import RunStatus
from backend.tests.support import (
    FakeClock,
    InMemoryRunRepository,
    RecordingEventBus,
    build_services,
    check,
    finish,
    seed_run,
)


def _wired():
    clock = FakeClock()
    repo = InMemoryRunRepository()
    events = RecordingEventBus()
    return clock, repo, events, build_services(runs=repo, events=events)


def test_list_runs() -> None:
    print("\n== ListRuns: paging, filtering, projection ==")
    clock, repo, _, services = _wired()
    seed_run(repo, clock)
    seed_run(repo, clock, config_path="configs/b.toml")
    running = seed_run(repo, clock, config_path="configs/c.toml", start=True)

    result = services.list_runs.execute()
    check(result.count == 3 and len(result.runs) == 3, "default query lists all runs")
    check(result.runs[0].id == 3, "newest first")
    check(result.runs[0].mode == "distillation", "DTO projects entity fields")

    filtered = services.list_runs.execute(ListRunsQuery(status="running"))
    check(
        filtered.count == 1 and filtered.runs[0].id == running.id,
        "status filter narrows correctly",
    )
    check(
        filtered.runs[0].status is RunStatus.RUNNING,
        "DTO carries the RunStatus enum, not a raw string",
    )

    paged = services.list_runs.execute(ListRunsQuery(limit=2))
    check(paged.count == 2, "limit caps the page")


def test_query_validation() -> None:
    print("\n== ListRuns rejects impossible queries ==")
    _, _, _, services = _wired()
    for bad_limit in (0, -1, 501):
        try:
            services.list_runs.execute(ListRunsQuery(limit=bad_limit))
            check(False, f"limit {bad_limit} must be rejected")
        except InvalidQueryError as exc:
            check(exc.code == "invalid_query", f"limit {bad_limit} -> invalid_query")
    try:
        services.list_runs.execute(ListRunsQuery(status="bogus"))
        check(False, "unknown status must be rejected")
    except InvalidQueryError as exc:
        check(exc.code == "invalid_query", "unknown status -> invalid_query")
        check(
            "bogus" in str(exc) and "running" in str(exc),
            "message names the bad value and the legal ones",
        )


def test_get_run() -> None:
    print("\n== GetRun: found, missing, absurd ==")
    clock, repo, _, services = _wired()
    run = seed_run(repo, clock, config_path="configs/target.toml", mode="lora")

    dto = services.get_run.execute(run.id)
    check(dto.id == run.id and dto.config_path == "configs/target.toml", "found -> DTO")
    check(dto.status is RunStatus.CREATED, "freshly seeded run reads as created")

    try:
        services.get_run.execute(999)
        check(False, "missing run must raise")
    except RunNotFoundError as exc:
        check(exc.code == "run_not_found", "missing run -> run_not_found")

    try:
        services.get_run.execute(0)
        check(False, "id 0 must be rejected")
    except InvalidQueryError as exc:
        check(exc.code == "invalid_query", "id 0 -> invalid_query")


def test_delete_runs() -> None:
    print("\n== DeleteRuns: count + one domain event ==")
    clock, repo, events, services = _wired()

    empty = services.delete_runs.execute()
    check(empty.deleted == 0, "delete on empty history -> 0")
    check(events.published == [], "no event when nothing was deleted")

    seed_run(repo, clock)
    seed_run(repo, clock)
    result = services.delete_runs.execute()
    check(result.deleted == 2, "delete reports rows removed")
    check(len(repo.list()) == 0, "history is empty afterwards")
    published = events.published
    check(
        len(published) == 1
        and isinstance(published[0], RunsDeleted)
        and published[0].deleted == 2,
        "exactly one RunsDeleted event carrying the count",
    )


def main() -> None:
    test_list_runs()
    test_query_validation()
    test_get_run()
    test_delete_runs()
    finish()


if __name__ == "__main__":
    main()
