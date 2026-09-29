"""Shared test support: assertions, fakes, and a raw-ASGI client.

House style (matches the repo's smoke tests): each ``test_*.py`` file
runs standalone -- ``check()`` prints PASS/FAIL per assertion, and
``finish()`` exits non-zero if anything failed. No pytest dependency.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.run_repository import RunRepository
from backend.application.services import ApplicationServices
from backend.application.use_cases import DeleteRuns, GetRun, ListRuns
from backend.domain.entities.run import Run
from backend.domain.events import DomainEvent
from backend.domain.exceptions import DomainError
from backend.domain.value_objects import RunId, RunStatus
from backend.infrastructure.events.callback_event_bus import CallbackEventBus

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}: {message}")
    if not condition:
        FAILURES.append(message)


def finish() -> None:
    print()
    print("=" * 60)
    if FAILURES:
        print(f"SMOKE TEST: {len(FAILURES)} FAILURE(S)")
        for failure in FAILURES:
            print(f"  - {failure}")
        sys.exit(1)
    print("SMOKE TEST: ALL CHECKS PASSED")


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeClock:
    """Deterministic clock; starts at a fixed instant and only moves
    when a test says so."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


class RecordingEventBus(CallbackEventBus):
    """The real bus plus a record of everything published."""

    def __init__(self) -> None:
        super().__init__()
        self.published: list[DomainEvent] = []

    def publish(self, event: DomainEvent) -> None:
        self.published.append(event)
        super().publish(event)


class InMemoryRunRepository(RunRepository):
    """Same semantics as the SQLite port, no persistence."""

    def __init__(self) -> None:
        self._runs: dict[int, Run] = {}
        self._next_id = 1

    def add(self, run: Run) -> Run:
        if run.id is not None:
            raise DomainError(f"run already has id {run.id}")
        run.assign_id(RunId(self._next_id))
        self._next_id += 1
        self._runs[run.id] = run
        return run

    def get(self, run_id: RunId) -> Run | None:
        return self._runs.get(run_id)

    def list(self, *, limit: int = 50, status: RunStatus | None = None) -> list[Run]:
        runs = [r for r in self._runs.values() if status is None or r.status is status]
        runs.sort(key=lambda r: r.id, reverse=True)
        return runs[:limit]

    def update(self, run: Run) -> bool:
        if run.id is None:
            raise DomainError("cannot update an unpersisted run (no id yet)")
        if run.id not in self._runs:
            return False
        self._runs[run.id] = run
        return True

    def find_active(self) -> Run | None:
        running = [r for r in self._runs.values() if r.status is RunStatus.RUNNING]
        return max(running, key=lambda r: r.id) if running else None

    def delete_all(self) -> int:
        deleted = len(self._runs)
        self._runs.clear()
        return deleted


def build_services(
    *, runs: RunRepository | None = None, events: CallbackEventBus | None = None
) -> ApplicationServices:
    """Wire the use cases against fakes (the composition root's test twin)."""
    runs = runs if runs is not None else InMemoryRunRepository()
    events = events if events is not None else RecordingEventBus()
    return ApplicationServices(
        list_runs=ListRuns(runs),
        get_run=GetRun(runs),
        delete_runs=DeleteRuns(runs, events),
        event_bus=events,
    )


def seed_run(
    repo: RunRepository,
    clock: FakeClock,
    *,
    config_path: str = "configs/test.toml",
    mode: str = "distillation",
    total_steps: int = 100,
    start: bool = False,
) -> Run:
    """Create + persist a run; optionally transition it to ``running``."""
    run = Run.create(
        config_path=config_path,
        mode=mode,
        total_steps=total_steps,
        created_at=clock.now(),
    )
    repo.add(run)
    if start:
        run.mark_started(pid=111, at=clock.now())
        repo.update(run)
    return run


# --------------------------------------------------------------------------
# Raw-ASGI client (no httpx dependency, like the legacy smoke tests)
# --------------------------------------------------------------------------


async def _asgi_call(app, path: str, *, method: str = "GET"):
    method = method or "GET"
    raw_path, _, query = path.partition("?")
    start: dict = {}
    chunks: list[bytes] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            start.update(message)
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": raw_path,
        "raw_path": raw_path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": [(b"host", b"localhost")],
        "client": ("1.2.3.4", 1234),
        "server": ("localhost", 8766),
    }
    await app(scope, receive, send)
    headers = {k.decode().lower(): v.decode() for k, v in start.get("headers", [])}
    return start.get("status"), headers, b"".join(chunks)


def asgi_request(app, path: str, *, method: str = "GET") -> tuple[int, dict, object]:
    """One request through the whole app; returns (status, headers, body).

    ``body`` is parsed JSON when the response is JSON, else the text,
    else ``None``.
    """
    status, headers, raw = asyncio.run(_asgi_call(app, path, method=method))
    text = raw.decode("utf-8", errors="replace")
    body: object = None
    if text:
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = text
    return status or 0, headers, body
