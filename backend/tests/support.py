"""Shared test support: assertions, fakes, and a raw-ASGI client.

House style (matches the repo's smoke tests): each ``test_*.py`` file
runs standalone -- ``check()`` prints PASS/FAIL per assertion, and
``finish()`` exits non-zero if anything failed. No pytest dependency.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.config_inspector import (
    ConfigInspector,
    ConfigSummary,
)
from backend.application.ports.run_repository import RunRepository
from backend.application.ports.training_gateway import (
    TrainingGateway,
    TrainingLaunch,
)
from backend.application.errors import ConfigNotFoundError
from backend.application.services import ApplicationServices
from backend.application.supervisor import RunSupervisor
from backend.application.use_cases import (
    DeleteRuns,
    GetActiveRun,
    GetRun,
    GetRunLog,
    ListRuns,
    ReconcileRuns,
    StartTraining,
    StopTraining,
)
from backend.domain.entities.run import Run
from backend.domain.events import DomainEvent
from backend.domain.exceptions import DomainError
from backend.domain.value_objects import RunId, RunStatus
from backend.infrastructure.directory_run_artifacts import DirectoryRunArtifacts
from backend.infrastructure.events.callback_event_bus import CallbackEventBus
from backend.infrastructure.jsonl_progress_source import JsonlProgressSource as _Jsonl
from backend.infrastructure.workspace import WorkspaceLayout

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


def wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll ``predicate`` until true or timeout; for supervisor threads."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


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

    def types(self) -> list[str]:
        return [event.event_type for event in self.published]


class InMemoryRunRepository(RunRepository):
    """Same semantics as the SQLite port, no persistence.

    ``_statuses`` mirrors the persisted status separately from the
    entity objects (which callers mutate *before* writing): the CAS
    must see what the "row" said, not what some in-flight entity copy
    now says.
    """

    _UNFINISHED = (RunStatus.CREATED, RunStatus.RUNNING)

    def __init__(self) -> None:
        self._runs: dict[int, Run] = {}
        self._statuses: dict[int, RunStatus] = {}
        self._next_id = 1

    def add(self, run: Run) -> Run:
        if run.id is not None:
            raise DomainError(f"run already has id {run.id}")
        run.assign_id(RunId(self._next_id))
        self._next_id += 1
        self._runs[run.id] = run
        self._statuses[run.id] = run.status
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
        self._statuses[run.id] = run.status
        return True

    def update_if_status(self, run: Run, expected: RunStatus) -> bool:
        if run.id is None:
            raise DomainError("cannot update an unpersisted run (no id yet)")
        if run.id not in self._runs:
            return False
        if self._statuses.get(run.id) is not expected:
            return False
        self._runs[run.id] = run
        self._statuses[run.id] = run.status
        return True

    def find_active(self) -> Run | None:
        unfinished = [
            r
            for r in self._runs.values()
            if self._statuses.get(r.id) in self._UNFINISHED
        ]
        return max(unfinished, key=lambda r: r.id) if unfinished else None

    def list_unfinished(self) -> list[Run]:
        unfinished = [
            r
            for r in self._runs.values()
            if self._statuses.get(r.id) in self._UNFINISHED
        ]
        unfinished.sort(key=lambda r: r.id, reverse=True)
        return unfinished

    def delete_all(self) -> int:
        deleted = len(self._runs)
        self._runs.clear()
        self._statuses.clear()
        return deleted


class FakeTrainingGateway(TrainingGateway):
    """Scriptable gateway: tests decide what is alive and how it exits.

    spawn() registers the pid as alive with exit code 0; tests kill it
    by discarding the pid from ``alive`` (setting ``exit_codes[pid]``
    first for a non-zero exit). ``spawn_error`` fails the launch.
    """

    def __init__(self) -> None:
        self.spawned: list[TrainingLaunch] = []
        self.stopped: list[tuple[int, bool]] = []
        self.killed: list[int] = []
        self.alive: set[int] = set()
        self.exit_codes: dict[int, int] = {}
        self.spawn_error: Exception | None = None
        self.next_pid = 4242

    def spawn(self, launch: TrainingLaunch) -> int:
        if self.spawn_error is not None:
            raise self.spawn_error
        self.spawned.append(launch)
        pid = self.next_pid
        self.next_pid += 1
        self.alive.add(pid)
        self.exit_codes[pid] = 0
        return pid

    def is_alive(self, pid: int) -> bool:
        return pid in self.alive

    def wait_exit_code(self, pid: int, timeout: float = 5.0) -> int | None:
        if pid in self.alive:
            return None
        return self.exit_codes.get(pid)

    def stop(self, pid: int, *, force: bool = False) -> bool:
        self.stopped.append((pid, force))
        self.alive.discard(pid)
        return True

    def kill(self, pid: int) -> bool:
        self.killed.append(pid)
        if pid not in self.alive:
            return False
        self.alive.discard(pid)
        return True


class FakeConfigInspector(ConfigInspector):
    """Existence check is real; summaries are scripted.

    ``invalid`` maps a path string to an exception to raise (for
    config_invalid tests); everything else resolves to ``default``.
    """

    def __init__(self) -> None:
        self.default = ConfigSummary(mode="distillation", total_steps=100)
        self.summaries: dict[str, ConfigSummary] = {}
        self.invalid: dict[str, Exception] = {}

    def summarize(self, config_path: Path) -> ConfigSummary:
        key = str(config_path)
        if key in self.invalid:
            raise self.invalid[key]
        if not config_path.exists():
            raise ConfigNotFoundError(f"config file not found: {config_path}")
        return self.summaries.get(key, self.default)


def build_services(
    *,
    runs: RunRepository | None = None,
    events: RecordingEventBus | None = None,
    gateway: TrainingGateway | None = None,
    inspector: ConfigInspector | None = None,
    artifacts: DirectoryRunArtifacts | None = None,
    supervisor: RunSupervisor | None = None,
    clock: FakeClock | None = None,
    project_root: Path | None = None,
    runs_dir: Path | None = None,
    poll_interval: float = 0.05,
) -> ApplicationServices:
    """Wire the use cases against fakes (the composition root's twin)."""
    runs = runs if runs is not None else InMemoryRunRepository()
    events = events if events is not None else RecordingEventBus()
    gateway = gateway if gateway is not None else FakeTrainingGateway()
    inspector = inspector if inspector is not None else FakeConfigInspector()
    clock = clock if clock is not None else FakeClock()
    project_root = project_root if project_root is not None else Path(
        tempfile.mkdtemp(prefix="backend-project-")
    )
    runs_dir = runs_dir if runs_dir is not None else Path(
        tempfile.mkdtemp(prefix="backend-runs-")
    )
    if artifacts is None:
        artifacts = DirectoryRunArtifacts(
            WorkspaceLayout(project_root, runs_dir=runs_dir)
        )
    if supervisor is None:
        supervisor = RunSupervisor(
            runs=runs,
            events=events,
            gateway=gateway,
            progress=_Jsonl(),
            artifacts=artifacts,
            clock=clock,
            poll_interval=poll_interval,
        )
    return ApplicationServices(
        list_runs=ListRuns(runs),
        get_run=GetRun(runs),
        delete_runs=DeleteRuns(runs, events),
        get_active_run=GetActiveRun(runs),
        start_training=StartTraining(
            runs=runs,
            events=events,
            gateway=gateway,
            inspector=inspector,
            artifacts=artifacts,
            supervisor=supervisor,
            clock=clock,
            project_root=project_root,
        ),
        stop_training=StopTraining(
            runs=runs, events=events, gateway=gateway, clock=clock
        ),
        get_run_log=GetRunLog(runs=runs, artifacts=artifacts),
        reconcile_runs=ReconcileRuns(
            runs=runs, events=events, gateway=gateway, clock=clock
        ),
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
    pid: int | None = 111,
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
        run.mark_started(pid=pid, at=clock.now())
        repo.update(run)
    return run


# --------------------------------------------------------------------------
# Raw-ASGI client (no httpx dependency, like the legacy smoke tests)
# --------------------------------------------------------------------------


async def _asgi_call(app, path: str, *, method: str = "GET", json_body=None):
    method = method or "GET"
    raw_path, _, query = path.partition("?")
    start: dict = {}
    chunks: list[bytes] = []
    body = b""
    headers = [(b"host", b"localhost")]
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        headers.append((b"content-type", b"application/json"))
        headers.append((b"content-length", str(len(body)).encode()))

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

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
        "headers": headers,
        "client": ("1.2.3.4", 1234),
        "server": ("localhost", 8766),
    }
    await app(scope, receive, send)
    resp_headers = {k.decode().lower(): v.decode() for k, v in start.get("headers", [])}
    return start.get("status"), resp_headers, b"".join(chunks)


def asgi_request(
    app, path: str, *, method: str = "GET", json_body=None
) -> tuple[int, dict, object]:
    """One request through the whole app; returns (status, headers, body).

    ``body`` is parsed JSON when the response is JSON, else the text,
    else ``None``.
    """
    status, headers, raw = asyncio.run(
        _asgi_call(app, path, method=method, json_body=json_body)
    )
    text = raw.decode("utf-8", errors="replace")
    body: object = None
    if text:
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = text
    return status or 0, headers, body
