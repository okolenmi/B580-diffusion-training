"""Unit tests -- StartTraining / StopTraining / GetActiveRun /
GetRunLog / ReconcileRuns (fakes for gateway/inspector, in-memory repo).

Run directly: python backend/tests/test_start_stop.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.dto import StartTrainingCommand
from backend.application.errors import (
    ConfigInvalidError,
    ConfigNotFoundError,
    InvalidQueryError,
    NoActiveRunError,
    RunAlreadyActiveError,
    RunNotFoundError,
    RunNotRunningError,
    TrainingLaunchError,
)
from backend.application.ports.run_artifacts import RunArtifactsPaths
from backend.tests.support import (
    FakeClock,
    FakeConfigInspector,
    FakeTrainingGateway,
    InMemoryRunRepository,
    RecordingEventBus,
    build_services,
    check,
    finish,
    seed_run,
)


def _env(tmp: str, artifacts: object | None = None) -> SimpleNamespace:
    project = Path(tmp) / "project"
    (project / "configs").mkdir(parents=True)
    runs = Path(tmp) / "runs"
    config = project / "configs" / "test.toml"
    config.write_text("[common]\nsteps = 100\n", encoding="utf-8")

    repo = InMemoryRunRepository()
    events = RecordingEventBus()
    gateway = FakeTrainingGateway()
    inspector = FakeConfigInspector()
    clock = FakeClock()
    services = build_services(
        runs=repo,
        events=events,
        gateway=gateway,
        inspector=inspector,
        clock=clock,
        project_root=project,
        runs_dir=runs,
        poll_interval=0.02,
        artifacts=artifacts,  # type: ignore[arg-type]
    )
    return SimpleNamespace(
        project=project,
        runs_dir=runs,
        config=config,
        repo=repo,
        events=events,
        gateway=gateway,
        inspector=inspector,
        clock=clock,
        services=services,
    )


def test_start_happy_path() -> None:
    print("\n== StartTraining: happy path ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        dto = env.services.start_training.execute(
            StartTrainingCommand(config_path="configs/test.toml")
        )
        check(dto.id == 1 and dto.status.value == "running", "run created + started")
        check(
            dto.mode == "distillation" and dto.total_steps == 100,
            "config summary landed on the run",
        )
        check(
            env.events.types() == ["run_created", "run_started"],
            f"events run_created, run_started (got {env.events.types()})",
        )
        check(len(env.gateway.spawned) == 1, "gateway spawned exactly once")
        launch = env.gateway.spawned[0]
        check(
            launch.config_path == env.config,
            f"absolute config path passed ({launch.config_path})",
        )
        check(
            launch.log_path == env.runs_dir / "run_1" / "log.txt",
            "log path follows run_<id>/log.txt convention",
        )
        check(
            launch.progress_path
            == env.runs_dir / "run_1" / "log.progress.jsonl",
            "progress path matches the child's convention",
        )
        check(
            launch.start_from == "teacher" and launch.reset_optimizer is False,
            "launch options forwarded",
        )
        check((env.runs_dir / "run_1").is_dir(), "run directory prepared")
        check(
            env.services.get_active_run.execute().id == 1,
            "get_active_run sees the running run",
        )

        try:
            env.services.start_training.execute(
                StartTrainingCommand(config_path="configs/test.toml")
            )
            check(False, "second start must be rejected")
        except RunAlreadyActiveError as exc:
            check(exc.code == "run_already_active", "409 conflict code")
            check(len(env.gateway.spawned) == 1, "no second spawn")


def test_start_config_errors() -> None:
    print("\n== StartTraining: config validation ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        try:
            env.services.start_training.execute(
                StartTrainingCommand(config_path="missing.toml")
            )
            check(False, "missing config must be rejected")
        except ConfigNotFoundError as exc:
            check(exc.code == "config_not_found", "404 config_not_found code")

        bad = env.project / "configs" / "bad.toml"
        bad.write_text("not a real config", encoding="utf-8")
        env.inspector.invalid[str(bad)] = ConfigInvalidError("unparseable")
        try:
            env.services.start_training.execute(
                StartTrainingCommand(config_path="configs/bad.toml")
            )
            check(False, "invalid config must be rejected")
        except ConfigInvalidError as exc:
            check(exc.code == "config_invalid", "422 config_invalid code")

        try:
            env.services.start_training.execute(
                StartTrainingCommand(
                    config_path="configs/test.toml", start_from="nowhere"
                )
            )
            check(False, "unknown start_from must be rejected")
        except InvalidQueryError as exc:
            check(exc.code == "invalid_query", "422 invalid_query code")

        check(env.repo.list() == [], "no run rows created by failed starts")
        check(env.gateway.spawned == [], "nothing spawned")


def test_start_spawn_failure_finalises_run() -> None:
    print("\n== StartTraining: spawn failure finalises the row ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        env.gateway.spawn_error = TrainingLaunchError("venv python missing")
        try:
            env.services.start_training.execute(
                StartTrainingCommand(config_path="configs/test.toml")
            )
            check(False, "launch failure must propagate")
        except TrainingLaunchError as exc:
            check(exc.code == "training_launch_failed", "500 launch code")

        run = env.repo.get(1)
        check(run is not None and run.status.value == "failed", "row is failed")
        check(run.error == "venv python missing", "error recorded on the row")
        check(
            env.events.types() == ["run_created", "run_failed"],
            f"events recorded the failure (got {env.events.types()})",
        )

        # The lock must not be wedged: a later start succeeds.
        env.gateway.spawn_error = None
        dto = env.services.start_training.execute(
            StartTrainingCommand(config_path="configs/test.toml")
        )
        check(dto.id == 2 and dto.status.value == "running", "retry succeeds")


class FlakyArtifacts:
    """``prepare()`` raises for the first run only -- the docs 07 F-02
    repro (PermissionError after the row insert), then behaves."""

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir
        self._fail_first = True

    def _paths(self, run_id: int) -> RunArtifactsPaths:
        root = self._runs_dir / f"run_{run_id}"
        return RunArtifactsPaths(
            directory=root,
            log=root / "log.txt",
            progress=root / "log.progress.jsonl",
        )

    def prepare(self, run_id: int) -> RunArtifactsPaths:
        if self._fail_first:
            self._fail_first = False
            raise PermissionError("runs directory is read-only (injected)")
        paths = self._paths(run_id)
        paths.directory.mkdir(parents=True, exist_ok=True)
        return paths

    def paths_for(self, run_id: int) -> RunArtifactsPaths:
        return self._paths(run_id)

    def append_log_note(self, run_id: int, note: str) -> None:
        paths = self._paths(run_id)
        paths.log.parent.mkdir(parents=True, exist_ok=True)
        with open(paths.log, "a", encoding="utf-8") as fh:
            fh.write(note + "\n")


def test_start_prepare_failure_finalises_row() -> None:
    # docs 07 F-02 -- prepare() raising after the row insert must not
    # strand the row at `created` (was: find_active() counted the ghost
    # forever, every start 409 until restart).
    print("\n== StartTraining: prepare() failure does not strand the row ==")
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = FlakyArtifacts(Path(tmp) / "runs")
        env = _env(tmp, artifacts=artifacts)
        try:
            env.services.start_training.execute(
                StartTrainingCommand(config_path="configs/test.toml")
            )
            check(False, "prepare() failure must propagate")
        except PermissionError:
            check(True, "prepare() failure propagates to the caller")

        run = env.repo.get(1)
        check(
            run is not None and run.status.value == "failed",
            "row is failed, not stranded at created",
        )
        check(
            run is not None and run.error is not None and "read-only" in run.error,
            f"error recorded on the row (got {run.error!r})",
        )
        check(
            env.events.types() == ["run_created", "run_failed"],
            f"events recorded the failure (got {env.events.types()})",
        )
        check(env.gateway.spawned == [], "nothing was spawned")

        # A second start is accepted -- no ghost blocks it.
        dto = env.services.start_training.execute(
            StartTrainingCommand(config_path="configs/test.toml")
        )
        check(dto.id == 2, f"retry accepted (got run {dto.id})")
        check(dto.status.value == "running", "retry actually spawned")


def test_stop_training() -> None:
    print("\n== StopTraining ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        seed_run(env.repo, env.clock, start=True)  # pid 111
        dto = env.services.stop_training.execute(1, force=False)
        check(dto.status.value == "cancelled", "running run cancelled")
        check(dto.error == "stop requested", "cancellation reason recorded")
        check(
            env.gateway.stopped == [(111, False)],
            f"graceful stop signalled (got {env.gateway.stopped})",
        )
        check(
            "run_cancelled" in env.events.types(),
            "run_cancelled published",
        )
        cancelled = next(
            e for e in env.events.published if type(e).__name__ == "RunCancelled"
        )
        check(cancelled.reason == "stop requested", "event carries the reason")

        try:
            env.services.stop_training.execute(1)
            check(False, "stopping a cancelled run must be rejected")
        except RunNotRunningError as exc:
            check(exc.code == "run_not_running", "409 run_not_running code")

        try:
            env.services.stop_training.execute(99)
            check(False, "stopping an unknown run must be rejected")
        except RunNotFoundError as exc:
            check(exc.code == "run_not_found", "404 for unknown id")

        seed_run(env.repo, env.clock, start=False)  # created, never started
        try:
            env.services.stop_training.execute(2)
            check(False, "stopping a created run must be rejected")
        except RunNotRunningError:
            check(True, "created run cannot be stopped")

        seed_run(env.repo, env.clock, start=True, pid=222)
        dto = env.services.stop_training.execute(3, force=True)
        check(dto.error == "stop requested (force)", "force reason recorded")
        check(
            env.gateway.stopped[-1] == (222, True),
            "force flag forwarded to the gateway",
        )


def test_get_active_run() -> None:
    print("\n== GetActiveRun ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        try:
            env.services.get_active_run.execute()
            check(False, "no active run must raise")
        except NoActiveRunError as exc:
            check(exc.code == "no_active_run", "404 no_active_run code")

        created = seed_run(env.repo, env.clock, start=False)
        dto = env.services.get_active_run.execute()
        check(dto.id == created.id, "created run counts as active")

        created.mark_started(pid=5, at=env.clock.now())
        env.repo.update(created)
        dto = env.services.get_active_run.execute()
        check(dto.status.value == "running", "running run is active")

        created.mark_completed(at=env.clock.now())
        env.repo.update(created)
        try:
            env.services.get_active_run.execute()
            check(False, "completed run is not active")
        except NoActiveRunError:
            check(True, "no active run once terminal")


def test_get_run_log() -> None:
    print("\n== GetRunLog ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        try:
            env.services.get_run_log.execute(1)
            check(False, "unknown run must be rejected")
        except RunNotFoundError:
            check(True, "unknown run -> 404")

        seed_run(env.repo, env.clock, start=True)
        result = env.services.get_run_log.execute(1)
        check(result.log == "", "missing log file yields empty string")

        log_path = env.runs_dir / "run_1" / "log.txt"
        log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
        result = env.services.get_run_log.execute(1, lines=2)
        check(
            result.log == "line2\nline3\n",
            f"tail honours lines (got {result.log!r})",
        )

        for bad in (0, 501):
            try:
                env.services.get_run_log.execute(1, lines=bad)
                check(False, f"lines={bad} must be rejected")
            except InvalidQueryError:
                check(True, f"lines={bad} rejected")


def test_reconcile_runs() -> None:
    print("\n== ReconcileRuns (startup sweep) ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        result = env.services.reconcile_runs.execute()
        check(result.cleaned == 0, "empty database sweeps nothing")

        seed_run(env.repo, env.clock, start=False)  # id 1: abandoned mid-launch

        orphan = seed_run(env.repo, env.clock, start=True, pid=777)  # id 2
        env.gateway.alive.add(777)  # leftover process still running

        gone = seed_run(env.repo, env.clock, start=True, pid=888)  # id 3
        check(888 not in env.gateway.alive, "pid 888 is already gone")

        seed_run(env.repo, env.clock, start=True, pid=None)  # id 4: no pid

        result = env.services.reconcile_runs.execute()
        check(result.cleaned == 4, f"all four rows finalised (got {result.cleaned})")

        check(env.repo.get(1).status.value == "failed", "created -> failed")
        check(
            env.repo.get(1).error == "server stopped before the run launched",
            "mid-launch failure reason",
        )
        check(env.repo.get(2).status.value == "cancelled", "live orphan cancelled")
        check(
            env.repo.get(2).error is not None
            and "orphan cleanup" in env.repo.get(2).error,
            "orphan reason recorded",
        )
        check(env.repo.get(3).status.value == "failed", "dead orphan -> failed")
        check(env.repo.get(4).status.value == "failed", "pidless run -> failed")
        check(env.repo.find_active() is None, "nothing active after the sweep")
        # Newest-first sweep: id 3 (gone) attempted before id 2 (alive).
        check(
            env.gateway.killed == [888, 777],
            f"kill attempted for both pid'd runs (got {env.gateway.killed})",
        )

        types = env.events.types()
        check(types.count("run_failed") == 3, "three run_failed events")
        check(types.count("run_cancelled") == 1, "one run_cancelled event")

        # Second sweep is a no-op.
        check(
            env.services.reconcile_runs.execute().cleaned == 0,
            "sweep is idempotent",
        )


def main() -> None:
    test_start_happy_path()
    test_start_config_errors()
    test_start_spawn_failure_finalises_run()
    test_start_prepare_failure_finalises_row()
    test_stop_training()
    test_get_active_run()
    test_get_run_log()
    test_reconcile_runs()
    finish()


if __name__ == "__main__":
    main()
