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
    RunDirectoryCollisionError,
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
    wait_until,
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

    def tail_log(self, run_id: int, lines: int) -> str:
        paths = self._paths(run_id)
        try:
            text = paths.log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "".join(text.splitlines(keepends=True)[-lines:])


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


def test_run_log_tail_is_read_from_the_end() -> None:
    # docs 07 F-14: the tail is sliced by walking backwards, so the
    # boundary cases matter -- a log much larger than one scan block, and
    # a last line without a trailing newline.
    print("\n== GetRunLog: backwards tail over a big log ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        seed_run(env.repo, env.clock, start=True)
        log_path = env.runs_dir / "run_1" / "log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = 5000
        with open(log_path, "w", encoding="utf-8") as fh:
            for i in range(lines):
                fh.write(f"line {i} {'x' * 200}\n")
            fh.write("last line, no newline")
        size = log_path.stat().st_size
        check(size > 1_000_000, f"the fixture log is big (got {size} bytes)")

        result = env.services.get_run_log.execute(1, lines=3)
        check(
            result.log == f"line {lines - 2} {'x' * 200}\nline {lines - 1} {'x' * 200}\n"
                          f"last line, no newline",
            f"the last three lines come back (got {result.log[:60]!r}...)",
        )
        check("line 0" not in result.log, "and nothing from the head of the file")
        whole = env.services.get_run_log.execute(1, lines=500)
        check(len(whole.log.splitlines()) == 500, f"max page size (got {len(whole.log.splitlines())})")


def test_reconcile_runs() -> None:
    print("\n== ReconcileRuns (startup sweep, adoption included) ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        result = env.services.reconcile_runs.execute()
        check(result.cleaned == 0 and result.adopted == 0, "empty database sweeps nothing")

        seed_run(env.repo, env.clock, start=False)  # id 1: abandoned mid-launch

        survivor = seed_run(env.repo, env.clock, start=True, pid=777)  # id 2
        env.gateway.alive.add(777)  # still training when the server died

        gone = seed_run(env.repo, env.clock, start=True, pid=888)  # id 3
        check(888 not in env.gateway.alive, "pid 888 is already gone")

        stranger = seed_run(env.repo, env.clock, start=True, pid=999)  # id 4
        env.gateway.alive.add(999)  # a live process...
        env.gateway.foreign.add(999)  # ...that is not our trainer (pid reuse)

        seed_run(env.repo, env.clock, start=True, pid=None)  # id 5: no pid

        result = env.services.reconcile_runs.execute()
        check(result.cleaned == 4, f"four rows finalised (got {result.cleaned})")
        check(result.adopted == 1, f"one run adopted (got {result.adopted})")

        check(env.repo.get(1).status.value == "failed", "created -> failed")
        check(
            env.repo.get(1).error == "server stopped before the run launched",
            "mid-launch failure reason",
        )
        check(env.repo.get(3).status.value == "failed", "dead orphan -> failed")
        check(env.repo.get(4).status.value == "failed", "a stranger is never adopted")
        check(999 in env.gateway.alive, "and the stranger was never signalled")
        check(env.repo.get(5).status.value == "failed", "pidless run -> failed")

        # The trainer that was still alive keeps running, and the server
        # is watching it again (docs 07 F-11).
        check(env.repo.get(2).status.value == "running", "live trainer stays running")
        check(
            env.gateway.killed == [999, 888],
            f"kill attempted only where no adoption applied (got {env.gateway.killed})",
        )
        check(survivor.id == 2, "sanity: the adopted row is the survivor")
        progress = env.runs_dir / "run_2" / "log.progress.jsonl"
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(
            '{"phase":"step","step":3,"total":10,"loss":0.5}\n', encoding="utf-8"
        )
        wait_until(lambda: env.repo.get(2).done_steps == 3)
        check(
            env.repo.get(2).done_steps == 3,
            "the adopted run reports progress again",
        )
        log = (env.runs_dir / "run_2" / "log.txt").read_text(encoding="utf-8")
        check("REAPTIED" in log, f"the re-attachment is recorded in the log (got {log!r})")

        types = env.events.types()
        check(types.count("run_failed") == 4, "four run_failed events")
        check(types.count("run_cancelled") == 0, "no run was cancelled out from under a trainer")

        # Second sweep leaves the adopted row alone (it is still running).
        again = env.services.reconcile_runs.execute()
        check(
            again.adopted == 1 and env.repo.get(2).status.value == "running",
            f"sweep is idempotent for an adopted run (got {again})",
        )


def test_start_refuses_an_occupied_run_directory() -> None:
    # docs 07 F-04: the trainer opens log.txt with "w", so a run id whose
    # directory already holds another run's files must be refused, not
    # written over. Startup seeds ids above existing run dirs; this is the
    # net for anything that appeared afterwards.
    print("\n== StartTraining: an occupied runs/run_<id>/ is refused ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        occupied = env.runs_dir / "run_1"
        occupied.mkdir(parents=True)
        legacy_log = occupied / "log.txt"
        legacy_log.write_text("legacy run line\n" * 200, encoding="utf-8")
        before = legacy_log.stat().st_size

        try:
            env.services.start_training.execute(
                StartTrainingCommand(config_path="configs/test.toml")
            )
            check(False, "an occupied run directory must refuse the start")
        except RunDirectoryCollisionError as exc:
            check(exc.code == "run_directory_conflict", "409 run_directory_conflict")

        check(legacy_log.stat().st_size == before, "the legacy log was not truncated")
        check(
            legacy_log.read_text(encoding="utf-8") == "legacy run line\n" * 200,
            "its content is byte-for-byte intact",
        )
        check(env.gateway.spawned == [], "nothing was spawned")

        # The refused start still finalised its row (F-02 repair), so the
        # next attempt is not blocked by a ghost.
        row = env.repo.get(1)
        check(row is not None and row.status.value == "failed", "row is failed")
        check("run_1" in (row.error or ""), f"error names the directory (got {row.error!r})")

        # Operator moves the old run aside -> the next start proceeds.
        occupied.rename(env.runs_dir / "run_1_legacy")
        dto = env.services.start_training.execute(
            StartTrainingCommand(config_path="configs/test.toml")
        )
        check(dto.id == 2 and dto.status.value == "running", "start accepted once moved")
        check((env.runs_dir / "run_2" / "log.progress.jsonl").parent.is_dir(),
              "the new run gets its own directory")


def main() -> None:
    test_start_happy_path()
    test_start_config_errors()
    test_start_spawn_failure_finalises_run()
    test_start_prepare_failure_finalises_row()
    test_start_refuses_an_occupied_run_directory()
    test_stop_training()
    test_get_active_run()
    test_get_run_log()
    test_run_log_tail_is_read_from_the_end()
    test_reconcile_runs()
    finish()


if __name__ == "__main__":
    main()
