"""Unit tests -- RunSupervisor: progress telemetry, finalisation, races.

The supervisor thread runs against a fake gateway + real JSONL reader
over temp files; ``wait_until`` polls until its assertions hold.

Run directly: python backend/tests/test_supervisor.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.dto import StartTrainingCommand
from backend.tests.support import (
    FakeClock,
    FakeConfigInspector,
    FakeTrainingGateway,
    InMemoryRunRepository,
    RecordingEventBus,
    build_services,
    check,
    finish,
    wait_until,
)


def _env(tmp: str) -> SimpleNamespace:
    project = Path(tmp) / "project"
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "test.toml").write_text(
        "[common]\nsteps = 100\n", encoding="utf-8"
    )
    runs = Path(tmp) / "runs"
    repo = InMemoryRunRepository()
    events = RecordingEventBus()
    gateway = FakeTrainingGateway()
    clock = FakeClock()
    services = build_services(
        runs=repo,
        events=events,
        gateway=gateway,
        inspector=FakeConfigInspector(),
        clock=clock,
        project_root=project,
        runs_dir=runs,
        poll_interval=0.02,
    )
    env = SimpleNamespace(
        repo=repo, events=events, gateway=gateway, clock=clock,
        services=services, runs_dir=runs,
    )

    def start() -> int:
        dto = services.start_training.execute(
            StartTrainingCommand(config_path="configs/test.toml")
        )
        return dto.id

    def emit(line: dict) -> None:
        path = runs / "run_1" / "log.progress.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")

    env.start = start
    env.emit = emit
    env.progress_path = runs / "run_1" / "log.progress.jsonl"
    return env


def test_progress_then_completion() -> None:
    print("\n== supervisor: telemetry -> completed ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        run_id = env.start()
        pid = 4242
        check(env.gateway.is_alive(pid), "spawn registered a live pid")

        env.emit({"phase": "training_start", "total_steps": 100})
        env.emit({"phase": "step", "step": 1, "total": 100,
                  "loss": 2.0, "avg": 2.0, "lr": 0.0001})

        ok = wait_until(lambda: env.repo.get(run_id).done_steps == 1)
        check(ok, "first step sample applied")
        run = env.repo.get(run_id)
        check(
            run.phase == "training" and run.current_loss == 2.0,
            f"phase + loss applied (got {run.phase}, {run.current_loss})",
        )
        progressed = [
            e for e in env.events.published
            if type(e).__name__ == "RunProgressed"
        ]
        step_events = [e for e in progressed if e.step == 1]
        check(
            len(step_events) == 1,
            f"one RunProgressed for step 1 (got {len(step_events)})",
        )
        check(
            step_events
            and step_events[0].loss == 2.0
            and step_events[0].lr == 0.0001
            and step_events[0].total_steps == 100,
            "event carries step/loss/lr/total",
        )

        env.emit({"phase": "step", "step": 2, "total": 100,
                  "loss": 1.5, "avg": 1.75, "lr": 0.0001})
        check(
            wait_until(lambda: env.repo.get(run_id).done_steps == 2),
            "second step sample applied",
        )

        env.gateway.alive.discard(pid)  # exit code 0 from spawn
        log_file = env.runs_dir / "run_1" / "log.txt"
        # The marker is finalize's LAST action (after status + events),
        # so waiting on it also settles those assertions.
        check(
            wait_until(
                lambda: log_file.exists() and "RUN ENDED" in log_file.read_text()
            ),
            "post-exit marker appended to the log",
        )
        run = env.repo.get(run_id)
        check(run.status.value == "completed", "zero exit -> completed")
        check("run_completed" in env.events.types(), "run_completed published")
        check(
            "--- RUN ENDED: status=completed, exit_code=0" in log_file.read_text(),
            "marker content is exact",
        )


def test_failed_exit_code() -> None:
    print("\n== supervisor: non-zero exit -> failed ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        run_id = env.start()
        pid = 4242
        env.gateway.exit_codes[pid] = 1
        env.gateway.alive.discard(pid)
        log_file = env.runs_dir / "run_1" / "log.txt"
        check(
            wait_until(
                lambda: log_file.exists() and "RUN ENDED" in log_file.read_text()
            ),
            "failure marker appended to the log",
        )
        run = env.repo.get(run_id)
        check(run.status.value == "failed", "non-zero exit -> failed")
        check(run.error == "Exit code 1", f"error text (got {run.error})")
        check(run.exit_code == 1, "exit code recorded")
        check("run_failed" in env.events.types(), "run_failed published")
        check(
            "--- RUN ENDED: status=failed, exit_code=1" in log_file.read_text(),
            "marker content is exact",
        )


def test_stop_wins_over_finalise() -> None:
    print("\n== supervisor: stop request beats process-death finalisation ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        run_id = env.start()
        pid = 4242
        env.gateway.exit_codes[pid] = -9  # killed by our signal

        dto = env.services.stop_training.execute(run_id)
        check(dto.status.value == "cancelled", "stop marked the run cancelled")

        # The child dies from the signal; the supervisor must not
        # overwrite the cancellation with failed(-9).
        env.gateway.alive.discard(pid)
        time.sleep(0.15)  # several supervisor ticks
        run = env.repo.get(run_id)
        check(run.status.value == "cancelled", "status stays cancelled")
        check(
            "run_completed" not in env.events.types()
            and "run_failed" not in env.events.types(),
            "no terminal event from the supervisor",
        )
        check(
            env.events.types().count("run_cancelled") == 1,
            "exactly one run_cancelled",
        )


def test_total_steps_adoption() -> None:
    print("\n== supervisor: trainer may discover a larger total ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        run_id = env.start()
        env.emit({"phase": "training_start", "total_steps": 500})
        check(
            wait_until(lambda: env.repo.get(run_id).total_steps == 500),
            "larger total adopted",
        )
        env.emit({"phase": "step", "step": 1, "total": 500,
                  "loss": 2.0, "avg": 2.0, "lr": 0.0001})
        check(
            wait_until(
                lambda: any(
                    getattr(e, "total_steps", None) == 500
                    and type(e).__name__ == "RunProgressed"
                    for e in env.events.published
                )
            ),
            "telemetry reports the adopted total",
        )
        # A smaller total must NOT shrink the row.
        env.emit({"phase": "step", "step": 2, "total": 10,
                  "loss": 1.9, "avg": 1.9, "lr": 0.0001})
        check(
            wait_until(lambda: env.repo.get(run_id).done_steps == 2),
            "smaller-total sample applied",
        )
        check(env.repo.get(run_id).total_steps == 500, "total never shrinks")


def test_cache_phase_telemetry() -> None:
    print("\n== supervisor: cache phase counters ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        run_id = env.start()
        env.emit({"phase": "cache_start", "est_trajs": 1000})
        check(
            wait_until(
                lambda: env.repo.get(run_id).cache_total == 1000
                and env.repo.get(run_id).cache_done == 0
            ),
            "cache_start sets the estimate",
        )
        check(env.repo.get(run_id).phase == "cache", "phase is cache")
        check(env.repo.get(run_id).done_steps == 0, "no fake step progress")

        env.emit({"phase": "cache", "done": 500, "total": 1000})
        check(
            wait_until(lambda: env.repo.get(run_id).cache_done == 500),
            "cache progress applied",
        )
        env.emit({"phase": "cache_done", "total": 1000})
        check(
            wait_until(lambda: env.repo.get(run_id).cache_done == 1000),
            "cache completion applied",
        )


def main() -> None:
    test_progress_then_completion()
    test_failed_exit_code()
    test_stop_wins_over_finalise()
    test_total_steps_adoption()
    test_cache_phase_telemetry()
    finish()


if __name__ == "__main__":
    main()
