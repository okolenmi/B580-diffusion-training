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
from backend.infrastructure.jsonl_progress_source import JsonlProgressSource
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


class CrashingGateway(FakeTrainingGateway):
    """First ``is_alive`` probe explodes -- forces an in-loop crash so
    the supervisor's ``_guard`` crash-repair path is exercised (F-01)."""

    def __init__(self) -> None:
        super().__init__()
        self._crashed = False

    def is_alive(self, pid: int) -> bool:
        if not self._crashed:
            self._crashed = True
            raise RuntimeError("injected supervisor crash")
        return super().is_alive(pid)


def _env(tmp: str, *, gateway: FakeTrainingGateway | None = None) -> SimpleNamespace:
    project = Path(tmp) / "project"
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "test.toml").write_text(
        "[common]\nsteps = 100\n", encoding="utf-8"
    )
    runs = Path(tmp) / "runs"
    repo = InMemoryRunRepository()
    events = RecordingEventBus()
    gateway = gateway if gateway is not None else FakeTrainingGateway()
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

    def emit_raw(text: str) -> None:
        """Append bytes as-is (torn lines, garbage -- hostile input)."""
        path = runs / "run_1" / "log.progress.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)

    env.start = start
    env.emit = emit
    env.emit_raw = emit_raw
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


def test_hostile_progress_lines_do_not_break_the_tail() -> None:
    # docs 07 F-01 -- one malformed record must not kill the supervisor
    # thread (the r1 repro: {"step":"n/a"} raised inside _sample).
    print("\n== supervisor: hostile progress lines are survived ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        run_id = env.start()
        env.emit({"phase": "training_start", "total_steps": 100})

        env.emit_raw('{"phase":"step","step":"n/a"}\n')  # r1 repro line
        env.emit_raw("[1, 2, 3]\n")                      # not an object
        env.emit_raw('{"phase":"step","step":-5,"loss":9.9}\n')  # negative step
        env.emit_raw("this is not json\n")               # not JSON
        env.emit({"phase": "step", "step": 4, "total": 100,
                  "loss": 1.25, "avg": 1.3, "lr": 0.0001})

        check(
            wait_until(lambda: env.repo.get(run_id).done_steps == 4),
            "the good line after the garbage still applies",
        )
        run = env.repo.get(run_id)
        check(run.status.value == "running", "supervisor thread survived")
        check(run.current_loss == 1.25, f"loss applied (got {run.current_loss})")
        check(run.done_steps == 4, "malformed step never rewound progress")


def test_torn_progress_line_is_recovered() -> None:
    # docs 07 F-07 -- a record split across two writes must be parsed
    # once complete, not consumed-and-skipped as invalid JSON (r4 lost
    # step 42 this way and its offset stayed past the sample forever).
    print("\n== supervisor: torn line stays buffered until completed ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        run_id = env.start()
        env.emit({"phase": "training_start", "total_steps": 100})

        env.emit_raw('{"phase":"step","step":7,"loss":0.4,"total":100')  # no \n
        time.sleep(0.12)  # several supervisor ticks
        check(
            env.repo.get(run_id).done_steps == 0,
            "torn line not applied yet (and not consumed)",
        )

        env.emit_raw("}\n")  # the record completes
        check(
            wait_until(lambda: env.repo.get(run_id).done_steps == 7),
            "step 7 recovered once the line completed",
        )
        env.emit({"phase": "step", "step": 8, "total": 100, "loss": 0.3})
        check(
            wait_until(lambda: env.repo.get(run_id).done_steps == 8),
            "later lines still flow after the recovery",
        )


def test_final_samples_survive_process_exit() -> None:
    # docs 07 F-07 -- everything written in the death tick must be
    # drained after is_alive turns False, not left behind (r1 showed
    # a completed run reporting 0/100).
    print("\n== supervisor: final samples are drained after exit ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp)
        run_id = env.start()
        env.emit({"phase": "training_start", "total_steps": 100})
        env.emit({"phase": "step", "step": 100, "total": 100,
                  "loss": 0.5, "avg": 0.6, "lr": 0.0001})
        env.gateway.alive.discard(4242)  # process gone before the next tick

        check(
            wait_until(lambda: env.repo.get(run_id).status.value == "completed"),
            "run completes",
        )
        run = env.repo.get(run_id)
        check(run.done_steps == 100, f"final sample drained (got {run.done_steps})")
        check(run.current_loss == 0.5, f"final loss recorded (got {run.current_loss})")


def test_supervisor_crash_repairs_the_row() -> None:
    # docs 07 F-01 -- a crash inside _supervise must finalise the row
    # as failed and unblock the next start (was: row stuck running,
    # every start 409 until restart).
    print("\n== supervisor: crash fails the row and unblocks starts ==")
    with tempfile.TemporaryDirectory() as tmp:
        env = _env(tmp, gateway=CrashingGateway())
        run_id = env.start()

        # _fail_leftover stops the orphan trainer AFTER the row repair;
        # waiting on the stop implies the whole repair ran.
        check(
            wait_until(lambda: env.gateway.stopped == [(4242, False)]),
            "orphan trainer stopped gracefully",
        )
        run = env.repo.get(run_id)
        check(run.status.value == "failed", "crashed supervisor fails the row")
        check(
            run.error is not None and "supervisor crashed" in run.error,
            f"crash recorded on the row (got {run.error!r})",
        )
        check("run_failed" in env.events.types(), "run_failed published")

        # A new start is accepted without restarting the server.
        dto = env.services.start_training.execute(
            StartTrainingCommand(config_path="configs/test.toml")
        )
        check(dto.id == 2, f"retry accepted (got run {dto.id})")
        check(dto.status.value == "running", "retry actually spawned")


def test_terminal_lines_are_evidence_not_telemetry() -> None:
    print("\n== reader: terminal lines carry the trainer's verdict ==")
    path = Path(tempfile.mkdtemp(prefix="progress-")) / "log.progress.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"phase": "step", "step": 5, "loss": 0.5}) + "\n")
        fh.write(json.dumps({"phase": "finished"}) + "\n")
        fh.write(json.dumps({"phase": "something-new"}) + "\n")
    samples = JsonlProgressSource().read_new(path)
    kinds = [(s.step, s.terminal) for s in samples]
    check(
        kinds == [(5, None), (None, "finished")],
        f"a step sample, then the verdict, then nothing (got {kinds})",
    )
    check(
        samples[1].phase is None and samples[1].loss is None,
        "the terminal sample asserts no telemetry",
    )


def _adopted_scenario(terminal: str | None) -> tuple[object, int]:
    """One restart: a running row whose trainer is still alive, adopted."""
    tmp = tempfile.TemporaryDirectory()
    env = _env(tmp.name)
    seed_run(env.repo, env.clock, start=True, pid=4242)  # id 1, still running
    run_id = 1
    env.gateway.alive.add(4242)
    watcher = env.services.start_training._watcher
    watcher.adopt(
        run_id=run_id, pid=4242, progress_path=env.progress_path
    )
    if terminal is not None:
        env.emit({"phase": terminal})
    env.gateway.alive.discard(4242)
    wait_until(
        lambda: env.repo.get(run_id).status.value in {"completed", "failed"}
    )
    return env, run_id


def test_adopted_run_finalises_on_the_trainers_own_word() -> None:
    # docs 07 F-11: an adopted trainer is not our child, so its exit code
    # cannot be read. The trainer's terminal line decides -- and a
    # missing line is a failure, never a hopeful "completed".
    print("\n== supervisor: adopted runs finalise on the trainer's word ==")

    env, run_id = _adopted_scenario("finished")
    run = env.repo.get(run_id)
    check(run.status.value == "completed", f"finished -> completed (got {run.status.value})")
    check(run.exit_code is None, "exit code honestly unknown, not invented")
    check(run.done_steps == 0, "no progress was replayed from the empty history")

    env, run_id = _adopted_scenario("error")
    run = env.repo.get(run_id)
    check(run.status.value == "failed", f"error -> failed (got {run.status.value})")
    check(
        run.error == "trainer reported an error",
        f"the trainer's own reason is kept (got {run.error!r})",
    )

    env, run_id = _adopted_scenario(None)
    run = env.repo.get(run_id)
    check(
        run.status.value == "failed",
        f"no terminal line -> failed, never completed (got {run.status.value})",
    )
    check(
        run.error is not None and "no terminal progress line" in run.error,
        f"and it says why (got {run.error!r})",
    )


def main() -> None:
    test_progress_then_completion()
    test_failed_exit_code()
    test_stop_wins_over_finalise()
    test_total_steps_adoption()
    test_cache_phase_telemetry()
    test_hostile_progress_lines_do_not_break_the_tail()
    test_torn_progress_line_is_recovered()
    test_final_samples_survive_process_exit()
    test_supervisor_crash_repairs_the_row()
    test_terminal_lines_are_evidence_not_telemetry()
    test_adopted_run_finalises_on_the_trainers_own_word()
    finish()


if __name__ == "__main__":
    main()
