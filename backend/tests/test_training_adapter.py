"""Integration tests -- real CoreConfigInspector and the real command
builder (``SubprocessTrainingGateway._build_command``) against
``core.config_io``/``core.config_model``.

No training process is ever spawned: only argv construction and the
launch-failure contract (Popen errors -> TrainingLaunchError) are
exercised, with ``VENV_PYTHON`` pointed at a non-existent interpreter.

Run directly: python backend/tests/test_training_adapter.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.config_io import write_config
from core.config_model import TrainingConfig

from backend.application.errors import (
    ConfigInvalidError,
    ConfigNotFoundError,
    TrainingLaunchError,
)
from backend.application.ports.training_gateway import TrainingLaunch
from backend.infrastructure.core_config_inspector import CoreConfigInspector
from backend.infrastructure.subprocess_gateway import SubprocessTrainingGateway
from backend.infrastructure.workspace import WorkspaceLayout
from backend.tests.support import check, finish

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_config(directory: str, name: str = "cfg.toml") -> Path:
    path = Path(directory) / name
    write_config(path, TrainingConfig())
    return path


def _launch(config_path: Path, tmp: str, **overrides) -> TrainingLaunch:
    runs = Path(tmp) / "runs"
    values = dict(
        run_id=7,
        config_path=config_path,
        mode="distillation",
        total_steps=1000,
        start_from="teacher",
        reset_optimizer=False,
        log_path=runs / "run_7" / "log.txt",
        progress_path=runs / "run_7" / "log.progress.jsonl",
    )
    values.update(overrides)
    return TrainingLaunch(**values)


def test_real_inspector() -> None:
    print("\n== CoreConfigInspector over the real config model ==")
    with tempfile.TemporaryDirectory() as tmp:
        inspector = CoreConfigInspector(WorkspaceLayout(Path(tmp)))
        good = _write_config(tmp)
        summary = inspector.summarize(good)
        check(
            summary.mode == "distillation" and summary.total_steps == 1000,
            f"default config summarised (got {summary})",
        )

        try:
            inspector.summarize(Path(tmp) / "missing.toml")
            check(False, "missing config must raise")
        except ConfigNotFoundError as exc:
            check(exc.code == "config_not_found", "FileNotFoundError mapped to 404 code")

        garbage = Path(tmp) / "garbage.toml"
        garbage.write_text("this is { not toml", encoding="utf-8")
        try:
            inspector.summarize(garbage)
            check(False, "garbage config must raise")
        except ConfigInvalidError as exc:
            check(exc.code == "config_invalid", "parse failure mapped to 422 code")


def test_command_builder_flags() -> None:
    print("\n== SubprocessTrainingGateway._build_command (real config) ==")
    with tempfile.TemporaryDirectory() as tmp:
        config_path = _write_config(tmp)
        layout = WorkspaceLayout(_PROJECT_ROOT, runs_dir=Path(tmp) / "runs")
        gateway = SubprocessTrainingGateway(layout)

        teacher = gateway._build_command(_launch(config_path, tmp))
        check(teacher[0] == layout.venv_python, f"venv interpreter first ({teacher[0]})")
        check(
            teacher[1:4] == ["-m", "core.cli", "--config"],
            f"core.cli entry (got {teacher[1:4]})",
        )
        check(str(config_path) in teacher, "--config points at the config")
        check("--steps" in teacher and "1000" in teacher, "--steps forwarded")
        check("--run-id" in teacher and "7" in teacher, "--run-id forwarded")
        check("--fresh" in teacher, "teacher start_from -> --fresh")
        check("--reset-optimizer" not in teacher, "no reset by default")

        student = gateway._build_command(
            _launch(config_path, tmp, start_from="student")
        )
        check("--fresh" in student, "student start_from -> --fresh")
        # --student only when the config declares a student path.
        from core.config_io import read_config

        if read_config(config_path).paths.student:
            check("--student" in student, "config student path forwarded")
        else:
            check("--student" not in student, "no --student for an empty path")

        resume = gateway._build_command(
            _launch(config_path, tmp, start_from="resume")
        )
        check(
            "--start-from" in resume and "resume" in resume,
            "resume start_from -> --start-from resume",
        )
        check("--fresh" not in resume, "resume does not pass --fresh")

        lora = gateway._build_command(
            _launch(config_path, tmp, start_from="lora_checkpoint")
        )
        check("--fresh" in lora, "lora_checkpoint -> --fresh")

        reset = gateway._build_command(
            _launch(config_path, tmp, reset_optimizer=True)
        )
        check("--reset-optimizer" in reset, "reset_optimizer flag appended")


def test_spawn_wraps_launch_failures() -> None:
    print("\n== spawn raises TrainingLaunchError, never OSError ==")
    with tempfile.TemporaryDirectory() as tmp:
        config_path = _write_config(tmp)
        layout = WorkspaceLayout(_PROJECT_ROOT, runs_dir=Path(tmp) / "runs")
        gateway = SubprocessTrainingGateway(layout)
        launch = _launch(config_path, tmp)

        saved = os.environ.get("VENV_PYTHON")
        os.environ["VENV_PYTHON"] = "/definitely/not/a/real/interpreter"
        try:
            gateway.spawn(launch)
            check(False, "spawn with a bogus interpreter must raise")
        except TrainingLaunchError as exc:
            check(exc.code == "training_launch_failed", "wrapped in the contract code")
            check(
                "failed to launch trainer" in str(exc),
                f"message names the failure (got {exc})",
            )
        finally:
            if saved is None:
                os.environ.pop("VENV_PYTHON", None)
            else:
                os.environ["VENV_PYTHON"] = saved

        check(
            gateway.wait_exit_code(999999) is None,
            "unknown pid has no exit code",
        )


def main() -> None:
    test_real_inspector()
    test_command_builder_flags()
    test_spawn_wraps_launch_failures()
    finish()


if __name__ == "__main__":
    main()
