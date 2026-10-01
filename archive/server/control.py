"""Control layer — inspects state, offers options, builds commands.

This module uses the new TrainingConfig model with discriminated unions.
start_from is a per-launch UI concept, NOT stored in the config file.
"""

from pathlib import Path
from typing import Optional

from . import db
from .config import settings
from core.config_io import read_config
from core.config_model import TrainingConfig


def scan_checkpoints(config: TrainingConfig) -> dict:
    """Inspect config and filesystem to determine available starting points."""
    from paths import resolve_model_path

    base_path = config.paths.base_model or ""
    student_path = config.paths.student or ""
    lora_path = config.tuning.lora_output if config.tuning.method == "lora" else ""

    result = {
        "teacher": {
            "available": bool(base_path),
            "path": base_path,
            "label": "Base Model",
        },
        "student": {
            "available": bool(student_path),
            "path": student_path,
            "label": "Student",
        },
        "resume": {
            "available": False,
            "path": "",
            "label": "Resume",
        },
    }

    # Resume checkpoint: config.paths.resume_checkpoint is already correctly
    # derived by TrainingConfig's model validator (same stem as
    # checkpoint_output/lora_output, placed in the dedicated resume/
    # subfolder under checkpoints_dir/loras_dir -- see
    # core/config_model.py's _fill_resume_paths/_derive_path). Just check
    # whether that file actually exists yet.
    resume_path = config.paths.resume_checkpoint or ""
    if resume_path and Path(resume_path).exists():
        result["resume"] = {
            "available": True,
            "path": resume_path,
            "label": "Resume",
        }
    # If checkpoint_output/lora_output isn't set at all, there's no derived
    # resume_checkpoint to check -- and no way to guess which file in the
    # resume/ subfolder (if any) was meant for this run, since that
    # directory can hold resume files for many different runs at once. Left
    # as unavailable rather than guessing.

    # LoRA checkpoint
    if config.tuning.method == "lora" and lora_path:
        p_lora = resolve_model_path(lora_path, "lora")
        result["lora_checkpoint"] = {
            "available": p_lora.exists(),
            "path": lora_path if p_lora.exists() else "",
            "label": "LoRA Checkpoint",
        }

    return result


def get_control_options(config: TrainingConfig, db_path: Path) -> dict:
    """Return available control options for the UI."""
    checkpoints = scan_checkpoints(config)

    active = db.get_active_run(db_path)
    has_unfinished = active is not None and active["status"] == "running"

    last_finished = None
    runs = db.list_runs(db_path, limit=10)
    for r in runs:
        if r["status"] in ("finished", "stopped", "failed", "killed"):
            last_finished = {
                "id": r["id"],
                "config": r["config_path"],
                "mode": r["mode"],
                "done_steps": r["done_steps"],
                "total_steps": r["total_steps"],
                "avg_loss": r.get("avg_loss"),
                "status": r["status"],
            }
            break

    return {
        "start_from": checkpoints,
        "has_unfinished_run": has_unfinished,
        "last_finished": last_finished,
    }


def build_training_command(
    config: TrainingConfig,
    config_path: str,
    start_from: str = "teacher",
    reset_optimizer: bool = False,
    total_steps: int = 0,
    run_id: int = 0,
) -> list[str]:
    """Build CLI command from config + launch options.

    The config is already written to TOML; this builds flags that
    control the launch mode (fresh/student/resume/lora).
    """
    cmd = [
        settings.venv_python, "-m", "core.cli",
        "--config", config_path,
    ]

    if total_steps > 0:
        cmd.extend(["--steps", str(total_steps)])

    if run_id > 0:
        cmd.extend(["--run-id", str(run_id)])

    if start_from == "teacher":
        cmd.append("--fresh")
    elif start_from == "student":
        cmd.append("--fresh")
        student_path = config.paths.student or ""
        if student_path:
            cmd.extend(["--student", student_path])
    elif start_from == "lora_checkpoint":
        cmd.append("--fresh")
    elif start_from == "resume":
        cmd.extend(["--start-from", "resume"])
        resume_checkpoint = config.paths.resume_checkpoint or ""
        if resume_checkpoint and Path(resume_checkpoint).exists():
            cmd.extend(["--student", resume_checkpoint])
        if not reset_optimizer:
            resume_optimizer = config.paths.resume_optimizer or ""
            if resume_optimizer and Path(resume_optimizer).exists():
                cmd.extend(["--resume-optimizer", resume_optimizer])

    if reset_optimizer:
        cmd.append("--reset-optimizer")

    return cmd
