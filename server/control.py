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

    # Scan for resume checkpoint
    output_path = config.paths.checkpoint_output or ""
    if config.tuning.method == "lora" and getattr(config.tuning, "lora_output", None):
        output_path = config.tuning.lora_output

    if output_path:
        p_out = Path(output_path)
        out_dir = p_out.parent
        if not out_dir.is_absolute():
            out_dir = settings.comfy_dir / out_dir
        candidate = out_dir / (p_out.stem + ".resume.safetensors")
        if candidate.exists():
            result["resume"] = {
                "available": True,
                "path": str(candidate),
                "label": "Resume",
            }

    # Fallback: check config directory for .resume.safetensors
    if not result["resume"]["available"]:
        # We don't have the config path here, so skip directory scan
        pass

    # LoRA checkpoint
    if config.tuning.method == "lora" and lora_path:
        p_lora = Path(lora_path)
        if not p_lora.is_absolute():
            p_lora = settings.comfy_dir / p_lora
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
        student_path = config.paths.student or ""
        if student_path:
            cmd.extend(["--student", student_path])
    elif start_from == "lora_checkpoint":
        cmd.append("--fresh")
    elif start_from == "resume":
        cmd.extend(["--start-from", "resume"])
        resume_checkpoint = config.paths.resume_checkpoint or ""
        if resume_checkpoint:
            cmd.extend(["--student", resume_checkpoint])
        if not reset_optimizer:
            resume_optimizer = config.paths.resume_optimizer or ""
            if resume_optimizer:
                cmd.extend(["--resume-optimizer", resume_optimizer])

    if reset_optimizer:
        cmd.append("--reset-optimizer")

    return cmd
