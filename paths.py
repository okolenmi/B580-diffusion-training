"""Path resolver — unified path handling for the entire application.

This module provides consistent path resolution across server and trainer.
All paths are resolved relative to COMFY_DIR which is the working directory
for training runs.
"""

from pathlib import Path
import os
import sys


def _load_dotenv():
    """Load KEY=VALUE lines from a .env file at the project root into
    os.environ, if the file exists.

    This is the actual answer to "where do I put COMFY_DIR / VENV_PYTHON
    without exporting shell variables every session": create a file called
    `.env` right next to this one, containing e.g.:

        COMFY_DIR=/path/to/ComfyUI
        VENV_PYTHON=/path/to/venv/bin/python

    Real environment variables always win over the .env file (standard
    dotenv convention: this only fills in a variable that isn't already
    set) -- so `COMFY_DIR=/other/path ./run_server.sh` still overrides
    whatever's in .env for that one invocation.

    Deliberately not pulling in python-dotenv as a dependency for what's
    two lines of parsing.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()


# Explicit override, set via set_comfy_dir() (e.g. from config.paths.comfy_dir).
_comfy_dir_override = None


def set_comfy_dir(path: str | Path | None):
    """Set the ComfyUI directory explicitly (e.g., from config)."""
    global _comfy_dir_override
    if path:
        _comfy_dir_override = Path(path).resolve()
    else:
        _comfy_dir_override = None


def get_comfy_dir() -> Path:
    """Get the ComfyUI directory (working directory for training).
    
    The ComfyUI directory is determined in this order:
    1. Explicitly set via set_comfy_dir()
    2. COMFY_DIR environment variable
    3. Current working directory if it contains 'comfy' subdirectory
    4. Parent of this file's project root (assumes the project (this repo's root) is a sibling of ComfyUI)
    """
    global _comfy_dir_override
    
    # Check explicit setting first
    if _comfy_dir_override is not None:
        if _comfy_dir_override.exists():
            return _comfy_dir_override
        # Invalid, fall through to other methods
    
    # Check environment variable
    env_comfy = os.environ.get("COMFY_DIR")
    if env_comfy:
        p = Path(env_comfy).resolve()
        if p.exists():
            return p
    
    # Check current working directory
    cwd = Path.cwd()
    if (cwd / "comfy").exists():
        return cwd.resolve()
    
    # Fallback: assume project structure is:
    #   some_folder/
    #     ├── ComfyUI/
    #     └── <this project>/
    # We need to find ComfyUI relative to this file
    project_root = get_project_root()
    potential = project_root.parent / "ComfyUI"
    if potential.exists():
        return potential.resolve()
    
    # Last resort: raise error
    raise RuntimeError(
        "Cannot find ComfyUI directory. Set COMFY_DIR environment variable, "
        "call set_comfy_dir(), or run from the ComfyUI directory."
    )


def get_project_root() -> Path:
    """Get this project's root directory."""
    return Path(__file__).resolve().parent


def get_runs_dir() -> Path:
    """Get the directory for training run logs and outputs."""
    return get_project_root() / "runs"


def get_datasets_dir() -> Path:
    """Get the root directory for managed datasets."""
    return get_project_root() / "datasets"


def get_dataset_db_path() -> Path:
    """Get the path to the dataset SQLite database."""
    return get_datasets_dir() / "datasets.db"


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a path to absolute form.
    
    If path is already absolute, return as-is.
    If relative, resolve from base (defaults to comfy_dir).
    """
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    
    if base is None:
        base = get_comfy_dir()
    return (base / p).resolve()


def get_log_path(run_id: int) -> Path:
    """Get the log file path for a run."""
    run_dir = get_runs_dir() / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "log.txt"


def get_progress_path(run_id: int) -> Path:
    """Get the progress JSONL file path for a run.
    
    Uses the naming convention: log.progress.jsonl
    """
    run_dir = get_runs_dir() / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "log.progress.jsonl"


def get_run_dir(run_id: int) -> Path:
    """Get the run directory."""
    run_dir = get_runs_dir() / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# Explicit overrides, set via set_checkpoints_dir()/set_loras_dir() (e.g. from a
# persisted server setting -- see server/config.py's Settings.checkpoints_dir).
_checkpoints_dir_override = None
_loras_dir_override = None


def set_checkpoints_dir(path: str | Path | None):
    global _checkpoints_dir_override
    _checkpoints_dir_override = Path(path).resolve() if path else None


def set_loras_dir(path: str | Path | None):
    global _loras_dir_override
    _loras_dir_override = Path(path).resolve() if path else None


def get_checkpoints_dir() -> Path:
    """Directory where full checkpoints (teacher/student/full-finetune
    .safetensors files) live. Used both to resolve relative checkpoint paths
    in config, and to list available checkpoints for the picker UI.

    Resolution order:
    1. Explicit override via set_checkpoints_dir()
    2. CHECKPOINTS_DIR environment variable (or .env)
    3. <comfy_dir>/models/checkpoints -- ComfyUI's own standard layout, so
       checkpoints you already have there are found with zero extra setup
    4. <project_root>/checkpoints -- last-resort fallback that doesn't
       depend on comfy_dir being resolvable at all
    """
    if _checkpoints_dir_override is not None:
        return _checkpoints_dir_override
    env = os.environ.get("CHECKPOINTS_DIR")
    if env:
        return Path(env).resolve()
    try:
        return get_comfy_dir() / "models" / "checkpoints"
    except RuntimeError:
        return get_project_root() / "checkpoints"


def get_loras_dir() -> Path:
    """Directory where LoRA adapter .safetensors files live. Same resolution
    order as get_checkpoints_dir(), using LORAS_DIR / <comfy_dir>/models/loras."""
    if _loras_dir_override is not None:
        return _loras_dir_override
    env = os.environ.get("LORAS_DIR")
    if env:
        return Path(env).resolve()
    try:
        return get_comfy_dir() / "models" / "loras"
    except RuntimeError:
        return get_project_root() / "loras"


def _model_base_dir(kind: str) -> Path:
    if kind == "checkpoint":
        return get_checkpoints_dir()
    if kind == "lora":
        return get_loras_dir()
    raise ValueError(f"kind must be 'checkpoint' or 'lora', got {kind!r}")


def resolve_model_path(path_str: str | Path, kind: str) -> Path:
    """Resolve a checkpoint or LoRA path that may be given as either an
    absolute path or a path relative to checkpoints_dir/loras_dir (kind).
    This is what lets config fields like paths.student or
    tuning.lora_continue_from just be a filename ("my_lora.safetensors")
    instead of a full path, while still allowing an absolute path for
    anything stored elsewhere.
    """
    return resolve_path(path_str, base=_model_base_dir(kind))


def get_resume_dir(kind: str) -> Path:
    """Dedicated subfolder for auto-managed resume files (the periodic
    mid-training checkpoint/optimizer-state saves) -- kept separate from
    checkpoints_dir/loras_dir's top level so they don't clutter the file
    picker or get confused with checkpoints/LoRAs a user actually chose to
    keep. Created on first use.
    """
    d = _model_base_dir(kind) / "resume"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_model_files(kind: str) -> list[str]:
    """List available .safetensors files under checkpoints_dir/loras_dir
    (kind), as paths relative to that directory, for the file-picker
    dropdown. Excludes the resume/ subfolder -- those are auto-managed
    working files, not something you'd manually pick from a list.
    Returns an empty list (not an error) if the directory doesn't exist yet.
    """
    base = _model_base_dir(kind)
    if not base.is_dir():
        return []
    results = []
    for p in sorted(base.rglob("*.safetensors")):
        try:
            rel = p.relative_to(base)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "resume":
            continue
        results.append(str(rel))
    return results
