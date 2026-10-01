from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

import sys
import os

# Ensure the paths module is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from paths import get_comfy_dir, get_project_root, get_runs_dir, get_log_path, get_progress_path, get_dataset_db_path, get_checkpoints_dir, get_loras_dir


class Settings(BaseSettings):
    """Server configuration settings."""

    host: str = "0.0.0.0"
    port: int = 8765

    # Use paths module for all directory-related settings
    # These properties are computed dynamically
    
    model_config = SettingsConfigDict(
        env_prefix="COMFY_",
        case_sensitive=False
    )

    def _persisted_setting(self, key: str) -> str:
        """Read a server-level setting persisted via POST /settings (stored
        in the DB, same mechanism already used for "default_config").
        Returns "" if unset, or if the DB isn't reachable for some reason
        -- this must never be allowed to crash path resolution.
        """
        try:
            from . import db
            return db.get_setting(self.db_path, key, "")
        except Exception:
            return ""

    @property
    def comfy_dir(self) -> Path:
        """ComfyUI directory - working directory for training.

        Resolution order:
        1. Everything paths.get_comfy_dir() checks: COMFY_DIR (set directly,
           or via a .env file at the project root -- see paths.py's
           _load_dotenv() for the simple no-server-required way to set
           this), cwd heuristic, sibling-folder auto-detect.
        2. Persisted server setting ("comfy_dir"), set via POST /settings.
           Last resort, not first -- so a stale UI-set value from a
           previous experiment can never silently shadow a freshly-edited
           .env file or env var.
        """
        try:
            return get_comfy_dir()
        except RuntimeError:
            override = self._persisted_setting("comfy_dir")
            if override and Path(override).is_dir():
                return Path(override)
            raise

    @property
    def project_root(self) -> Path:
        """This project's root directory."""
        return get_project_root()
    
    @property
    def workspace_root(self) -> Path:
        """Parent of project root (where ComfyUI and this project are siblings)."""
        return self.project_root.parent
    
    @property
    def db_path(self) -> Path:
        """Database path inside project root."""
        return self.project_root / "server/data/server.db"

    @property
    def dataset_db_path(self) -> Path:
        """Dataset database path."""
        return get_dataset_db_path()
    
    @property
    def venv_python(self) -> str:
        """Python executable used to launch the training subprocess.

        Resolution order:
        1. VENV_PYTHON environment variable (set directly, or via a .env
           file at the project root -- see paths.py's _load_dotenv()).
        2. <workspace_root>/venv/bin/python, if it exists (project / ComfyUI
           / venv all siblings -- kept as a convenience default).
        3. Persisted server setting ("venv_python"), set via POST /settings.
           Last resort, so a stale UI-set value can't silently shadow a
           freshly-edited .env file or env var.
        4. Fallback to whatever "python" resolves to on PATH.
        """
        env_venv = os.environ.get("VENV_PYTHON")
        if env_venv and Path(env_venv).exists():
            return env_venv

        venv_path = self.workspace_root / "venv/bin/python"
        if venv_path.exists():
            return str(venv_path)

        override = self._persisted_setting("venv_python")
        if override and Path(override).exists():
            return override

        # Fallback to system python if venv doesn't exist
        return "python"
    
    @property
    def checkpoints_dir(self) -> Path:
        """Directory where full checkpoints (teacher/student/full-finetune
        .safetensors) live -- used to resolve relative checkpoint paths in
        config, and to list available checkpoints for the picker UI.

        Resolution order:
        1. Persisted server setting ("checkpoints_dir"), set via POST /settings.
        2. paths.get_checkpoints_dir(): CHECKPOINTS_DIR env var/.env, then
           <comfy_dir>/models/checkpoints (ComfyUI's own layout, so existing
           checkpoints are found with zero extra setup), then a last-resort
           <project_root>/checkpoints fallback.
        """
        override = self._persisted_setting("checkpoints_dir")
        if override and Path(override).is_dir():
            return Path(override)
        return get_checkpoints_dir()

    @property
    def loras_dir(self) -> Path:
        """Directory where LoRA adapter .safetensors files live. Same
        resolution order as checkpoints_dir, via paths.get_loras_dir()."""
        override = self._persisted_setting("loras_dir")
        if override and Path(override).is_dir():
            return Path(override)
        return get_loras_dir()

    @property
    def runs_dir(self) -> Path:
        """Runs directory - where run logs and outputs are stored."""
        return get_runs_dir()

    def run_dir(self, run_id: int) -> Path:
        """Get the directory for a specific run."""
        return get_runs_dir() / f"run_{run_id}"

    def run_log_path(self, run_id: int) -> Path:
        """Get the log file path for a run."""
        return get_log_path(run_id)

    def run_progress_path(self, run_id: int) -> Path:
        """Get the progress file path for a run."""
        return get_progress_path(run_id)


settings = Settings()