from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

import sys
import os

# Ensure the paths module is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from paths import get_comfy_dir, get_project_root, get_runs_dir, get_log_path, get_progress_path, get_dataset_db_path


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

    @property
    def comfy_dir(self) -> Path:
        """ComfyUI directory - working directory for training."""
        return get_comfy_dir()
    
    @property
    def project_root(self) -> Path:
        """Comfy-converter project root."""
        return get_project_root()
    
    @property
    def workspace_root(self) -> Path:
        """Parent of project root (where ComfyUI and comfy-converter are siblings)."""
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
        """Python executable in the virtual environment."""
        venv_path = self.workspace_root / "venv/bin/python"
        if venv_path.exists():
            return str(venv_path)
        # Fallback to system python if venv doesn't exist
        return "python"
    
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