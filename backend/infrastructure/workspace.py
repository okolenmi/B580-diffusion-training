"""WorkspaceLayout -- repo conventions (venv, dirs, run artifacts).

Bridging note (deliberate, not laziness): production run-artifact
paths are delegated to the repo's ``paths`` module -- the *child*
trainer derives its progress file from ``paths.get_progress_path``, so
parent and child cannot drift. Importing ``paths`` performs its
documented .env fill-on-import (the same thing ``run_server.sh`` does
for the shell); this happens once at composition time, never as an
accidental import side effect of a backend module.

The old server's DB-backed setting overrides ("venv_python",
"default_config", ...) are an M3 concern; here only env + defaults
resolve.

``runs_dir=`` overrides the artifact root for tests: paths are then
computed locally (same convention: ``run_<id>/log.txt`` +
``log.progress.jsonl``) without touching the real ``runs/`` folder or
importing the repo module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


class WorkspaceLayout:
    def __init__(
        self, project_root: Path, *, runs_dir: Path | None = None
    ) -> None:
        self._root = project_root
        self._runs_dir = runs_dir

    # -- repo bridge ---------------------------------------------------

    def _paths(self):
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))
        import paths  # noqa: PLC0415 -- explicit bridge, see module doc

        return paths

    # -- dirs ----------------------------------------------------------

    @property
    def project_root(self) -> Path:
        return self._root

    @property
    def comfy_dir(self) -> Path:
        """Working directory for the trainer (cwd of the subprocess)."""
        return self._paths().get_comfy_dir()

    @property
    def venv_python(self) -> str:
        """Interpreter for the trainer (env tier; M3 adds settings)."""
        self._paths()  # ensure repo .env is loaded before reading env
        env = os.environ.get("VENV_PYTHON", "")
        if env:
            return env
        workspace = self._root.parent  # venv sits beside the project
        candidate = workspace / "venv" / "bin" / "python"
        return str(candidate) if candidate.exists() else "python"

    @property
    def checkpoints_dir(self) -> Path:
        return self._paths().get_checkpoints_dir()

    @property
    def loras_dir(self) -> Path:
        return self._paths().get_loras_dir()

    # -- run artifacts -------------------------------------------------

    @property
    def runs_dir(self) -> Path:
        if self._runs_dir is not None:
            return self._runs_dir
        return self._paths().get_runs_dir()

    def run_dir(self, run_id: int) -> Path:
        return self.runs_dir / f"run_{run_id}"

    def log_path(self, run_id: int) -> Path:
        if self._runs_dir is None:
            return self._paths().get_log_path(run_id)  # mkdirs itself
        path = self.run_dir(run_id) / "log.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def progress_path(self, run_id: int) -> Path:
        """Must equal what the child derives (``paths.get_progress_path``)."""
        if self._runs_dir is None:
            return self._paths().get_progress_path(run_id)  # mkdirs itself
        path = self.run_dir(run_id) / "log.progress.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
