"""WorkspaceLayout -- repo conventions (dirs, venv, run artifacts).

Path *policy* lives in ``path_tiers``; this class binds it to the
process's project root and adds the run-artifact conventions.

Bridging note (deliberate, not laziness): production paths are
delegated to the repo's ``paths`` module -- the *child* trainer
derives its progress file from ``paths.get_progress_path``, so
parent and child cannot drift. Importing ``paths`` performs its
documented .env fill-on-import (the same thing ``run_server.sh``
does for the shell); this happens once at composition time, never as
an accidental import side effect of a backend module.

``settings_kv=`` is the settings store's raw getter: with it, the
DB-tier overrides join path resolution (getters read it fresh on
every access, so a settings change is visible immediately, with no
cache to invalidate). Without it (tests), only env + defaults
resolve.

``runs_dir=`` overrides the artifact root for tests: paths are then
computed locally (same convention: ``run_<id>/log.txt`` +
``log.progress.jsonl``) without touching the real ``runs/`` folder or
importing the repo module.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

from . import path_tiers

GetSetting = Callable[[str, str], str]


class WorkspaceLayout:
    def __init__(
        self,
        project_root: Path,
        *,
        runs_dir: Path | None = None,
        settings_kv: GetSetting | None = None,
    ) -> None:
        self._root = project_root
        self._runs_dir = runs_dir
        self._settings_kv = settings_kv

    # -- settings tier -------------------------------------------------

    def _setting(self, key: str, default: str = "") -> str:
        if self._settings_kv is None:
            return default
        return self._settings_kv(key, default)

    # -- dirs ----------------------------------------------------------

    @property
    def project_root(self) -> Path:
        return self._root

    @property
    def comfy_dir(self) -> Path:
        """Working directory for the trainer (cwd of the subprocess).

        May raise ``RuntimeError`` when no ComfyUI install can be
        identified and no setting overrides one -- reported as
        ``null`` by the settings endpoint rather than hidden.
        """
        return path_tiers.comfy_dir(self._root, self._setting)

    @property
    def venv_python(self) -> str:
        return path_tiers.venv_python(self._root, self._setting)

    @property
    def checkpoints_dir(self) -> Path:
        return path_tiers.checkpoints_dir(self._root, self._setting)

    @property
    def loras_dir(self) -> Path:
        return path_tiers.loras_dir(self._root, self._setting)

    @property
    def datasets_dir(self) -> Path:
        return path_tiers.datasets_dir(self._root)

    # -- run artifacts -------------------------------------------------

    @property
    def runs_dir(self) -> Path:
        if self._runs_dir is not None:
            return self._runs_dir
        return path_tiers.import_paths(self._root).get_runs_dir()

    def run_dir(self, run_id: int) -> Path:
        return self.runs_dir / f"run_{run_id}"

    def log_path(self, run_id: int) -> Path:
        if self._runs_dir is None:
            return path_tiers.import_paths(self._root).get_log_path(run_id)  # mkdirs itself
        path = self.run_dir(run_id) / "log.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def progress_path(self, run_id: int) -> Path:
        """Must equal what the child derives (``paths.get_progress_path``)."""
        if self._runs_dir is None:
            return path_tiers.import_paths(self._root).get_progress_path(run_id)  # mkdirs itself
        path = self.run_dir(run_id) / "log.progress.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
