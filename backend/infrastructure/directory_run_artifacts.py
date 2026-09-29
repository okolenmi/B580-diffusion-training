"""DirectoryRunArtifacts -- the RunArtifacts port over the workspace layout."""

from __future__ import annotations

import logging

from ..application.ports.run_artifacts import (
    RunArtifacts,
    RunArtifactsPaths,
)
from .workspace import WorkspaceLayout

logger = logging.getLogger(__name__)


class DirectoryRunArtifacts(RunArtifacts):
    def __init__(self, layout: WorkspaceLayout) -> None:
        self._layout = layout

    def prepare(self, run_id: int) -> RunArtifactsPaths:
        paths = self.paths_for(run_id)
        paths.directory.mkdir(parents=True, exist_ok=True)
        return paths

    def paths_for(self, run_id: int) -> RunArtifactsPaths:
        return RunArtifactsPaths(
            directory=self._layout.run_dir(run_id),
            log=self._layout.log_path(run_id),
            progress=self._layout.progress_path(run_id),
        )

    def append_log_note(self, run_id: int, note: str) -> None:
        try:
            with open(self._layout.log_path(run_id), "a", encoding="utf-8") as fh:
                fh.write(note + "\n")
        except OSError as exc:
            logger.warning("cannot append log note for run %s: %s", run_id, exc)
