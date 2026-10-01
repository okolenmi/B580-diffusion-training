"""DirectoryRunArtifacts -- the RunArtifacts port over the workspace layout."""

from __future__ import annotations

import logging
import os

from ..application.errors import RunDirectoryCollisionError
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
        # Never write into a directory that already holds files: the
        # trainer opens log.txt with "w", so a colliding id would
        # truncate whatever run lived there (docs 07 F-04). Startup
        # seeds ids above every existing run dir; this is the net for
        # anything that appeared afterwards.
        existing = [child.name for child in paths.directory.iterdir()]
        if existing:
            raise RunDirectoryCollisionError(
                f"{paths.directory} already exists and is not empty "
                f"(found: {', '.join(sorted(existing)[:5])}). It belongs to "
                f"another run -- move it aside, then start again; this "
                f"server never overwrites another run's files."
            )
        return paths

    def highest_existing_run_id(self) -> int:
        """Highest ``runs/run_<id>`` directory already on disk (0 = none).

        The startup seed for :meth:`SqliteRunRepository.continue_ids_above`:
        this adapter owns the ``run_<id>`` naming convention, so the scan
        belongs here rather than in the composition root.
        """
        runs_dir = self._layout.runs_dir
        if not runs_dir.is_dir():
            return 0
        highest = 0
        for child in runs_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("run_"):
                continue
            suffix = child.name[len("run_"):]
            if suffix.isdigit():
                highest = max(highest, int(suffix))
        return highest

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

    def tail_log(self, run_id: int, lines: int) -> str:
        """Last ``lines`` lines of the log, read from the end.

        A training log is the one file here that grows without bound, and
        the only thing anyone ever wants from it is its tail -- reading
        the whole thing to slice 500 lines out of it cost a
        multi-hundred-MB allocation on a long run (docs 07 F-14). So seek
        backwards in doubling blocks until enough newlines are in hand.
        """
        path = self._layout.log_path(run_id)
        try:
            with open(path, "rb") as fh:
                return self._tail_from_end(fh, lines)
        except FileNotFoundError:
            return ""  # the child may not have created it yet
        except OSError as exc:
            logger.warning("cannot tail log for run %s: %s", run_id, exc)
            return ""

    # Block size for the backwards scan: big enough that a 500-line tail
    # is two or three reads even for long lines, small enough to stay a
    # few hundred KB on disk with a cache miss.
    _SCAN_BLOCK = 64 * 1024

    @classmethod
    def _tail_from_end(cls, fh, lines: int) -> str:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        if size == 0:
            return ""
        block = cls._SCAN_BLOCK
        collected = b""
        position = size
        while position > 0 and collected.count(b"\n") <= lines:
            step = min(block, position)
            position -= step
            fh.seek(position)
            collected = fh.read(step) + collected
            block *= 2  # double the window: fewer reads on a big log
        text = collected.decode("utf-8", errors="replace")
        return "".join(text.splitlines(keepends=True)[-lines:])
