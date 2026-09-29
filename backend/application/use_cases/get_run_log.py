"""GetRunLog -- tail a run's log file."""

from __future__ import annotations

import logging

from ..dto import LogResult
from ..errors import InvalidQueryError, RunNotFoundError
from ..ports.run_artifacts import RunArtifacts
from ..ports.run_repository import RunRepository

logger = logging.getLogger(__name__)


class GetRunLog:
    MAX_LINES = 500
    DEFAULT_LINES = 100

    def __init__(self, *, runs: RunRepository, artifacts: RunArtifacts) -> None:
        self._runs = runs
        self._artifacts = artifacts

    def execute(self, run_id: int, *, lines: int = DEFAULT_LINES) -> LogResult:
        if self._runs.get(run_id) is None:
            raise RunNotFoundError(f"run {run_id} does not exist")
        if not 1 <= lines <= self.MAX_LINES:
            raise InvalidQueryError(
                f"lines must be between 1 and {self.MAX_LINES}, got {lines}"
            )
        log_path = self._artifacts.paths_for(run_id).log
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return LogResult(log="")  # child may not have created it yet
        except OSError as exc:
            # Parity with the legacy reader: unreadable log is empty
            # output, not a 500. Observability, not state.
            logger.warning("cannot read log for run %s: %s", run_id, exc)
            return LogResult(log="")
        tail = "".join(text.splitlines(keepends=True)[-lines:])
        return LogResult(log=tail)
