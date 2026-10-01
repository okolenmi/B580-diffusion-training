"""GetRunLog -- tail a run's log file.

The tailing itself belongs to the artifacts adapter, which walks the file
backwards: a run's log is the one artifact that grows without bound, and
reading it whole to slice 500 lines out was a request-sized allocation
on a long run (docs 07 F-14).
"""

from __future__ import annotations

from ..dto import LogResult
from ..limits import DEFAULT_LOG_LINES, MAX_LOG_LINES
from ..errors import InvalidQueryError, RunNotFoundError
from ..ports.run_artifacts import RunArtifacts
from ..ports.run_repository import RunRepository


class GetRunLog:
    MAX_LINES = MAX_LOG_LINES
    DEFAULT_LINES = DEFAULT_LOG_LINES

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
        return LogResult(log=self._artifacts.tail_log(run_id, lines))
