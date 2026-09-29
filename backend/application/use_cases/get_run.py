"""GetRun -- fetch exactly one run as a DTO."""

from __future__ import annotations

from ..dto import RunDTO, to_run_dto
from ..errors import InvalidQueryError, RunNotFoundError
from ..ports.run_repository import RunRepository
from ...domain.value_objects import RunId


class GetRun:
    def __init__(self, runs: RunRepository) -> None:
        self._runs = runs

    def execute(self, run_id: RunId) -> RunDTO:
        if run_id < 1:
            raise InvalidQueryError(f"run id must be a positive integer, got {run_id}")
        run = self._runs.get(run_id)
        if run is None:
            raise RunNotFoundError(f"run {run_id} does not exist")
        return to_run_dto(run)
