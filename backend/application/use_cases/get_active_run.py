"""GetActiveRun -- the run that is created or running right now."""

from __future__ import annotations

from ..dto import RunDTO, to_run_dto
from ..errors import NoActiveRunError
from ..ports.run_repository import RunRepository


class GetActiveRun:
    def __init__(self, runs: RunRepository) -> None:
        self._runs = runs

    def execute(self) -> RunDTO:
        run = self._runs.find_active()
        if run is None:
            raise NoActiveRunError("no run is currently active")
        return to_run_dto(run)
