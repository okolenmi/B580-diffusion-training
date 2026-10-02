"""ListRuns -- newest-first page of runs, optionally status-filtered."""

from __future__ import annotations

from ..limits import MAX_PAGE_SIZE
from ..dto import ListRunsQuery, ListRunsResult, to_run_dto
from ..errors import InvalidQueryError
from ..ports.run_repository import RunRepository
from ...domain.value_objects import RunStatus


class ListRuns:
    """Single responsibility: validate the query, fetch, project."""

    MAX_LIMIT = MAX_PAGE_SIZE

    def __init__(self, runs: RunRepository) -> None:
        self._runs = runs

    def execute(self, query: ListRunsQuery | None = None) -> ListRunsResult:
        query = query if query is not None else ListRunsQuery()
        status = self._parse_status(query.status)
        if not 1 <= query.limit <= self.MAX_LIMIT:
            raise InvalidQueryError(
                f"limit must be between 1 and {self.MAX_LIMIT}, got {query.limit}"
            )
        found = self._runs.list_runs(limit=query.limit, status=status)
        dtos = tuple(to_run_dto(run) for run in found)
        return ListRunsResult(runs=dtos, count=len(dtos))

    @staticmethod
    def _parse_status(raw: str | None) -> RunStatus | None:
        if raw is None:
            return None
        try:
            return RunStatus(raw)
        except ValueError:
            raise InvalidQueryError(
                f"unknown status {raw!r}; expected one of "
                f"{sorted(s.value for s in RunStatus)}"
            ) from None
