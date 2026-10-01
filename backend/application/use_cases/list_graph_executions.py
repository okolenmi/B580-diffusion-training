"""ListGraphExecutions -- newest-first page of execution summaries.

Exists so a page reload (or a second tab) can find a still-running
execution without client-side memory of its id -- the same reason the
legacy endpoint existed, now backed by durable rows instead of a
module-global dict.
"""

from __future__ import annotations

from ..limits import DEFAULT_EXECUTION_PAGE_SIZE, MAX_PAGE_SIZE
from ..dto import ExecutionListResult, to_execution_summary_dto
from ..errors import InvalidQueryError
from ..ports.graph_execution_repository import GraphExecutionRepository

DEFAULT_LIMIT = DEFAULT_EXECUTION_PAGE_SIZE
MAX_LIMIT = MAX_PAGE_SIZE  # one ceiling for every list endpoint


class ListGraphExecutions:
    def __init__(self, executions: GraphExecutionRepository) -> None:
        self._executions = executions

    def execute(self, *, limit: int = DEFAULT_LIMIT) -> ExecutionListResult:
        if not 1 <= limit <= MAX_LIMIT:
            raise InvalidQueryError(
                f"limit must be between 1 and {MAX_LIMIT}, got {limit}"
            )
        found = self._executions.list(limit=limit)
        summaries = tuple(to_execution_summary_dto(item) for item in found)
        return ExecutionListResult(executions=summaries, count=len(summaries))
