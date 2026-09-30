"""GetGraphExecution -- full detail of one execution (poll endpoint).

Returns identity, status, per-node results with timings, and the graph
snapshot of what actually ran (the row's stored copy -- independent of
later edits or library changes).
"""

from __future__ import annotations

from ..dto import GraphExecutionDTO, to_execution_dto
from ..errors import GraphExecutionNotFoundError
from ..ports.graph_execution_repository import GraphExecutionRepository


class GetGraphExecution:
    def __init__(self, executions: GraphExecutionRepository) -> None:
        self._executions = executions

    def execute(self, execution_id: int) -> GraphExecutionDTO:
        execution = self._executions.get(execution_id)
        if execution is None:
            raise GraphExecutionNotFoundError(
                f"execution {execution_id} does not exist"
            )
        return to_execution_dto(execution)
