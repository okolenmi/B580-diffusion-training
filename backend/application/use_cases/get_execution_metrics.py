"""GetExecutionMetrics -- the monitor frames of one run, live or finished."""

from __future__ import annotations

from typing import Any

from ..errors import GraphExecutionNotFoundError
from ..ports.execution_metrics import ExecutionMetrics
from ..ports.graph_execution_repository import GraphExecutionRepository


class GetExecutionMetrics:
    def __init__(self, executions: GraphExecutionRepository, metrics: ExecutionMetrics) -> None:
        self._executions = executions
        self._metrics = metrics

    def _require(self, execution_id: int) -> None:
        if self._executions.get(execution_id) is None:
            raise GraphExecutionNotFoundError(f"execution {execution_id} does not exist")

    def streams(self, execution_id: int) -> list[str]:
        self._require(execution_id)
        return self._metrics.streams(execution_id)

    def frames(self, execution_id: int, stream: str) -> list[dict[str, Any]]:
        self._require(execution_id)
        return self._metrics.read(execution_id, stream)
