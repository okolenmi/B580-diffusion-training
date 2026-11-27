"""GetExecutionLog -- what a run's process printed, for a run that is live or ended badly."""

from __future__ import annotations

from ..errors import GraphExecutionNotFoundError
from ..ports.execution_logs import ExecutionLogs
from ..ports.graph_execution_repository import GraphExecutionRepository


class GetExecutionLog:
    #: A training log is megabytes of progress lines; a reader wants the end, where the traceback is.
    MAX_BYTES = 200_000

    def __init__(self, executions: GraphExecutionRepository, logs: ExecutionLogs) -> None:
        self._executions = executions
        self._logs = logs

    def execute(self, execution_id: int) -> str:
        if self._executions.get(execution_id) is None:
            raise GraphExecutionNotFoundError(f"execution {execution_id} does not exist")
        text = self._logs.tail(execution_id, self.MAX_BYTES)
        if text is None:
            raise GraphExecutionNotFoundError(f"execution {execution_id} has no log: a run that finished is not kept, only runs that failed or were stopped (the newest few)")
        return text
