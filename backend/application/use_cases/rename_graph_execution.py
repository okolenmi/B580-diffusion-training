"""RenameGraphExecution -- give a run (any status) a name, or clear it. Not part of the lifecycle, so it never races a status change."""

from __future__ import annotations

from ..dto import GraphExecutionSummaryDTO, to_execution_summary_dto
from ..errors import GraphExecutionNotFoundError
from ..ports.graph_execution_repository import GraphExecutionRepository

MAX_NAME = 80


def clean_name(name: str | None) -> str | None:
    """Trimmed and capped; blank means no name."""
    text = (name or "").strip()[:MAX_NAME]
    return text or None


class RenameGraphExecution:
    def __init__(self, executions: GraphExecutionRepository) -> None:
        self._executions = executions

    def execute(self, execution_id: int, name: str | None) -> GraphExecutionSummaryDTO:
        if not self._executions.rename(execution_id, clean_name(name)):
            raise GraphExecutionNotFoundError(f"execution {execution_id} does not exist")
        return to_execution_summary_dto(self._executions.get(execution_id))
