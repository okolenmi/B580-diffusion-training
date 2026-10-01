"""DeleteGraphExecutions -- wipe execution history.

Deletes every row, active ones included (mirrors ``DeleteRuns``): a
worker whose row disappears simply loses every subsequent CAS and exits
quietly -- no orphaned thread, no resurrected row.
"""

from __future__ import annotations

from ..dto import DeleteGraphExecutionsResult
from ..event_publisher import EventPublisher
from ..ports.graph_execution_repository import GraphExecutionRepository
from ...domain.events import GraphExecutionsDeleted


class DeleteGraphExecutions:
    def __init__(
        self, executions: GraphExecutionRepository, events: EventPublisher
    ) -> None:
        self._executions = executions
        self._events = events

    def execute(self) -> DeleteGraphExecutionsResult:
        deleted = self._executions.delete_all()
        if deleted:
            self._events.emit(GraphExecutionsDeleted(deleted=deleted))
        return DeleteGraphExecutionsResult(deleted=deleted)
