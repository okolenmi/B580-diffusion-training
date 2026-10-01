"""ReconcileGraphExecutions -- startup sweep of rows a dead process left.

Called once from the composition root before the server accepts
requests. Worker threads do not survive their process, so *every*
non-terminal row is debris:

- ``queued`` -> failed ("server stopped before the execution started"
  -- its thread never claimed it, nothing ran);
- ``running`` -> failed ("server restarted while the execution was in
  flight" -- whatever nodes completed are already in ``results``).

CAS-protected like every other terminal writer (impossible to race at
startup, but the invariant stays enforced in one place).
"""

from __future__ import annotations

import logging

from ..dto import ReconcileResult
from ..ports.clock import Clock
from ..event_publisher import EventPublisher
from ..ports.graph_execution_repository import GraphExecutionRepository
from ...domain.value_objects import GraphStatus

logger = logging.getLogger(__name__)


class ReconcileGraphExecutions:
    def __init__(
        self,
        *,
        executions: GraphExecutionRepository,
        events: EventPublisher,
        clock: Clock,
    ) -> None:
        self._executions = executions
        self._events = events
        self._clock = clock

    def execute(self) -> ReconcileResult:
        cleaned = 0
        for execution in self._executions.list_unfinished():
            expected = execution.status
            if expected is GraphStatus.QUEUED:
                error = "server stopped before the execution started"
            else:
                error = "server restarted while the execution was in flight"
            execution.mark_failed(at=self._clock.now(), error=error)

            if not self._executions.update_if_status(execution, expected=expected):
                logger.warning(
                    "reconcile: execution %s was finalised by another writer",
                    execution.id,
                )
                continue
            cleaned += 1
            self._events.publish(execution)
            logger.info(
                "reconciled graph execution %s -> %s",
                execution.id,
                execution.status.value,
            )
        return ReconcileResult(cleaned=cleaned)
