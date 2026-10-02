"""ReconcileGraphExecutions -- startup sweep of rows a dead process left.

Called once from the composition root before the server accepts
requests. Worker threads do not survive their process, so *every*
non-terminal row is debris:

- ``queued`` -> failed ("server stopped before the execution started"
  -- its thread never claimed it, nothing ran);
- ``running`` -> failed ("server restarted while the execution was in
  flight" -- whatever nodes completed are already in ``results``).

CAS-protected like every other terminal writer (impossible to race at
startup, but the invariant stays enforced in one place) -- via
``ExecutionLifecycleWriter.fail_if_unfinished``, which the supervisor's
own crash-repair path uses too.
"""

from __future__ import annotations

import logging

from ..dto import ReconcileResult
from ..ports.clock import Clock
from ..lifecycle_writer import ExecutionLifecycleWriter
from ..ports.graph_execution_repository import GraphExecutionRepository
from ...domain.value_objects import GraphStatus

logger = logging.getLogger(__name__)


class ReconcileGraphExecutions:
    def __init__(
        self,
        *,
        executions: GraphExecutionRepository,
        writer: ExecutionLifecycleWriter,
        clock: Clock,
    ) -> None:
        self._executions = executions
        self._writer = writer
        self._clock = clock

    def execute(self) -> ReconcileResult:
        cleaned = 0
        for execution in self._executions.list_unfinished():
            if execution.status is GraphStatus.QUEUED:
                error = "server stopped before the execution started"
            else:
                error = "server restarted while the execution was in flight"

            if not self._writer.fail_if_unfinished(execution, error=error):
                logger.warning(
                    "reconcile: execution %s was finalised by another writer",
                    execution.id,
                )
                continue
            cleaned += 1
            logger.info(
                "reconciled graph execution %s -> %s",
                execution.id,
                execution.status.value,
            )
        return ReconcileResult(cleaned=cleaned)
