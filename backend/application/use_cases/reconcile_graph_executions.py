"""ReconcileGraphExecutions -- startup sweep of unfinished rows.

Called once from the composition root before the server accepts
requests. An unfinished row is *usually* debris -- this server did not
start it, so whatever did is gone. But "usually" is doing real work in
that sentence, and since WP-22 it has a reason:

A graph run lives in its own session, so the child survives the server
that started it. Restarting the server is therefore not the end of a run
that is still going, and failing its row would throw away work that is
still making progress and leave a process nobody is watching -- writing to
an event file with no reader, holding the card, unstoppable through the
API.

So each unfinished row is offered to the launcher first. Adopted, it gets
a watcher and a final outcome. Not adopted, the row is failed exactly as
before, with the same reasons, because that is still the common case and
its message ("server restarted while the execution was in flight") is the
honest description of what happened to it.

Adoption cannot resurrect a *queued* row whose child never started
writing: the event file is what the watcher reads, so a run that produced
none is not observable, and it fails like any other debris.

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
from ..ports.execution_launcher import ExecutionLauncher
from ..ports.graph_execution_repository import GraphExecutionRepository
from ...domain.value_objects import GraphStatus

logger = logging.getLogger(__name__)


class ReconcileGraphExecutions:
    def __init__(
        self,
        *,
        executions: GraphExecutionRepository,
        writer: ExecutionLifecycleWriter,
        launcher: ExecutionLauncher,
        clock: Clock,
    ) -> None:
        self._executions = executions
        self._writer = writer
        self._launcher = launcher
        self._clock = clock

    def execute(self) -> ReconcileResult:
        adopted = 0
        cleaned = 0
        for execution in self._executions.list_unfinished():
            if self._launcher.adopt(execution.require_id()) is not None:
                adopted += 1
                continue

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
        if adopted:
            logger.info("adopted %d still-running graph execution(s)", adopted)
        return ReconcileResult(cleaned=cleaned, adopted=adopted)
