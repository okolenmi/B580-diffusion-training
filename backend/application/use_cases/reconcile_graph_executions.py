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
a watcher and a final outcome.

Not adopted does **not** mean failed. A run that outlived the server has
two distinguishable endings, and only the first one is a crash:

* it finished, and said so -- an outcome record on disk, plus every node
  result it produced. The row is settled from that record: the verdict is
  the run's, and the results are recovered rather than discarded.
* it was killed -- no outcome record. Then there is nothing to read and the
  absence *is* the evidence, so the row fails with the reason that
  describes it ("server stopped before the execution started" /
  "server restarted while the execution was in flight").

Reading the record first is what makes the difference. Measured: a
3000-node run killed the server after it started, finished all 3000 nodes
and wrote a clean `{"kind": "outcome", "error": null}` -- and was reported
as `error` with zero results, because "no process" was the only thing
checked.

This is round-2 finding N-03 ("a run that finished while the server was
down is marked failed") arriving again in newer code. It was recorded as
moot when the run route was removed, which it was -- for that route.

CAS-protected like every other terminal writer (impossible to race at
startup, but the invariant stays enforced in one place) -- via
``ExecutionLifecycleWriter``, whose ``fail_if_unfinished`` the supervisor's
own crash-repair path uses too, and whose ``finalise_from_record`` this
path uses when the run left an answer behind.
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
        recorded = 0
        for execution in self._executions.list_unfinished():
            execution_id = execution.require_id()
            if self._launcher.adopt(execution_id) is not None:
                adopted += 1
                continue

            # Before concluding anything from the absence of a process,
            # ask what the run said about itself. A run that finished
            # while this server was down left an outcome record and its
            # node results on disk, and failing the row on the strength of
            # "there is no process" would report a completed run as a
            # crash and throw away work that succeeded.
            outcome = self._launcher.recorded_outcome(execution_id)
            if outcome is not None:
                if not self._writer.finalise_from_record(
                    execution, error=outcome.error, results=outcome.results,
                ):
                    logger.warning(
                        "reconcile: execution %s was finalised by another writer",
                        execution.id,
                    )
                    continue
                recorded += 1
                logger.info(
                    "reconciled graph execution %s -> %s from the run's own "
                    "record (%d result(s) recovered)",
                    execution.id,
                    execution.status.value,
                    len(outcome.results),
                )
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
        if recorded:
            logger.info(
                "settled %d execution(s) from the record their run left behind",
                recorded,
            )
        return ReconcileResult(cleaned=cleaned, adopted=adopted)
