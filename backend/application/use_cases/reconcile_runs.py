"""ReconcileRuns -- startup sweep of runs left unfinished by a prior process.

Called once from the composition root before the server accepts
requests. Each unfinished row is finalised through compare-and-swap so
a racing writer (impossible at startup, but the CAS keeps the
invariant enforced in one place) can never be overwritten:

- ``created``  -> failed  (the server died before the spawn completed)
- ``running``, no pid      -> failed  (nothing to reconcile against)
- ``running``, kill() True -> cancelled (leftover process reaped)
- ``running``, kill() False-> failed  (process already gone or PID
  reused by something that is no longer our trainer)
"""

from __future__ import annotations

import logging

from ..dto import ReconcileResult
from ..ports.clock import Clock
from ..ports.event_bus import EventBus
from ..ports.run_repository import RunRepository
from ..ports.training_gateway import TrainingGateway
from ...domain.value_objects import RunStatus

logger = logging.getLogger(__name__)


class ReconcileRuns:
    def __init__(
        self,
        *,
        runs: RunRepository,
        events: EventBus,
        gateway: TrainingGateway,
        clock: Clock,
    ) -> None:
        self._runs = runs
        self._events = events
        self._gateway = gateway
        self._clock = clock

    def execute(self) -> ReconcileResult:
        cleaned = 0
        for run in self._runs.list_unfinished():
            expected = RunStatus.RUNNING
            if run.status is RunStatus.CREATED:
                expected = RunStatus.CREATED
                run.mark_failed(
                    at=self._clock.now(),
                    error="server stopped before the run launched",
                )
            elif run.pid is None:
                run.mark_failed(
                    at=self._clock.now(),
                    error="orphan cleanup: no pid stored",
                )
            elif self._gateway.kill(run.pid):
                run.cancel(
                    at=self._clock.now(),
                    reason="orphan cleanup: killed leftover training process",
                )
            else:
                run.mark_failed(
                    at=self._clock.now(),
                    error="orphan cleanup: process already gone or no longer ours",
                )

            if not self._runs.update_if_status(run, expected=expected):
                logger.warning(
                    "reconcile: run %s was finalised by another writer", run.id
                )
                continue
            cleaned += 1
            for event in run.collect_events():
                self._events.publish(event)
            logger.info(
                "reconciled run %s -> %s", run.id, run.status.value
            )
        return ReconcileResult(cleaned=cleaned)
