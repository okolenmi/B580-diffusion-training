"""ReconcileRuns -- startup sweep of runs left unfinished by a prior process.

Called once from the composition root before the server accepts
requests. Each unfinished row is either **adopted** or finalised,
through compare-and-swap so a racing writer (impossible at startup, but
the CAS keeps the invariant enforced in one place) can never be
overwritten:

- ``created``  -> failed  (the server died before the spawn completed)
- ``running``, no pid      -> failed  (nothing to reconcile against)
- ``running``, pid alive and still ours -> **adopted**: the supervisor
  re-attaches and keeps watching (docs 07 F-11). Trainers are started in
  their own session precisely so they survive a server restart; killing
  them here -- as the legacy server did -- threw away hours of work for
  no reason.
- ``running``, kill() True -> cancelled (leftover process reaped)
- ``running``, kill() False-> failed  (process already gone or PID
  reused by something that is no longer our trainer)

Adoption deliberately does not touch the row: it stays ``running`` with
the pid it always had. Clients refetch authoritative state when they
(re)connect, so no event is needed for a change none of them saw.
"""

from __future__ import annotations

import logging

from ..dto import ReconcileResult
from ..ports.clock import Clock
from ..event_publisher import EventPublisher
from ..lifecycle_writer import RunLifecycleWriter
from ..ports.run_artifacts import RunArtifacts
from ..ports.run_repository import RunRepository
from ..ports.run_watcher import RunWatcher
from ..ports.training_gateway import TrainingGateway
from ...domain.value_objects import RunStatus

logger = logging.getLogger(__name__)


class ReconcileRuns:
    def __init__(
        self,
        *,
        runs: RunRepository,
        writer: RunLifecycleWriter,
        gateway: TrainingGateway,
        clock: Clock,
        watcher: RunWatcher,
        artifacts: RunArtifacts,
    ) -> None:
        self._runs = runs
        self._writer = writer
        self._gateway = gateway
        self._clock = clock
        # Required, not optional: with these missing the sweep would
        # silently fall back to killing live trainers, which is the data
        # loss docs 07 F-11 exists to prevent (docs 08 S-02).
        self._watcher = watcher
        self._artifacts = artifacts

    def execute(self) -> ReconcileResult:
        cleaned = 0
        adopted = 0
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
            elif self._adopt(run):
                adopted += 1
                logger.info(
                    "reconciled run %s -> adopted (pid %s still training)",
                    run.id, run.pid,
                )
                continue
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

            if not self._writer.commit(run, expected=expected):
                logger.warning(
                    "reconcile: run %s was finalised by another writer", run.id
                )
                continue
            cleaned += 1
            logger.info(
                "reconciled run %s -> %s", run.id, run.status.value
            )
        return ReconcileResult(cleaned=cleaned, adopted=adopted)

    def _adopt(self, run) -> bool:
        """Re-attach to a trainer that survived the server. True = adopted.

        Every precondition is checked *before* anything is started: the
        process must exist, still look like this project's trainer, and
        have somewhere to write progress we can tail.
        """
        if run.id is None or run.pid is None:
            return False
        if not self._gateway.is_alive(run.pid) or not self._gateway.owns(run.pid):
            return False
        progress = self._artifacts.paths_for(run.id).progress
        self._watcher.adopt(run_id=run.id, pid=run.pid, progress_path=progress)
        return True