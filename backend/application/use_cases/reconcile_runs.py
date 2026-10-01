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
- ``running``, kill() False -> read the progress file and finalise from
  the trainer's own last word, exactly as adoption does (docs 08 N-03)

That last row used to be an unconditional ``failed`` at 0 steps, on the
reasoning that a pid we cannot signal is a trainer we know nothing
about. The premise was wrong: the trainer wrote a progress file, and it
is the same evidence adoption already trusts. So a run that finished
100/100 while the server was down was recorded ``failed`` 0/100 -- the
worst possible reading of a successful run, and the common case after an
overnight restart. A dead pid with no progress file at all still fails,
and still with no steps, because then there is genuinely nothing to read.

Adoption deliberately does not touch the row: it stays ``running`` with
the pid it always had. Clients refetch authoritative state when they
(re)connect, so no event is needed for a change none of them saw.
"""

from __future__ import annotations

import logging

from ..dto import ReconcileResult
from ..ports.clock import Clock
from ..lifecycle_writer import RunLifecycleWriter
from ..ports.progress_source import ProgressSource
from ..ports.run_artifacts import RunArtifacts
from ..ports.run_repository import RunRepository
from ..ports.run_watcher import RunWatcher
from ..ports.training_gateway import TrainingGateway
from ..run_verdict import apply_verdict, fold_samples
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
        progress: ProgressSource,
    ) -> None:
        self._runs = runs
        self._writer = writer
        self._gateway = gateway
        self._clock = clock
        # Required, not optional: finalising a dead-pid run from the
        # trainer's own progress file is the same evidence adoption uses
        # (docs 08 N-03). Without it a finished run reads as failed at 0
        # steps, which is how it used to behave.
        self._progress = progress
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
                self._finalise_from_progress(run)

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

    def _finalise_from_progress(self, run) -> None:
        """A ``running`` row whose process is gone: read what the trainer
        left behind and finalise from that (docs 08 N-03).

        The process is not ours to signal, so its exit code is
        unreadable and the trainer's own terminal progress line is the
        only verdict available -- the same evidence, and the same rule,
        that `_adopt`'s path uses when it later finalises. The verdict
        itself lives in `run_verdict.apply_verdict` so the two cannot
        drift again, which is how the original defect happened.

        Reads from offset 0: a fresh `ProgressSource` is constructed per
        call site and this is a one-shot read, so the whole file is
        consumed rather than a tail.

        Every read is best-effort. A missing, empty, truncated or
        unparseable file is not an error here -- it just means there is
        no evidence, and the correct finalisation for absent evidence is
        a failure, never a hopeful completion.
        """
        samples = []
        if run.id is not None:
            progress_path = self._artifacts.paths_for(run.id).progress
            try:
                samples = self._progress.read_new(progress_path)
            except OSError as exc:
                # Unreadable file (permissions, vanished mid-read). Same
                # outcome as no file: fail with no steps claimed.
                logger.warning(
                    "reconcile: cannot read progress of run %s (%s): %s",
                    run.id, progress_path, exc,
                )
                samples = []

        verdict = fold_samples(run, samples, self._clock)
        status_word = apply_verdict(
            run,
            verdict=verdict,
            exit_code=None,  # unreadable: we did not spawn it
            clock=self._clock,
            unreadable_exit_note=(
                "orphan cleanup: process already gone or no longer ours"
                + ("" if samples else "; its progress file holds nothing"
                   " readable either, so there is no evidence of progress")
            ),
        )
        logger.info(
            "reconciled run %s -> %s from its progress file (%d samples, "
            "verdict=%r, done_steps=%s)",
            run.id, status_word, len(samples), verdict, run.done_steps,
        )