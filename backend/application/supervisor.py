"""RunSupervisor -- background watcher for a spawned run.

One daemon thread per active run (there is only ever one). The loop is
the run's single writer of *final* status:

1. Re-fetch the run every tick. If it is no longer ``running``
   (someone stopped or reconciled it), exit without writing -- that
   writer already won.
2. While the process is alive, tail the progress file; apply each
   sample to the entity and persist with a status CAS, then publish
   ``RunProgressed`` telemetry. A lost CAS means we raced a terminal
   writer: discard and exit.
3. When the process dies, read the exit code, finalise
   (completed / failed), publish lifecycle events, and append the
   ``--- RUN ENDED`` marker to the log -- but only if the CAS on
   ``running`` still succeeds. Losers write nothing.

Exceptions are logged, never raised into the thread: a crashed
supervisor must not take the server with it; reconciliation on the
next startup covers the leftover row.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .ports.clock import Clock
from .ports.event_bus import EventBus
from .ports.progress_source import ProgressSample, ProgressSource
from .ports.run_artifacts import RunArtifacts
from .ports.run_repository import RunRepository
from .ports.training_gateway import TrainingGateway
from ..domain.entities.run import Run
from ..domain.events import RunProgressed
from ..domain.value_objects import RunId, RunStatus

logger = logging.getLogger(__name__)


class RunSupervisor:
    def __init__(
        self,
        *,
        runs: RunRepository,
        events: EventBus,
        gateway: TrainingGateway,
        progress: ProgressSource,
        artifacts: RunArtifacts,
        clock: Clock,
        poll_interval: float = 0.5,
    ) -> None:
        self._runs = runs
        self._events = events
        self._gateway = gateway
        self._progress = progress
        self._artifacts = artifacts
        self._clock = clock
        self._poll = poll_interval

    def watch(
        self, *, run_id: RunId, pid: int, progress_path: Path
    ) -> threading.Thread:
        """Start watching a spawned run; returns the daemon thread."""
        thread = threading.Thread(
            target=self._guard,
            args=(run_id, pid, progress_path),
            name=f"backend-supervisor-{run_id}",
            daemon=True,
        )
        thread.start()
        return thread

    def _guard(self, run_id: RunId, pid: int, progress_path: Path) -> None:
        try:
            self._supervise(run_id, pid, progress_path)
        except Exception:  # noqa: BLE001 -- thread must not die silently
            logger.exception("supervisor for run %s crashed", run_id)

    def _supervise(self, run_id: RunId, pid: int, progress_path: Path) -> None:
        while True:
            run = self._runs.get(run_id)
            if run is None or run.status is not RunStatus.RUNNING:
                return  # stopped/reconciled elsewhere; that writer won
            if not self._gateway.is_alive(pid):
                break
            for sample in self._progress.read_new(progress_path):
                if not self._apply_sample(run, sample):
                    return  # CAS lost mid-batch: terminal writer won
            time.sleep(self._poll)
        self._finalize(run_id, pid)

    def _apply_sample(self, run: Run, sample: ProgressSample) -> bool:
        total = run.total_steps
        if sample.total is not None and sample.total > total:
            total = sample.total  # trainer discovered a larger total
        run.record_progress(
            done_steps=run.done_steps if sample.step is None else sample.step,
            at=self._clock.now(),
            current_loss=sample.loss,
            avg_loss=sample.avg,
            phase=sample.phase,
            total_steps=total,
            cache_done=sample.cache_done,
            cache_total=sample.cache_total,
        )
        if not self._runs.update_if_status(run, expected=RunStatus.RUNNING):
            return False
        self._events.publish(
            RunProgressed(
                run_id=run.id,  # type: ignore[arg-type]
                step=run.done_steps,
                total_steps=run.total_steps,
                loss=run.current_loss,
                avg_loss=run.avg_loss,
                lr=sample.lr,
                phase=run.phase,
                cache_done=run.cache_done,
                cache_total=run.cache_total,
                occurred_at=self._clock.now(),
            )
        )
        return True

    def _finalize(self, run_id: RunId, pid: int) -> None:
        exit_code = self._gateway.wait_exit_code(pid, timeout=5.0)
        run = self._runs.get(run_id)
        if run is None or run.status is not RunStatus.RUNNING:
            return  # stop request or reconcile beat us to the row
        at = self._clock.now()
        if exit_code == 0:
            run.mark_completed(at=at)
            status_word = "completed"
        elif exit_code is None:
            run.mark_failed(
                at=at, error="process died (exit code unavailable)"
            )
            status_word = "failed"
        else:
            run.mark_failed(
                at=at, error=f"Exit code {exit_code}", exit_code=exit_code
            )
            status_word = "failed"
        if not self._runs.update_if_status(run, expected=RunStatus.RUNNING):
            return
        for event in run.collect_events():
            self._events.publish(event)
        self._artifacts.append_log_note(
            run_id,
            f"--- RUN ENDED: status={status_word}, exit_code={exit_code} ---",
        )
