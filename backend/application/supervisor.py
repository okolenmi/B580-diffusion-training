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
3. When the process dies, drain the progress file ONE final time
   (lines written in the death tick must not be lost), then read the
   exit code, finalise (completed / failed), publish lifecycle events,
   and append the ``--- RUN ENDED`` marker to the log -- but only if
   the CAS on ``running`` still succeeds. Losers write nothing.

Exceptions are logged, never raised into the thread: a crashed
supervisor must not take the server with it. A crash still has to
repair its row -- ``_guard`` fails the leftover (and stops the
unwatched trainer) exactly as ``GraphExecutionSupervisor._guard``
does, so a stuck ``running`` row can never block the next start until
restart (docs 07 F-01).
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
        self._lock = threading.Lock()
        # RunIds we re-attached to rather than spawned: their exit code
        # is not readable, so finalisation uses the trainer's own
        # terminal line instead (docs 07 F-11).
        self._adopted: set[RunId] = set()
        self._terminal: dict[RunId, str] = {}

    def watch(
        self, *, run_id: RunId, pid: int, progress_path: Path
    ) -> threading.Thread:
        """Start watching a spawned run; returns the daemon thread."""
        return self._start(run_id, pid, progress_path, adopted=False)

    def adopt(
        self, *, run_id: RunId, pid: int, progress_path: Path
    ) -> threading.Thread:
        """Re-attach to a trainer this process did not spawn.

        Trainers are started in their own session so they survive a
        server restart; the legacy server killed those orphans anyway,
        losing hours of work (docs 07 F-11). Adoption keeps the run
        honest about what is knowable:

        * the progress file is tailed from *here* -- the history it
          already holds is consumed without being applied, so an adopted
          run cannot rewind to a step from before the restart;
        * the exit code is unreadable (the process is not our child), so
          the trainer's own terminal line decides completed vs failed;
        * a log note records the re-attachment.
        """
        self._progress.read_new(progress_path)  # consume history, apply none
        return self._start(run_id, pid, progress_path, adopted=True)

    def _start(
        self, run_id: RunId, pid: int, progress_path: Path, *, adopted: bool
    ) -> threading.Thread:
        with self._lock:
            if adopted:
                self._adopted.add(run_id)
        try:
            self._artifacts.append_log_note(
                run_id,
                f"--- RUN {'REAPTIED (adopted after a server restart)' if adopted else ''}"
                f" -- server watching pid {pid} ---",
            )
        except Exception:  # noqa: BLE001 -- the note must never block watching
            logger.exception("could not append the watch note for run %s", run_id)
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
            self._fail_leftover(run_id, pid)

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
        # The process is gone: its final lines may have landed after
        # the last in-loop read (or inside the very tick that saw the
        # death) -- drain once more so completion records the true last
        # sample instead of the previous poll's (docs 07 F-07).
        if not self._drain(run_id, progress_path):
            return  # terminal writer won during the drain
        self._finalize(run_id, pid)

    def _drain(self, run_id: RunId, progress_path: Path) -> bool:
        """One final read after process death. Returns False when a
        terminal writer already owns the row (skip finalisation)."""
        run = self._runs.get(run_id)
        if run is None or run.status is not RunStatus.RUNNING:
            return False
        for sample in self._progress.read_new(progress_path):
            if not self._apply_sample(run, sample):
                return False
        return True

    def _apply_sample(self, run: Run, sample: ProgressSample) -> bool:
        if sample.terminal is not None:
            self._terminal[run.id] = sample.terminal  # type: ignore[index]
        if sample.terminal is not None and all(
            getattr(sample, field) is None
            for field in ("step", "total", "loss", "avg", "lr", "phase",
                          "cache_done", "cache_total")
        ):
            # A pure terminal line: nothing to apply, but it is evidence
            # -- and the client should hear about the run's end through
            # the normal lifecycle path, not through telemetry.
            return True
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
        adopted = run_id in self._adopted
        verdict = self._terminal.get(run_id)
        self._adopted.discard(run_id)
        self._terminal.pop(run_id, None)
        if adopted:
            # An adopted trainer is not our child: its exit code cannot
            # be read, so the trainer's own last word decides -- and a
            # missing word is a failure, never a hopeful "completed"
            # (docs 07 F-11).
            exit_code = None  # honest in the marker: unreadable, not zero
            if verdict == "finished":
                run.mark_completed(at=at)
                status_word = "completed"
            else:
                run.mark_failed(
                    at=at,
                    error=(
                        "trainer reported an error"
                        if verdict == "error"
                        else "process exited with no terminal progress line "
                             "(adopted after a server restart; its exit "
                             "code cannot be read)"
                    ),
                )
                status_word = "failed"
        elif exit_code == 0:
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

    def _fail_leftover(self, run_id: RunId, pid: int) -> None:
        """Last-ditch row repair after a supervisor crash: never leave
        the row ``running`` (it would block the next start until
        restart, docs 07 F-01), and never leave a live trainer nobody
        watches (two trainers on one device)."""
        try:
            run = self._runs.get(run_id)
            if run is None or run.status.is_terminal:
                return  # already finalised by another writer
            expected = run.status
            run.mark_failed(
                at=self._clock.now(),
                error="run supervisor crashed (see server log)",
            )
            if not self._runs.update_if_status(run, expected=expected):
                return  # stop/reconcile won the row; their pid handling stands
            for event in run.collect_events():
                self._events.publish(event)
            try:
                self._artifacts.append_log_note(
                    run_id,
                    "--- RUN ENDED: status=failed, supervisor crash ---",
                )
            except Exception:  # noqa: BLE001 -- repair is best-effort
                logger.exception("log note after supervisor crash failed (non-fatal)")
            # If the trainer is still alive, stop it gracefully (a dead
            # pid is a no-op): the row says failed, so nothing watches
            # this process any more.
            self._gateway.stop(pid)
        except Exception:  # noqa: BLE001 -- already in the crash path
            logger.exception(
                "could not fail leftover run %s after supervisor crash", run_id
            )
