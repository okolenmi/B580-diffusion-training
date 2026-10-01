"""StartTraining -- validate, persist, spawn, and watch one run."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from ..dto import RunDTO, StartTrainingCommand, to_run_dto, START_FROM_OPTIONS
from ..errors import (
    InvalidQueryError,
    RunAlreadyActiveError,
    TrainingLaunchError,
)
from ..ports.clock import Clock
from ..project_paths import ProjectPaths
from ..ports.config_inspector import ConfigInspector
from ..lifecycle_writer import RunLifecycleWriter
from ..ports.run_artifacts import RunArtifacts
from ..ports.run_repository import RunRepository
from ..ports.run_watcher import RunWatcher
from ..ports.training_gateway import TrainingGateway, TrainingLaunch
from ...domain.entities.run import Run
from ...domain.value_objects import RunStatus

logger = logging.getLogger(__name__)


class StartTraining:
    """The only place a run is born.

    Single-run invariant: check-then-act under a lock (this backend is
    one process; the repository additionally guards status transitions
    with compare-and-swap, but two concurrent starts must not both pass
    the active check before either creates its row).
    """

    def __init__(
        self,
        *,
        runs: RunRepository,
        writer: RunLifecycleWriter,
        gateway: TrainingGateway,
        inspector: ConfigInspector,
        artifacts: RunArtifacts,
        watcher: RunWatcher,
        clock: Clock,
        paths: ProjectPaths,
    ) -> None:
        self._runs = runs
        self._writer = writer
        self._gateway = gateway
        self._inspector = inspector
        self._artifacts = artifacts
        self._watcher = watcher
        self._clock = clock
        self._paths = paths
        self._lock = threading.Lock()

    def execute(self, command: StartTrainingCommand) -> RunDTO:
        with self._lock:
            active = self._runs.find_active()
            if active is not None:
                raise RunAlreadyActiveError(
                    f"run {active.id} is still {active.status.value}"
                )
            if command.start_from not in START_FROM_OPTIONS:
                raise InvalidQueryError(
                    f"unknown start_from {command.start_from!r}; "
                    f"expected one of {list(START_FROM_OPTIONS)}"
                )

            config_path = self._paths.config(command.config_path)
            summary = self._inspector.summarize(config_path)

            run = Run.create(
                config_path=command.config_path,
                mode=summary.mode,
                total_steps=summary.total_steps,
                created_at=self._clock.now(),
            )
            self._runs.add(run)

            # Everything from the insert to the watcher handover can
            # fail (artifacts.prepare, spawn, publish, thread start).
            # Any failure must repair the row before re-raising: a row
            # left `created` (or `running` with no watcher) blocks
            # every future start until restart (docs 07 F-02).
            pid: int | None = None
            try:
                paths = self._artifacts.prepare(run.id)  # type: ignore[arg-type]
                launch = TrainingLaunch(
                    run_id=run.id,  # type: ignore[arg-type]
                    config_path=config_path,
                    mode=summary.mode,
                    total_steps=summary.total_steps,
                    start_from=command.start_from,
                    reset_optimizer=command.reset_optimizer,
                    log_path=paths.log,
                    progress_path=paths.progress,
                )
                pid = self._gateway.spawn(launch)
                run.mark_started(pid=pid, at=self._clock.now())
                if not self._writer.commit(run, expected=RunStatus.CREATED):
                    # Unreachable while the lock is held (only
                    # reconciliation, which runs before the server
                    # accepts requests, touches created rows) -- but
                    # never leave a live process attached to a row
                    # someone else reclaimed.
                    logger.error(
                        "run %s was reclaimed during launch; killing pid %s",
                        run.id, pid,
                    )
                    self._gateway.kill(pid)
                    raise TrainingLaunchError(
                        f"run {run.id} was reclaimed during startup launch"
                    )
                self._watcher.watch(
                    run_id=run.id,  # type: ignore[arg-type]
                    pid=pid,
                    progress_path=paths.progress,
                )
            except TrainingLaunchError as exc:
                self._repair_failed_start(run, pid, exc)
                raise
            except Exception as exc:
                logger.exception("startup of run %s failed", run.id)
                self._repair_failed_start(run, pid, exc)
                raise
            return to_run_dto(run)

    def _repair_failed_start(self, run: Run, pid: int | None, exc: Exception) -> None:
        """Terminal-state repair for a failed start sequence: created
        -> failed (so a retry is accepted), or -- when the row already
        reached running but the watcher never attached -- kill the
        orphan and fail it. Reclaims by other writers are left alone."""
        try:
            current = self._runs.get(run.id)
            if current is None or current.status.is_terminal:
                return
            error = str(exc) or type(exc).__name__
            if current.status is RunStatus.CREATED:
                current.mark_failed(at=self._clock.now(), error=error)
                # `prior=(run,)` because the launch entity still buffers
                # RunCreated: the stream must read created -> failed, not
                # the other way round.
                self._writer.commit(
                    current, expected=RunStatus.CREATED, prior=(run,)
                )
                return
            if current.status is RunStatus.RUNNING and pid is not None:
                current.mark_failed(
                    at=self._clock.now(), error=f"startup handover failed: {error}"
                )
                if self._writer.commit(
                    current, expected=RunStatus.RUNNING, prior=(run,)
                ):
                    self._gateway.kill(pid)  # nobody watches this process
        except Exception:  # noqa: BLE001 -- already on the failure path
            logger.exception("could not repair run %s after failed start", run.id)

