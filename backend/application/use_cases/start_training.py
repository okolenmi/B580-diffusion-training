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
from ..ports.config_inspector import ConfigInspector
from ..ports.event_bus import EventBus
from ..ports.run_artifacts import RunArtifacts
from ..ports.run_repository import RunRepository
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
        events: EventBus,
        gateway: TrainingGateway,
        inspector: ConfigInspector,
        artifacts: RunArtifacts,
        supervisor: "RunSupervisor",  # noqa: F821 -- application sibling
        clock: Clock,
        project_root: Path,
    ) -> None:
        self._runs = runs
        self._events = events
        self._gateway = gateway
        self._inspector = inspector
        self._artifacts = artifacts
        self._supervisor = supervisor
        self._clock = clock
        self._root = project_root
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

            config_path = self._resolve(command.config_path)
            summary = self._inspector.summarize(config_path)

            run = Run.create(
                config_path=command.config_path,
                mode=summary.mode,
                total_steps=summary.total_steps,
                created_at=self._clock.now(),
            )
            self._runs.add(run)

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

            try:
                pid = self._gateway.spawn(launch)
            except TrainingLaunchError as exc:
                # Launch failed before the process existed: finalise the
                # row here (created -> failed), publish, and re-raise so
                # the caller sees the envelope. Nothing to reap.
                run.mark_failed(at=self._clock.now(), error=str(exc))
                self._runs.update_if_status(run, expected=RunStatus.CREATED)
                self._publish(run)
                raise

            run.mark_started(pid=pid, at=self._clock.now())
            if not self._runs.update_if_status(run, expected=RunStatus.CREATED):
                # Unreachable while the lock is held (only reconciliation,
                # which runs before the server accepts requests, touches
                # created rows) -- but never leave a live process
                # attached to a row someone else reclaimed.
                logger.error(
                    "run %s was reclaimed during launch; killing pid %s",
                    run.id, pid,
                )
                self._gateway.kill(pid)
                raise TrainingLaunchError(
                    f"run {run.id} was reclaimed during startup launch"
                )

            self._publish(run)
            self._supervisor.watch(
                run_id=run.id,  # type: ignore[arg-type]
                pid=pid,
                progress_path=paths.progress,
            )
            return to_run_dto(run)

    def _resolve(self, raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else (self._root / path)

    def _publish(self, run: Run) -> None:
        for event in run.collect_events():
            self._events.publish(event)
