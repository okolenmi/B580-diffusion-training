"""StopTraining -- graceful stop (or force) of the running run."""

from __future__ import annotations

from ..dto import RunDTO, to_run_dto
from ..errors import RunNotFoundError, RunNotRunningError
from ..ports.clock import Clock
from ..event_publisher import EventPublisher
from ..ports.run_repository import RunRepository
from ..ports.training_gateway import TrainingGateway
from ...domain.value_objects import RunStatus


class StopTraining:
    """Signal first, then claim the cancellation.

    Order matters: the process may be finishing on its own. We signal,
    then ``cancel`` + compare-and-swap on RUNNING. If the supervisor
    (seeing the process exit) wins the CAS first, this use case reports
    ``run_not_running`` with the winner's status -- honest, never
    silently mislabels a completed run as cancelled.
    """

    def __init__(
        self,
        *,
        runs: RunRepository,
        events: EventPublisher,
        gateway: TrainingGateway,
        clock: Clock,
    ) -> None:
        self._runs = runs
        self._events = events
        self._gateway = gateway
        self._clock = clock

    def execute(self, run_id: int, *, force: bool = False) -> RunDTO:
        run = self._runs.get(run_id)
        if run is None:
            raise RunNotFoundError(f"run {run_id} does not exist")
        if run.status is not RunStatus.RUNNING:
            raise RunNotRunningError(
                f"run {run_id} is {run.status.value}, not running"
            )

        if run.pid is not None:
            self._gateway.stop(run.pid, force=force)

        reason = "stop requested" + (" (force)" if force else "")
        run.cancel(at=self._clock.now(), reason=reason)
        if not self._runs.update_if_status(run, expected=RunStatus.RUNNING):
            fresh = self._runs.get(run_id)
            won = fresh.status.value if fresh else "?"
            raise RunNotRunningError(
                f"run {run_id} already finished as {won}"
            )

        self._events.publish(run)
        return to_run_dto(run)
