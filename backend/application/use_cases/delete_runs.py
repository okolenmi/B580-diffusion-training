"""DeleteRuns -- wipe run history and announce it as a domain event."""

from __future__ import annotations

from ..dto import DeleteRunsResult
from ..ports.event_bus import EventBus
from ..ports.run_repository import RunRepository
from ...domain.events import RunsDeleted


class DeleteRuns:
    def __init__(self, runs: RunRepository, events: EventBus) -> None:
        self._runs = runs
        self._events = events

    def execute(self) -> DeleteRunsResult:
        deleted = self._runs.delete_all()
        if deleted:
            self._events.publish(RunsDeleted(deleted=deleted))
        return DeleteRunsResult(deleted=deleted)
