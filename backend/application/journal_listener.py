"""What a run's life says in the journal: queued, started, how it ended and why."""

from __future__ import annotations

from ..domain.events import (
    GraphExecutionFailed, GraphExecutionFinished, GraphExecutionQueued, GraphExecutionsDeleted, GraphExecutionStarted, GraphExecutionStopped,
)
from .ports.event_bus import Sequenced
from .ports.journal import Journal


class JournalListener:
    """Subscribed to the event bus; one line per lifecycle event of a run."""

    def __init__(self, journal: Journal) -> None:
        self._journal = journal

    def __call__(self, item: Sequenced) -> None:
        event = item.event
        if isinstance(event, GraphExecutionQueued):
            self._say("info", event.execution_id, f"queued ({event.node_count} nodes)")
        elif isinstance(event, GraphExecutionStarted):
            self._say("info", event.execution_id, "started")
        elif isinstance(event, GraphExecutionFinished):
            self._say("success", event.execution_id, "finished")
        elif isinstance(event, GraphExecutionFailed):
            self._say("error", event.execution_id, f"failed: {event.error or 'no reason was given'}")
        elif isinstance(event, GraphExecutionStopped):
            self._say("warn", event.execution_id, f"stopped{': ' + event.reason if event.reason else ''}")
        elif isinstance(event, GraphExecutionsDeleted):
            self._journal.add("info", "server", f"deleted {event.deleted} run{'s' if event.deleted != 1 else ''} from the history")

    def _say(self, level: str, execution_id: int, text: str) -> None:
        self._journal.add(level, f"run #{execution_id}", text)
