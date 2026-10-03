"""DeleteGraphExecutions -- wipe execution history.

Deletes every row, active ones included (mirrors ``DeleteRuns``): a
worker whose row disappears simply loses every subsequent CAS and exits
quietly -- no orphaned thread, no resurrected row.

And then sweeps the scratch, because a row is not the whole of a run.
Every execution leaves a graph.json, an events file and a log behind
(round-3 N3-07), and deleting only the rows left all three growing
forever while the UI said the history had been cleared. Measured at
845,756 bytes for one 4000-node execution.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..dto import DeleteGraphExecutionsResult
from ..event_publisher import EventPublisher
from ..ports.graph_execution_repository import GraphExecutionRepository
from .sweep_execution_scratch import SweepExecutionScratch
from ...domain.events import GraphExecutionsDeleted

logger = logging.getLogger(__name__)


class DeleteGraphExecutions:
    def __init__(
        self,
        executions: GraphExecutionRepository,
        events: EventPublisher,
        scratch: SweepExecutionScratch,
    ) -> None:
        self._executions = executions
        self._events = events
        self._scratch = scratch

    @property
    def scratch_dir(self) -> Path:
        """The directory this clears along with the rows."""
        return self._scratch.scratch_dir

    def execute(self) -> DeleteGraphExecutionsResult:
        deleted = self._executions.delete_all()
        if deleted:
            self._events.emit(GraphExecutionsDeleted(deleted=deleted))
        # Every row is gone, so every run's scratch is now orphaned and the
        # sweep takes all of it. Run even when `deleted == 0`: the rows may
        # have been deleted by something else -- a replaced database -- and
        # the files would otherwise have no way to be recognised as debris.
        swept = self._scratch.execute()
        if swept.files:
            logger.info(
                "deleted graph execution history also removed %d scratch "
                "file(s), %.1f MB", swept.files, swept.bytes / 1_000_000,
            )
        return DeleteGraphExecutionsResult(deleted=deleted)
