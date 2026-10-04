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
from ..memory_admission import LedgerSource, graph_owner, release
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
        memory_ledger: LedgerSource | None = None,
    ) -> None:
        self._executions = executions
        self._events = events
        self._scratch = scratch
        # Wiping history deletes *active* rows too, so their claims
        # must go with them -- otherwise capacity would stay spent on
        # rows that no longer exist. None only where the container has
        # no ledger (then no claim exists either). The composition
        # roots pass one.
        self._memory_ledger = memory_ledger

    @property
    def scratch_dir(self) -> Path:
        """The directory this clears along with the rows."""
        return self._scratch.scratch_dir

    def execute(self) -> DeleteGraphExecutionsResult:
        # Claims live only on unfinished rows (a finished row's claim
        # went back with `_finish`/reconcile), so unfinished is every
        # claim there can be. Release *before* the delete: afterwards
        # the ids are gone and the owner strings would be unjoinable.
        if self._memory_ledger is not None:
            for execution in self._executions.list_unfinished():
                if execution.id is not None:
                    release(self._memory_ledger, graph_owner(execution.id))
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
