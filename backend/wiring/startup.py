"""What runs once before the server accepts requests: settle rows and files a dead process left behind."""

from __future__ import annotations

import logging

from ..application.services import ApplicationServices
from ..application.use_cases import SweepExecutionScratch

logger = logging.getLogger(__name__)


class StartupSweep:
    """Reconcile unfinished executions, then sweep their scratch -- in that order."""

    def __init__(self, services: ApplicationServices, scratch: SweepExecutionScratch) -> None:
        self._services = services
        self._scratch = scratch

    def run(self) -> None:
        self._reconcile()
        self._sweep_scratch()

    def _reconcile(self) -> None:
        # Nothing may observe an unfinished row from a dead process once the server accepts requests.
        reconciled = self._services.graphs.reconcile_executions.execute()
        if reconciled.cleaned:
            logger.info("reconciled %d unfinished graph execution(s) at startup", reconciled.cleaned)
        if reconciled.still_running:
            # At warning rather than info, and here rather than only in the use case's own log: this is the one
            # startup state that is not settled, and an operator who does not see it will find out by trying to
            # start a run and being told one is already active.
            logger.warning(
                "%d graph execution(s) left running at startup on a child that could not be adopted; their rows "
                "stay active on purpose, so the single-active check keeps refusing a second run. Each process is "
                "visible in `ps`.",
                reconciled.still_running,
            )

    def _sweep_scratch(self) -> None:
        # After the reconcile, not before: adopting an unfinished run reads its event file, so the sweep has to see
        # the difference between "finished" and "still going", and that is what the reconcile has just settled.
        swept = self._scratch.execute()
        if swept.files:
            logger.info(
                "removed scratch for %d finished graph execution(s) at startup: %d file(s), %.1f MB",
                swept.runs, swept.files, swept.bytes / 1_000_000,
            )
