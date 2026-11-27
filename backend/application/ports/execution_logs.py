"""The console output of a run's child process, kept on disk for runs that ended badly (see `SweepExecutionScratch`)."""

from __future__ import annotations

from typing import Protocol


class ExecutionLogs(Protocol):
    def tail(self, execution_id: int, max_bytes: int) -> str | None:
        """The last ``max_bytes`` of the run's log as text, or None when there is no log (a finished run's is removed)."""
