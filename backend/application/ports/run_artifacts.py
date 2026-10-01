"""RunArtifacts port -- where a run's files live on disk."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunArtifactsPaths:
    directory: Path
    log: Path
    progress: Path


class RunArtifacts(ABC):
    """Filesystem layout for one run.

    ``progress`` must equal the path the *child* derives for the same
    run id (``paths.get_progress_path``); the adapter guarantees that by
    delegating to the shared ``paths`` module in production.
    """

    @abstractmethod
    def prepare(self, run_id: int) -> RunArtifactsPaths:
        """Create the run directory if needed and return its paths."""
        raise NotImplementedError

    @abstractmethod
    def paths_for(self, run_id: int) -> RunArtifactsPaths:
        """Paths for an existing run (no directory creation required)."""
        raise NotImplementedError

    @abstractmethod
    def append_log_note(self, run_id: int, note: str) -> None:
        """Append a line to the run's log (post-exit marker, etc.).
        Must not raise on I/O failure -- it is observability, not state."""
        raise NotImplementedError

    @abstractmethod
    def tail_log(self, run_id: int, lines: int) -> str:
        """The last ``lines`` lines of the run's log, read from the end.

        Reading a growing log to show its tail is the classic way to turn
        a 300 MB file into a 300 MB request (docs 07 F-14), so the
        adapter walks backwards instead. A missing file answers ``""``
        (the child may not have written one yet).
        """
        raise NotImplementedError
