"""ProgressSource port -- tail the trainer's JSONL progress file."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProgressSample:
    """One normalised telemetry record.

    ``None`` = field absent in that record (phase semantics differ:
    cache samples carry no step/loss, step samples carry no cache
    counters). The supervisor treats ``None`` as "leave unchanged".

    ``terminal`` carries the trainer's own verdict (``"finished"`` or
    ``"error"``) when it wrote one. It is *not* telemetry -- the running
    status is the supervisor's to write -- but it is the only exit
    evidence available for a process this server did not spawn (an
    adopted trainer after a restart), so the reader surfaces it
    separately (docs 07 F-11).
    """

    step: int | None = None
    total: int | None = None
    loss: float | None = None
    avg: float | None = None
    lr: float | None = None
    phase: str | None = None
    cache_done: int | None = None
    cache_total: int | None = None
    terminal: str | None = None


class ProgressSource(ABC):
    """Incremental reader: each call returns samples appended since the
    previous call for that path (stateful per path, offset-tailed)."""

    @abstractmethod
    def read_new(self, progress_path: Path) -> list[ProgressSample]:
        raise NotImplementedError
