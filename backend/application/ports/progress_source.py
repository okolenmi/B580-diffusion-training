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

    @property
    def is_terminal_only(self) -> bool:
        """A pure verdict line: the trainer's last word and no telemetry.

        The supervisor needs this to tell "the run ended" from "the run
        made progress". It used to ask by ``getattr``-ing eight field
        names, so adding a field or renaming one quietly turned the check
        into an always-true (docs 08 S-14).
        """
        return self.terminal is not None and not any(self.has_telemetry)

    @property
    def has_telemetry(self) -> tuple[bool, ...]:
        """Which telemetry fields this record carries (see the class
        docstring: ``None`` means the field was absent)."""
        return (
            self.step is not None,
            self.total is not None,
            self.loss is not None,
            self.avg is not None,
            self.lr is not None,
            self.phase is not None,
            self.cache_done is not None,
            self.cache_total is not None,
        )


class ProgressSource(ABC):
    """Incremental reader: each call returns samples appended since the
    previous call for that path (stateful per path, offset-tailed)."""

    @abstractmethod
    def read_new(self, progress_path: Path) -> list[ProgressSample]:
        raise NotImplementedError
