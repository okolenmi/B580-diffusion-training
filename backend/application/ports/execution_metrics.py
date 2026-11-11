"""What a finished run reported to its monitors, kept after the run's scratch is swept.

The supervisor's event file holds every monitor frame of a run, but it is deleted once the row is terminal (see
`SweepExecutionScratch`). Charts of a past run are what is worth keeping from it, so the sweep hands each terminal run's
frames to this store first. A reader asks for one run's frames of one stream and gets them from the archive, or from the
event file while the run is live or not swept yet.
"""

from __future__ import annotations

from typing import Any, Protocol


class ExecutionMetrics(Protocol):
    def archive(self, execution_id: int) -> bool:
        """Keep this run's monitor frames; True when there was something to keep. Safe to repeat."""

    def streams(self, execution_id: int) -> list[str]:
        """Names of the streams the run reported on (e.g. ``node:<uid>``), empty when nothing is recorded."""

    def read(self, execution_id: int, stream: str) -> list[dict[str, Any]]:
        """The frames of one stream in the order reported, thinned to a bounded count (the last one is always kept)."""

    def ids(self) -> list[int]:
        """Runs that have an archive, so a sweep can drop the ones whose row is gone."""

    def remove(self, execution_id: int) -> None:
        """Drop one run's archive; no error when there is none."""
