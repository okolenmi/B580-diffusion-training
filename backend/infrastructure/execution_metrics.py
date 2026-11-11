"""File-backed `ExecutionMetrics`: one small JSON per run next to the supervision scratch.

An archive is ``metrics/execution_<id>.json`` = ``{stream: [frame, ...]}``. Frames are thinned on the way in so a
twelve-hour run costs about the same as a short one; a chart cannot show more points than it has pixels anyway.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from ..application.ports.graph_task_stream import EventKind
from .graph_event_stream import ExecutionEventTail

logger = logging.getLogger(__name__)

#: Frames kept per stream. Evenly spaced, so a curve keeps its shape.
MAX_FRAMES = 4000
_ARCHIVE = re.compile(r"^execution_(\d+)\.json$")


def thin(frames: list[dict[str, Any]], limit: int = MAX_FRAMES) -> list[dict[str, Any]]:
    """At most `limit` frames, evenly spaced, first and last included."""
    if len(frames) <= limit:
        return frames
    if limit < 2:
        return frames[-1:]
    step = (len(frames) - 1) / (limit - 1)
    return [frames[round(n * step)] for n in range(limit)]


class FileExecutionMetrics:
    def __init__(self, scratch_dir: Path) -> None:
        self._scratch = scratch_dir
        self._dir = scratch_dir / "metrics"

    def _archive_path(self, execution_id: int) -> Path:
        return self._dir / f"execution_{execution_id}.json"

    def _events_path(self, execution_id: int) -> Path:
        return self._scratch / f"execution_{execution_id}.events.jsonl"

    def _from_events(self, execution_id: int) -> dict[str, list[dict[str, Any]]]:
        path = self._events_path(execution_id)
        if not path.is_file():
            return {}
        frames: dict[str, list[dict[str, Any]]] = {}
        for event in ExecutionEventTail(path).poll():
            if event.kind is EventKind.MONITOR:
                frames.setdefault(str(event.payload.get("monitor_id")), []).append(event.payload.get("data") or {})
        return frames

    def _load(self, execution_id: int) -> dict[str, list[dict[str, Any]]]:
        archive = self._archive_path(execution_id)
        if archive.is_file():
            try:
                return json.loads(archive.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.warning("unreadable metrics archive %s", archive, exc_info=True)
                return {}
        return self._from_events(execution_id)

    def archive(self, execution_id: int) -> bool:
        if self._archive_path(execution_id).is_file():
            return True
        frames = {stream: thin(rows) for stream, rows in self._from_events(execution_id).items() if rows}
        if not frames:
            return False
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            target = self._archive_path(execution_id)
            partial = target.with_suffix(".tmp")
            partial.write_text(json.dumps(frames, allow_nan=False, separators=(",", ":")), encoding="utf-8")
            os.replace(partial, target)
        except (OSError, ValueError):
            logger.warning("could not archive metrics of execution %s", execution_id, exc_info=True)
            return False
        return True

    def streams(self, execution_id: int) -> list[str]:
        return sorted(self._load(execution_id))

    def read(self, execution_id: int, stream: str) -> list[dict[str, Any]]:
        return thin(self._load(execution_id).get(stream, []))

    def ids(self) -> list[int]:
        if not self._dir.is_dir():
            return []
        return sorted(int(m.group(1)) for p in self._dir.iterdir() if (m := _ARCHIVE.match(p.name)))

    def remove(self, execution_id: int) -> None:
        try:
            self._archive_path(execution_id).unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("could not remove metrics archive of execution %s", execution_id, exc_info=True)
