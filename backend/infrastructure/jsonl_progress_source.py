"""JsonlProgressSource -- offset-tailed reader for log.progress.jsonl.

Faithful port of the legacy ``ProgressFileReader`` semantics minus its
side effects (it wrote rows and broadcast SSE itself): stateful per
path, appends only, one :class:`ProgressSample` per recognised line.

Phase mapping (trainer schema -> normalised sample):

- ``cache_start`` / ``cache`` / ``cache_done`` -> phase ``cache`` with
  cache counters (the legacy DB stored ``cache_done`` as a phase; we
  keep ``phase`` binary -- cache|training -- and carry the counters).
- ``training_start`` / ``step`` -> phase ``training`` (step samples
  carry loss/avg/lr and the trainer's authoritative total).
- terminal or unknown lines -> no sample (process exit decides status;
  telemetry must never assert it).

A file that shrinks (rotation/truncation) resets the offset instead of
stranding the tail forever -- a latent legacy bug fixed here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..application.ports.progress_source import ProgressSource, ProgressSample

logger = logging.getLogger(__name__)


class JsonlProgressSource(ProgressSource):
    def __init__(self) -> None:
        self._offsets: dict[Path, int] = {}

    def read_new(self, progress_path: Path) -> list[ProgressSample]:
        try:
            if not progress_path.exists():
                return []
            size = progress_path.stat().st_size
            offset = self._offsets.get(progress_path, 0)
            if size < offset:
                offset = 0  # truncated/rotated: start over
            if size == offset:
                return []
            samples: list[ProgressSample] = []
            with open(progress_path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(offset)
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sample = self._sample(data)
                    if sample is not None:
                        samples.append(sample)
                self._offsets[progress_path] = fh.tell()
            return samples
        except OSError as exc:
            logger.warning("cannot read progress %s: %s", progress_path, exc)
            return []

    @staticmethod
    def _sample(data: dict) -> ProgressSample | None:
        phase = data.get("phase")
        if phase == "cache_start":
            total = int(data.get("est_trajs") or 1)
            return ProgressSample(step=0, phase="cache", cache_done=0, cache_total=total)
        if phase == "cache":
            return ProgressSample(
                step=0,
                phase="cache",
                cache_done=int(data.get("done") or 0),
                cache_total=int(data.get("total") or 1),
            )
        if phase == "cache_done":
            total = int(data.get("total") or 1)
            return ProgressSample(
                step=0, phase="cache", cache_done=total, cache_total=total
            )
        if phase == "training_start":
            raw_total = data.get("total_steps")
            return ProgressSample(
                step=0,
                phase="training",
                total=int(raw_total) if raw_total is not None else None,
            )
        if phase == "step":
            raw_total = data.get("total")
            return ProgressSample(
                step=int(data.get("step") or 0),
                total=int(raw_total) if raw_total is not None else None,
                loss=data.get("loss"),
                avg=data.get("avg"),
                lr=data.get("lr"),
                phase="training",
            )
        return None  # terminal ("finished"/"error") or unknown: no telemetry
