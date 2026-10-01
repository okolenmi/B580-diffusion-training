"""JsonlProgressSource -- offset-tailed reader for log.progress.jsonl.

Faithful port of the legacy ``ProgressFileReader`` semantics minus its
side effects (it wrote rows and broadcast SSE itself): stateful per
path, appends only, one :class:`ProgressSample` per recognised line.

Hostile input (docs 07 F-01/F-07): the trainer's file is outside this
process, so

- only NEWLINE-TERMINATED records are consumed; a torn tail line stays
  at the offset until the rest arrives (never parsed, never skipped,
  so a sample split across two writes cannot be lost);
- every field is coerced (wrong type, negative, bool -> the field's
  default): a malformed record degrades to "leave unchanged", it never
  raises out of ``read_new`` and never rewinds ``step`` to 0;
- a completed line that is not JSON, not an object, or otherwise
  unusable is skipped with a warning -- one bad record must not end
  the tail.

Offsets are BYTE positions (the file is opened in binary), so they
stay comparable with ``stat().st_size`` even for non-ASCII content --
the previous text-mode ``tell()`` could drift there.

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

# The trainer's own end-of-run verdicts (see ProgressSample.terminal).
_TERMINAL_PHASES = ("finished", "error")


def _int_field(value: object, default: int | None) -> int | None:
    """Coerce a count field to a non-negative int.

    Malformed (str that isn't a number, bool, None, negative) -> the
    caller's default; never raises. Negatives are rejected because the
    domain rejects them (a negative step would raise DomainError and
    kill the supervisor thread).
    """
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _float_field(value: object, default: float | None = None) -> float | None:
    """Coerce a telemetry float; malformed -> default (non-finite
    values pass through -- they are real data and get sanitised at the
    serialization boundary, docs 07 F-03)."""
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
            with open(progress_path, "rb") as fh:
                fh.seek(offset)
                chunk = fh.read()
            last_newline = chunk.rfind(b"\n")
            if last_newline == -1:
                return []  # torn tail line: wait for the rest (F-07)
            complete = chunk[: last_newline + 1]
            self._offsets[progress_path] = offset + len(complete)
            samples: list[ProgressSample] = []
            for raw in complete.split(b"\n"):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    logger.warning(
                        "skipping non-JSON progress line in %s: %r",
                        progress_path, raw[:160],
                    )
                    continue
                if not isinstance(data, dict):
                    logger.warning(
                        "skipping non-object progress line in %s: %r",
                        progress_path, raw[:160],
                    )
                    continue
                try:
                    sample = self._sample(data)
                except Exception:  # noqa: BLE001 -- one bad record must
                    logger.warning(  # never end the tail (F-01)
                        "unusable progress record in %s: %r",
                        progress_path, raw[:160],
                        exc_info=True,
                    )
                    continue
                if sample is not None:
                    samples.append(sample)
            return samples
        except OSError as exc:
            logger.warning("cannot read progress %s: %s", progress_path, exc)
            return []

    @staticmethod
    def _sample(data: dict) -> ProgressSample | None:
        # Every field goes through a coercing accessor: absent or
        # malformed -> None/default, which the supervisor treats as
        # "leave unchanged". Nothing here may raise (F-01).
        phase = data.get("phase")
        if phase == "cache_start":
            total = _int_field(data.get("est_trajs") or 1, 1)
            return ProgressSample(step=0, phase="cache", cache_done=0, cache_total=total)
        if phase == "cache":
            return ProgressSample(
                step=0,
                phase="cache",
                cache_done=_int_field(data.get("done") or 0, 0),
                cache_total=_int_field(data.get("total") or 1, 1),
            )
        if phase == "cache_done":
            total = _int_field(data.get("total") or 1, 1)
            return ProgressSample(
                step=0, phase="cache", cache_done=total, cache_total=total
            )
        if phase == "training_start":
            return ProgressSample(
                step=0,
                phase="training",
                total=_int_field(data.get("total_steps"), None),
            )
        if phase == "step":
            # step=None on malformed input: the run's progress stays
            # put instead of rewinding to a coerced 0.
            return ProgressSample(
                step=_int_field(data.get("step"), None),
                total=_int_field(data.get("total"), None),
                loss=_float_field(data.get("loss")),
                avg=_float_field(data.get("avg")),
                lr=_float_field(data.get("lr")),
                phase="training",
            )
        # Terminal lines assert nothing about *telemetry*, but they do
        # carry the trainer's own verdict on how it ended: that is the
        # only exit evidence there is for an adopted trainer, whose exit
        # code this process cannot read (docs 07 F-11).
        if phase in _TERMINAL_PHASES:
            return ProgressSample(terminal=phase)
        return None  # unknown line: no telemetry
