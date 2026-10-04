"""PeakRecord -- what previous runs actually peaked at, remembered.

The half of observed mode that makes it safe, and the piece that is easy to
leave out. `DeviceReservations.observe()` ratchets a claim *within* a run,
which means the reservation is only as good as the steps that have already
happened -- and a run whose memory grows past its reservation is a run that
dies **mid-flight**, after however long it had already been going. That is the
worst place for the failure, and it is the reason a within-run ratchet is not
enough on its own.

So the peak is a **persistent** record keyed by a fingerprint of the
configuration, written after each step and *read at admission*. The run that
cares already knows what its predecessor peaked at, so:

  * step 0 is covered, because the reservation came from a real measurement of
    the same thing rather than from a guess made in advance;
  * nothing fails mid-run, because the reservation was already the right size
    when the run started;
  * a run that *does* exceed its remembered peak is not a resource problem --
    the configuration changed, or the card is busier than last time -- and is
    worth reporting as exactly that rather than as an out-of-memory error.

Measured on the B580, which is what makes a record worth keeping: rank-64
LoRA at 1024 with checkpointing peaks at 7,666 MB at batch 2 and 8,954 MB at
batch 4, and residents are constant at 5,611 MB, so the only thing a
fingerprint has to distinguish is the part that varies -- batch, resolution,
rank, and which model. An unknown fingerprint is **not** a zero peak. It means
nothing has been measured for this configuration, and under
`check_comfy_conflicts`' rule it blocks admission rather than admitting a run
whose peak is nominally nothing.

The record is a plain JSON file because that is what it is: a handful of
numbers that a person should be able to read, edit and delete. It is written
atomically (temp file, rename) because two runs finishing at once must not
leave half a file behind -- and this is a cache of measurements, so a lost or
truncated record costs a re-measure, not correctness.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

#: Extra MB held above a remembered peak before admitting a run. The same
#: number, and for the same reason, as
#: `device_reservations.OBSERVED_PILLOW_MB`: this is a measurement, so the
#: pillow covers allocator jitter (reserved drift measured at 0-14 MB across
#: runs on the B580) and not the uncertainty of an estimate.
DEFAULT_PILLOW_MB = 150.0


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """What makes two runs "the same configuration" for peak purposes.

    Only the things that change peak. Deliberately *not* a hash of the whole
    graph: two graphs that differ in a caption or a checkpoint path but train
    the same shapes at the same batch have the same peak, and folding that
    difference in would mean re-measuring for every dataset shuffle.
    """

    model: str
    batch_size: int
    latent_h: int
    latent_w: int
    rank: int
    checkpointing: bool
    optimizer: str

    def key(self) -> str:
        return "|".join(str(getattr(self, f)) for f in (
            "model", "batch_size", "latent_h", "latent_w", "rank",
            "checkpointing", "optimizer"))


class PeakRecord:
    """Remembered peaks, per configuration, on disk.

    Thread-safe within a process (a lock around read-modify-write) and safe
    across processes by writing atomically. Losing a concurrent update is
    possible and acceptable: the cost is one more run measured at the new
    number rather than the old, which is a slower admission, never a wrong
    one -- the read-modify-write is re-read under the lock and the file is
    replaced whole.
    """

    def __init__(self, path: Path | str, pillow_mb: float = DEFAULT_PILLOW_MB):
        self.path = Path(path)
        self.pillow_mb = pillow_mb
        self._lock = threading.Lock()

    # -- storage -----------------------------------------------------------

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A corrupt cache is a cache miss, not an error. Refusing to run
            # because a measurement file was truncated would make a
            # convenience into a dependency.
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(self.path.parent),
            prefix=self.path.name, suffix=".tmp", delete=False)
        try:
            with handle:
                json.dump(data, handle, indent=2, sort_keys=True)
            os.replace(handle.name, self.path)
        except BaseException:
            # Do not leave the temp file behind on any failure, including
            # Ctrl-C: an orphan .tmp next to the record is litter that looks
            # like state.
            Path(handle.name).unlink(missing_ok=True)
            raise

    # -- the two operations that matter -------------------------------------

    def reservation_mb(self, fingerprint: Fingerprint) -> float | None:
        """What to reserve for this configuration before starting it.

        None when nothing has been measured for it. **Not zero** -- zero is a
        claim that a run needs nothing, and admitting on that is how two runs
        end up believing the card is theirs.
        """
        peak = self._read().get(fingerprint.key())
        if peak is None:
            return None
        return float(peak) + self.pillow_mb

    def record(self, fingerprint: Fingerprint, peak_mb: float) -> float:
        """Remember a peak just measured. Monotonic. Returns the stored value.

        Monotonic for the same reason `DeviceReservations.observe` is: a peak
        is a high-water mark, and a later smaller number is a step that
        happened not to be the worst one, not evidence that less is needed.
        Lowering it here would let the *next* run be admitted on a number this
        run already exceeded.
        """
        with self._lock:
            data = self._read()
            key = fingerprint.key()
            previous = data.get(key)
            value = float(peak_mb) if previous is None else max(
                float(previous), float(peak_mb))
            data[key] = value
            self._write(data)
            return value

    def forget(self, fingerprint: Fingerprint) -> None:
        """Drop one configuration's measurement, so the next run re-measures.

        The escape hatch for a configuration whose peak legitimately changed --
        a bigger model, a different card. Forgetting is safe because the cost
        of being wrong in this direction is a slower admission, and the cost of
        being wrong in the other direction (keeping a stale high number
        forever) is a card that slowly becomes un-admittable.
        """
        with self._lock:
            data = self._read()
            if data.pop(fingerprint.key(), None) is not None:
                self._write(data)

    def known(self) -> dict[str, float]:
        """Every remembered peak, for a person to read."""
        return {k: float(v) for k, v in self._read().items()}
