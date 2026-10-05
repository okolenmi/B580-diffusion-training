"""PeakStore port -- remembered peaks per configuration (MEM-04).

The two operations the application needs out of the peak record, in its
own terms: admission *reads* what a previous run of this exact
configuration reached, and the server's watcher *writes* what a child
just reported. The server is the single writer (a child reports numbers,
the server files them -- rule 3's one-writer shape), so this port never
needs a cross-process lock of its own; ``infrastructure/
memory_peak_store.py`` implements it over SQLite with ``MAX`` semantics.

``peak_mb`` answers None for a configuration that has never been
measured. None is not 0.0: zero is a claim that a run needs nothing, and
admitting on that is how two runs end up believing the card is theirs
(task rule 2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class PeakStore(ABC):
    """Remembered peaks, keyed by configuration fingerprint."""

    @abstractmethod
    def record(self, fingerprint: str, peak_mb: float) -> float:
        """Remember a peak just measured; monotonic. Returns the stored value.

        Monotonic for the same reason `DeviceReservations.observe` is: a
        peak is a high-water mark, and a later smaller number is a step
        that happened not to be the worst one, not evidence that less is
        needed. Lowering it would let the *next* run be admitted on a
        number this run already exceeded.
        """
        raise NotImplementedError

    @abstractmethod
    def peak_mb(self, fingerprint: str) -> float | None:
        """The remembered peak for one configuration.

        None when it has never been measured -- never 0.0 (see the module
        docstring).
        """
        raise NotImplementedError


__all__ = ["PeakStore"]
