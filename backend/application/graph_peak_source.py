"""Observed peaks for a graph about to start -- the read half of MEM-04.

Admission decides a run's demand from two facts that live in two
places: which configuration the graph *is* (the MEM-01 fingerprint, pure
and torch-free), and what a previous run of that configuration actually
reached (the peak store). ``GraphPeakSource`` composes the two into one answer
for ``StartGraphExecution``, which feeds it straight into
``effective_memory()`` as ``peak_record`` + ``fingerprint_key``.

Every failure mode lands in the same place: **unknown**, never zero.
No dataset reference, a fingerprint with a missing or wired field, a
store that will not open -- each
is an explicit ``ObservedPeak(None, None)`` with a logged warning, and
the start then claims exactly what it would have claimed with no peak
history at all (stated demand, or the exclusive exploratory claim).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from .memory_fingerprint import (
    MemoryFieldsResolver,
    UnknownFingerprint,
    graph_fingerprint,
)
from .ports.peak_store import PeakStore
from ..domain.graph import GraphDefinition

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ObservedPeak:
    """What a previous run of this exact configuration peaked at.

    Both fields carry explicit unknowns (task rule 2):

    * ``fingerprint_key`` None -- the graph's fingerprint cannot be
      computed, so there is nothing keyed to look up (and the watcher
      will have nothing to file this run's own peak under);
    * ``peak_mb`` None -- the fingerprint is known but has never been
      measured. **Not 0.0**: zero would be a claim that this
      configuration needs nothing.
    """

    fingerprint_key: str | None
    peak_mb: float | None

    def peak_record(self) -> dict[str, float] | None:
        """The one-entry record ``effective_memory`` reads.

        None when there is nothing to read -- an absent record and an
        absent key both leave the observed demand unknown, which is the
        only honest answer.
        """
        if self.fingerprint_key is None or self.peak_mb is None:
            return None
        return {self.fingerprint_key: self.peak_mb}


#: Where ``StartGraphExecution`` gets its observed peak. Mirrors
#: ``LedgerSource``: a plain callable the composition root wires, so the
#: use case depends on the answer, not on the machinery behind it.
PeakSource = Callable[[GraphDefinition], ObservedPeak]


def unknown_peaks(_graph: GraphDefinition) -> ObservedPeak:
    """A ``PeakSource`` for containers with no fingerprint inputs.

    Every graph's peaks are unknown -- the honest answer for a test
    composition and for any container that wired no store, never a zero
    peak.
    """
    return ObservedPeak(None, None)


def dataset_name_of(graph: GraphDefinition) -> str | None:
    """The dataset this graph trains on, as the dataset store names it.

    Read from the first node carrying a ``dataset`` param (the dataset pickers of the image and latent nodes). None
    when no node names one: such a graph has no fingerprint, never a defaulted one.
    """
    for node in graph.nodes:
        value = node.params.get("dataset")
        if isinstance(value, str) and value:
            return value
    return None


class GraphPeakSource:
    """Fingerprint the graph (MEM-01) and look its peak up (MEM-04 #2).

    One lookup per start, before the row exists: the answer travels into
    the row's ``memory_json`` so the watcher has the same key this
    decision used -- admission reads the past under it, the watcher files
    the run's own peak under it.
    """

    def __init__(
        self,
        *,
        peaks: PeakStore | None,
        resolve_memory_fields: MemoryFieldsResolver,
    ) -> None:
        # ``peaks`` None means the container has no store wired: the
        # fingerprint is still computed (the row needs its key), and
        # every peak read answers unknown rather than a zero.
        self._peaks = peaks
        self._resolve_memory_fields = resolve_memory_fields

    def observed(self, graph: GraphDefinition) -> ObservedPeak:
        name = dataset_name_of(graph)
        if name is None:
            logger.debug(
                "graph references no dataset; fingerprint unknown"
            )
            return ObservedPeak(None, None)
        try:
            fingerprint = graph_fingerprint(graph, name, resolve_memory_fields=self._resolve_memory_fields)
            if isinstance(fingerprint, UnknownFingerprint):
                logger.debug(
                    "fingerprint unknown for dataset %r: %s",
                    name,
                    fingerprint.reason,
                )
                return ObservedPeak(None, None)
            key = fingerprint.key()
            if self._peaks is None:
                return ObservedPeak(key, None)
            return ObservedPeak(key, self._peaks.peak_mb(key))
        except Exception:  # noqa: BLE001 -- admission must not fail on a lookup
            # A store the disk will not answer for means "nothing is
            # measured for this run", which claims strictly more capacity
            # than a remembered peak would -- the safe direction. The
            # warning carries the traceback because this is the one path
            # where a silent failure would quietly change what a run is
            # admitted against.
            logger.warning(
                "could not look up a remembered peak for dataset %r; "
                "treating this run as unmeasured",
                name,
                exc_info=True,
            )
            return ObservedPeak(None, None)


__all__ = [
    "GraphPeakSource",
    "ObservedPeak",
    "PeakSource",
    "dataset_name_of",
    "unknown_peaks",
]
