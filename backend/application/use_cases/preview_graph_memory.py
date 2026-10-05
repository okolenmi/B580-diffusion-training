"""PreviewGraphMemory -- what admission would decide, without admitting.

The editor's memory panel has to answer three questions about a graph
that is still being edited: what did this configuration peak at last
time, is there anything remembered at all, and would a run fit right
now. None of that can be answered client-side -- the fingerprint needs
the server (declared ``memory_fields`` plus the dataset's largest
bucket), and the fit answer needs the live ledger.

So this is a dry run of the *same* arithmetic ``StartGraphExecution``
performs, taking the very same ``PeakSource`` it takes and calling the
very same ``effective_memory``, rather than a second implementation of
it. A preview that could disagree with real admission would be worse
than no preview at all: it would promise a run fits and then refuse it.

Nothing is reserved, no row is written, no child is spawned. Every number
that cannot be known is ``None`` and says so; none is guessed at (task
rule 2):

* ``fingerprint_key`` None -- the graph's fingerprint could not be
  computed, so there is nothing to key a remembered peak by;
* ``peak_mb`` None -- the fingerprint is known but has never been
  measured. Not 0.0, which would claim the configuration needs nothing;
* ``verdict`` "unknown" -- no ledger, so the device total has never been
  read and there is nothing to fit against;
* ``verdict`` "exploratory" -- the *demand* is unknown, so a run would
  claim all free capacity and could start only when nothing else holds
  the card. That is neither a fit nor a refusal, and reporting "fits"
  would be the wrong kind of reassuring.
"""

from __future__ import annotations

from ..dto import GraphMemoryPreview
from ..graph_peak_source import PeakSource
from ..memory_admission import LedgerSource
from ...domain.graph import GraphDefinition
from ...domain.memory_settings import effective_memory

#: What the panel can be told. Named rather than a bare bool so the UI
#: cannot invent a fourth meaning for an unexpected string.
VERDICT_FITS = "fits"
VERDICT_DOES_NOT_FIT = "does_not_fit"
VERDICT_UNKNOWN = "unknown"
VERDICT_EXPLORATORY = "exploratory"


class PreviewGraphMemory:
    def __init__(self, *, memory_ledger: LedgerSource,
                 peak_source: PeakSource) -> None:
        self._memory_ledger = memory_ledger
        self._peak_source = peak_source

    def execute(self, graph: GraphDefinition) -> GraphMemoryPreview:
        ledger = self._memory_ledger()
        observed = self._peak_source(graph)
        memory = effective_memory(
            graph.memory,
            None,
            peak_record=observed.peak_record(),
            fingerprint_key=observed.fingerprint_key,
            capacity_mb=ledger.capacity_mb if ledger is not None else None,
        )
        exploratory = memory.demand_source == "unknown"

        if ledger is None:
            # No ledger: admission refuses on device-total-unknown before
            # it ever looks at the demand, so the preview says the same.
            return GraphMemoryPreview(
                fingerprint_key=observed.fingerprint_key,
                peak_mb=observed.peak_mb,
                demand_mb=memory.demand_mb,
                demand_source=memory.demand_source,
                exploratory=exploratory,
                device_demand_mb=None,
                verdict=VERDICT_UNKNOWN,
                reason="the device total is unknown, so nothing can be checked",
                capacity_mb=None,
                free_mb=None,
                held_mb=None,
                foreign_reserve_mb=None,
                holders={},
            )

        if exploratory or memory.demand_mb is None:
            # The same branch admission takes: an unknown demand claims
            # all free capacity as an exploratory exclusive run.
            device_demand = (
                memory.demand_mb
                if memory.demand_mb is not None
                else ledger.capacity_mb
            )
        else:
            # Stated/observed are allocator MB; a grant is device MB
            # (task rule 4: plus the per-process overhead).
            device_demand = memory.demand_mb + ledger.process_overhead_mb

        free = ledger.free_mb()
        if exploratory:
            verdict = VERDICT_EXPLORATORY
            reason = (
                "this configuration has never been measured, so a run would "
                "claim the whole free card and start only if nothing else "
                "is using it"
            )
        elif device_demand <= free:
            verdict = VERDICT_FITS
            reason = None
        else:
            verdict = VERDICT_DOES_NOT_FIT
            reason = (
                f"needs {device_demand:.0f} MB but only {free:.0f} MB is free"
            )

        snapshot = ledger.snapshot()
        return GraphMemoryPreview(
            fingerprint_key=observed.fingerprint_key,
            peak_mb=observed.peak_mb,
            demand_mb=memory.demand_mb,
            demand_source=memory.demand_source,
            exploratory=exploratory,
            device_demand_mb=device_demand,
            verdict=verdict,
            reason=reason,
            capacity_mb=ledger.capacity_mb,
            free_mb=free,
            held_mb=ledger.held_mb(),
            foreign_reserve_mb=ledger.foreign_reserve_mb,
            holders=snapshot.get("holders", {}),
        )