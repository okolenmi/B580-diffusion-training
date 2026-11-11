"""ExecutionAdmission -- what a graph run would hold on the device, and the claim for it.

One object for the two callers that admit a run: a start that goes straight to launch and the scheduler that
promotes a planned row later. Both need the same arithmetic (effective memory, then demand, then the ledger's
grant), and a second copy would be a second place to get the exploratory case wrong."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import MemoryUnavailableError
from .graph_peak_source import PeakSource
from .memory_admission import LedgerSource, admit, pending_owner
from .memory_ledger import Grant
from ..domain.graph import GraphDefinition
from ..domain.memory_settings import EffectiveMemory, effective_memory


@dataclass(frozen=True, slots=True)
class Claim:
    """The memory a run was admitted with and the provisional owner holding it (renamed once the row has an id)."""

    memory: EffectiveMemory
    grant: Grant
    owner: str


class ExecutionAdmission:
    def __init__(self, *, ledger: LedgerSource, peak_source: PeakSource) -> None:
        self._ledger = ledger
        self._peak_source = peak_source

    def memory_for(self, graph: GraphDefinition, overrides: dict | None) -> EffectiveMemory:
        """The effective settings and the demand known now (stated beats observed)."""
        ledger = self._ledger()
        observed = self._peak_source(graph)
        return effective_memory(
            graph.memory,
            overrides,
            peak_record=observed.peak_record(),
            fingerprint_key=observed.fingerprint_key,
            capacity_mb=ledger.capacity_mb if ledger is not None else None,
        )

    def claim(self, memory: EffectiveMemory) -> Claim:
        """Reserve for a run with these settings, or raise ``MemoryUnavailableError`` with the breakdown."""
        ledger = self._ledger()
        exploratory = memory.demand_source == "unknown"
        if ledger is None:
            # No ledger: admit() refuses with device-total-unknown before it looks at the demand.
            demand = 0.0
        elif exploratory or memory.demand_mb is None:
            # An unknown demand claims all free capacity as an exploratory exclusive run; it never becomes a zero claim.
            demand = memory.demand_mb if memory.demand_mb is not None else ledger.capacity_mb
        else:
            # Stated/observed are allocator MB; a grant is device MB (+ the per-process overhead).
            demand = memory.demand_mb + ledger.process_overhead_mb
        owner = pending_owner("graph")
        grant = admit(
            ledger, owner, demand,
            exploratory=exploratory,
            what="this graph execution",
            vram_min_mb=memory.vram_min_mb,
            source=self._ledger,
        )
        return Claim(memory, grant, owner)

    @staticmethod
    def worth_waiting(refusal: MemoryUnavailableError) -> bool:
        """Whether a refusal can clear by itself: someone holds memory that will come back. An unknown device or a run that
        does not fit an empty card never will, and is answered at once instead of planned."""
        details = refusal.details or {}
        return details.get("reason") != "device_total_unknown" and bool(details.get("holders"))

    def rename(self, claim: Claim, owner: str) -> None:
        ledger = self._ledger()
        if ledger is not None:
            ledger.rename(claim.owner, owner)

    def release(self, claim: Claim, owner: str | None = None) -> None:
        """Hand a claim back; both owner forms go, because release is idempotent and the rename may not have happened."""
        ledger = self._ledger()
        if ledger is None:
            return
        if owner is not None:
            ledger.release(owner)
        ledger.release(claim.owner)
