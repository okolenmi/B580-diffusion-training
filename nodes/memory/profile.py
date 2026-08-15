"""ResourceProfile: one aggregate, per-component VRAM snapshot.

See docs/training_pipeline_design.md section 5.5 for the rationale.
profile=True's existing report (nodes/train/step_pipeline.py's
MonitoringPhase) already has vram_allocated_mb/vram_reserved_mb (the
driver's own numbers, no per-component breakdown) and
tracked_footprint_mb (ResourceCoordinator.total_footprint_bytes(), a
single rolled-up total) -- neither answers "how much of my VRAM is the
text encoder cache vs. optimizer scratch vs. the model itself," which is
exactly the question docs/suspicious_findings.md's open
DeviceResident.footprint_bytes()/VRAM-pressure entries need answered.
This is that: a plain snapshot combining ResourceCoordinator's
per-resident breakdown, MemoryManager's per-tag pool stats, and
DeviceContext's own allocator stats into one object, so a caller (or a
person reading a profile=True report) sees all three side by side
instead of piecing them together from three separate call sites by hand.

Domain-independent by construction, the same discipline
nodes/resource_policy.py already established for a cross-domain-typed
object: DeviceContext (nodes/components/device.py, a domain package per
section 5.7) is referenced by forward-reference string type hint only
in capture()'s signature, so this module needs no real import from
components/ and stays a valid downward dependency for every domain
package -- matching nodes/memory/'s existing "domain-independent, lives
next to core.py" rule (see nodes/memory/manager.py's own docstring).
ResourceCoordinator/MemoryManager are real imports; both already live in
this same package, so importing them isn't a sideways-domain dependency.

Known real gap, not papered over: capture()'s `memory` argument is
optional because there is currently no single shared MemoryManager
instance reachable from SupervisedLoRATrainerNode.build() to pass in --
nodes/optimizer/strategies/chunked.py's ChunkedScratchBufferStrategy
constructs its own private MemoryManager(use_mempool=...) when none is
given (its own docstring says so explicitly), and nothing upstream of it
currently injects a shared one. Threading a shared MemoryManager through
optimizer construction so this field is actually populated in a real
SupervisedLoRATrainerNode run is real, separate work -- not done here
(this task is ResourceProfile itself, section 5.5; that gap belongs with
whichever future change first needs a shared MemoryManager for its own
reasons, budget-aware allocation being the most likely one). Until then,
memory_manager_stats is None in any real trainer-node profile, exactly
the way DeviceContext.memory_stats() already returns None for a backend
with no such concept -- same "None means not applicable here yet, not
zero" convention, not a new one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .coordinator import ResourceCoordinator
from .manager import MemoryManager


@dataclass(frozen=True)
class ResourceProfile:
    """A snapshot, not a subscription -- call capture() again for a new
    one; nothing here stays live. per_resident_bytes is keyed by
    whatever names capture()'s ResourceCoordinator was given at
    register() time."""

    per_resident_bytes: dict[str, int]
    memory_manager_stats: Optional[dict[str, Any]]
    allocator_stats: Optional[dict[str, float]]

    @classmethod
    def capture(cls, coordinator: ResourceCoordinator,
                memory: Optional[MemoryManager],
                device_ctx: "DeviceContext") -> "ResourceProfile":
        """memory=None is a real, expected case right now (see module
        docstring) -- memory_manager_stats is simply None in that case,
        not an error and not a 0. device_ctx.memory_stats() is likewise
        already Optional (None on CPU/no-allocator-concept backends) --
        passed straight through unchanged, this method makes no
        decisions about what "no data" means for either field, it only
        aggregates what each source already reports."""
        return cls(
            per_resident_bytes=coordinator.per_resident_footprint_bytes(),
            memory_manager_stats=memory.stats() if memory is not None else None,
            allocator_stats=device_ctx.memory_stats(),
        )
