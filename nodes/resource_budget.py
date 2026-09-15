"""ResourceBudget: a stated VRAM ceiling for one run, plus a safety margin.

Domain-independent by construction, the same way core.py is -- no real
imports from any domain package, so it stays a valid downward dependency
for all of them (see core.py's own docstring for the same discipline
applied to Port/Node).

Split out of the former nodes/resource_policy.py, which also defined
ResourcePolicy/ManualResourcePolicy -- an abstraction meant to bundle
checkpointing/LoRA-scaling/parameter-group choices into one object.
That part was removed: no Node in the registry ever produced a
ResourcePolicy, so ComfyUNetLoRANode's `resource_policy` Port was
unreachable from the graph editor (only constructible by hand, in
Python, which only its own smoke test ever did) -- dead weight, not a
real extension point. ResourceBudget has no such problem: it's real,
live infrastructure for nodes/memory/ and nodes/model/checkpoint_placement.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceBudget:
    """A stated VRAM ceiling for one run, plus a safety margin.

    Measure against the allocator's *reserved* memory, not allocated --
    reserved includes the allocator's held-but-currently-unused pool, so
    it's what actually determines whether the OS hands back an
    out-of-memory error next; allocated alone can understate real
    pressure."""
    vram_budget_mb: float
    vram_reserve_mb: float = 512.0
