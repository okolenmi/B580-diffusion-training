"""VRAMBudgetControllerNode: wire this into a trainer's own resource_control
input (matching how a MonitorNode wires into `monitor`) to give it a live
VRAM budget to actually stay under, not just a footprint number to log.

One concrete node, no ABC layer above it the way MonitorNode has one
below TrainingProgressMonitorNode -- there's exactly one way to build a
ResourceControlHandle today (a stated budget), not a family of them
yet; add the ABC split if and when a second one genuinely shows up,
same as everywhere else in this codebase that waited for a real second
case before generalizing.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Node, Port
from ..resource_policy import ResourceBudget
from .control_handle import BudgetedResourceControlHandle, ResourceControlHandle


class VRAMBudgetControllerNode(Node):
    """Produces a live budget enforcer a trainer calls into at its own
    step boundaries -- see nodes/memory/control_handle.py's own
    docstring for why this is a handle a node constructs and another
    consumes, not a second node running "alongside" the trainer.
    Nothing is registered with it yet at construction time: the
    trainer registers its own resources (model, optimizer, text
    encoder, ...) once it actually builds them, deciding for itself
    which of those it's safe to mark offloadable.
    """

    INPUTS: ClassVar[dict[str, Port]] = {
        "vram_budget_mb": Port(
            name="vram_budget_mb", type=float, required=True,
            doc="Target ceiling for VRAM actually reserved by the allocator. Once "
                "a connected trainer reports more than this, it'll start offloading "
                "whatever it registered as safe to offload.",
        ),
        "vram_reserve_mb": Port(
            name="vram_reserve_mb", type=float, required=False, default=512.0,
            doc="Safety margin below the budget above, left unused on purpose -- "
                "covers allocator overhead and short-lived spikes so the actual "
                "ceiling enforced is a bit under what you asked for, not exactly it.",
        ),
        "device": Port(name="device", type=str, required=False, default="xpu"),
    }

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "control": Port(
            name="control", type=ResourceControlHandle, required=True,
            doc="Wire into a trainer node's own resource-control input.",
        ),
    }

    def build(self, **inputs) -> dict[str, ResourceControlHandle]:
        self.validate_inputs(inputs)
        budget = ResourceBudget(
            vram_budget_mb=inputs["vram_budget_mb"],
            vram_reserve_mb=inputs.get("vram_reserve_mb", self.INPUTS["vram_reserve_mb"].default),
        )
        result = {
            "control": BudgetedResourceControlHandle(
                budget, inputs.get("device", self.INPUTS["device"].default)),
        }
        self.validate_outputs(result)
        return result
