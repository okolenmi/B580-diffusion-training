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

from ..components.device import DeviceContext
from ..core import Node, Port
from ..resource_budget import ResourceBudget
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
        "strict": Port(
            name="strict", type=bool, required=False, default=False,
            doc="False (default): today's behavior -- offloads whatever it can and "
                "continues even if reserved memory is still over budget afterward "
                "(real when model/optimizer, which are never offloadable, already "
                "exceed it alone). True: raise instead of continuing over budget once "
                "nothing registered offloadable is left to move -- a hard stop instead "
                "of training on with usage past the ceiling you asked for. Recommended "
                "True when the actual goal is staying clear of a GPU/driver VRAM-"
                "pressure failure (docs/known-issues/open.md), not just visibility into "
                "usage -- e.g. BudgetedLoRATrainerNode's whole reason to exist "
                "(nodes/train/budgeted.py).",
        ),
    }

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "control": Port(
            name="control", type=ResourceControlHandle, required=True,
            doc="Wire into a trainer node's own resource-control input.",
        ),
    }

    def build(self, **inputs) -> dict[str, ResourceControlHandle]:
        self.validate_inputs(inputs)
        device = inputs.get("device", self.INPUTS["device"].default)
        budget_mb = inputs["vram_budget_mb"]

        # See DeviceContext.total_memory_mb()'s own docstring for the full
        # reasoning -- real driver-queried total, not this process's own
        # allocator bookkeeping, checked directly against the number just
        # configured above rather than trusting it silently. Best-effort:
        # None (no XPU/CUDA here, or the query itself failed) means this
        # check simply can't run, not that the budget is fine -- same
        # posture DeviceContext's other methods already take everywhere
        # else in this module.
        total_mb = DeviceContext.for_device(device).total_memory_mb()
        if total_mb is not None and budget_mb > total_mb:
            print(f"[VRAMBudgetControllerNode] WARNING: vram_budget_mb={budget_mb:.0f}MB "
                  f"exceeds this device's own real total ({total_mb:.0f}MB, queried from the "
                  f"driver) -- every decision this budget drives (offloading, escalation, "
                  f"whether \"strict\" raises) will look correct against the number you gave "
                  f"it and still end in a real driver-level OOM, since nothing can actually "
                  f"be under a ceiling higher than the device's own capacity. Not "
                  f"auto-corrected here -- lower vram_budget_mb, or confirm this really is "
                  f"the right device if that total looks wrong.")

        budget = ResourceBudget(
            vram_budget_mb=budget_mb,
            vram_reserve_mb=inputs.get("vram_reserve_mb", self.INPUTS["vram_reserve_mb"].default),
            strict=inputs.get("strict", self.INPUTS["strict"].default),
        )
        result = {
            "control": BudgetedResourceControlHandle(budget, device),
        }
        self.validate_outputs(result)
        return result
