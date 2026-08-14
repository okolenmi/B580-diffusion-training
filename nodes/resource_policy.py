"""ResourceBudget/ResourcePolicy/ManualResourcePolicy.

Domain-independent by construction, the same way core.py is: every method
below uses a forward-reference string type hint only, so this module
needs no real imports from any domain package and stays a valid downward
dependency for all of them (see core.py's own docstring for the same
discipline applied to Port/Node).

ManualResourcePolicy is a pure carrier: it stores and returns whatever
domain objects it's constructed with, rather than importing domain
packages itself to compute defaults. A caller builds its own default
instance from whatever it already imports, the same pattern
diffusion_process ports already use elsewhere in this codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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


class ResourcePolicy(ABC):
    """Decides a run's checkpointing, LoRA scaling, and parameter-group
    choices as one object instead of three independent flags."""

    @abstractmethod
    def checkpointing_strategy(self) -> "ActivationCheckpointingStrategy":
        ...

    @abstractmethod
    def lora_scaling_policy(self) -> "LoRAScalingPolicy":
        ...

    @abstractmethod
    def parameter_group_policy(self) -> "ParameterGroupPolicy":
        ...


class ManualResourcePolicy(ResourcePolicy):
    """All three choices supplied explicitly at construction -- no
    inspection of budget or hardware."""

    def __init__(self, checkpointing: "ActivationCheckpointingStrategy",
                 lora_scaling_policy: "LoRAScalingPolicy",
                 parameter_group_policy: "ParameterGroupPolicy"):
        self._checkpointing = checkpointing
        self._lora_scaling_policy = lora_scaling_policy
        self._parameter_group_policy = parameter_group_policy

    def checkpointing_strategy(self) -> "ActivationCheckpointingStrategy":
        return self._checkpointing

    def lora_scaling_policy(self) -> "LoRAScalingPolicy":
        return self._lora_scaling_policy

    def parameter_group_policy(self) -> "ParameterGroupPolicy":
        return self._parameter_group_policy
