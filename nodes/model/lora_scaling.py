"""LoRAScalingPolicy/ClassicLoRAScaling/RankStabilizedScaling/_effective_alpha.

See docs/training_pipeline_design.md section 3.2 for the design
rationale (Kalajdzievski, arXiv:2312.03732, 2023) and lora_injector.py's
module docstring for the effective-alpha derivation -- unchanged by this
move, just relocated.

Originally defined directly in lora_injector.py; moved here once
nodes/model/adapter_injection.py needed these same classes and importing
them from lora_injector.py would have created a real cycle
(lora_injector.py -> adapter_injection.py -> lora_injector.py, since
adapter_injection.py is itself imported by lora_injector.py to wire the
adapter_strategy port). Exactly the situation section 5.7's Acyclic
Domain Dependency Rule already names: "a dependency that seems to need
to go sideways is a signal the shared piece belongs in ... a new
domain-independent module -- not that the rule should bend." Still
re-exported from lora_injector.py (`from .lora_scaling import ...`) so
every existing import of these names from nodes.model.lora_injector
keeps working unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LoRAScalingPolicy(ABC):

    @abstractmethod
    def scaling(self, alpha: float, rank: int) -> float:
        ...


class ClassicLoRAScaling(LoRAScalingPolicy):
    """Today's actual behavior -- core.lora's existing alpha/rank formula,
    unchanged. Default, so nothing wired to this Node today changes."""

    def scaling(self, alpha: float, rank: int) -> float:
        return alpha / rank


class RankStabilizedScaling(LoRAScalingPolicy):
    """Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning
    with LoRA" (arXiv:2312.03732, 2023). Its actual value depends on
    training at higher rank than this project's current default (rank=64)
    to have anything to stabilize -- worth pairing with a rank increase,
    not independently useful at the current default rank by itself."""

    def scaling(self, alpha: float, rank: int) -> float:
        return alpha / (rank ** 0.5)


def _effective_alpha(alpha: float, rank: int, policy: LoRAScalingPolicy) -> float:
    """The seam itself, pulled out as its own function so it's directly
    testable without constructing a whole ComfyUNetWrapper -- see
    lora_injector.py's module docstring for the derivation.
    ClassicLoRAScaling is the identity (returns alpha unchanged);
    anything else changes what core.lora ends up computing for `scaling`
    without core.lora itself changing at all."""
    return policy.scaling(alpha, rank) * rank
