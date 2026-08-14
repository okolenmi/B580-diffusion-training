"""FrozenWeightStore: how a frozen weight is actually kept in memory,
decoupled from the model code that uses it.
See docs/training_pipeline_design.md section 3.3 for design rationale.

BF16WeightStore keeps the frozen base exactly as loaded, no change to any
existing forward path. It exists so TrainableModel.footprint_bytes()
(nodes/model/handle.py) has something real to report through, and so a
quantized-storage implementation (e.g. QLoRA-style 4-bit, Dettmers et al.
arXiv:2305.14314) could exist later without restructuring TrainableModel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class FrozenWeightStore(ABC):

    @abstractmethod
    def footprint_bytes(self) -> int:
        ...

    @abstractmethod
    def materialize(self):
        """bf16 view for one forward pass -- may allocate a fresh
        dequantized tensor each call (a quantized-storage implementation)
        or just return the stored tensor directly (BF16WeightStore,
        below)."""


class BF16WeightStore(FrozenWeightStore):
    """The frozen base kept exactly as loaded, no change to any existing
    forward path. Not actually restricted to bf16 specifically --
    footprint_bytes()/materialize() are correct for whatever dtype the
    wrapped tensor already is."""

    def __init__(self, weight):
        self._weight = weight

    def footprint_bytes(self) -> int:
        return self._weight.numel() * self._weight.element_size()

    def materialize(self):
        return self._weight
