"""FrozenWeightStore: how a frozen weight is actually kept in memory,
decoupled from the model code that uses it. docs/training_pipeline_design.md
section 3.3.

The frozen base is this project's own documented single biggest static
VRAM allocation (docs/vram_and_lora_phase_split.md's "Considered, not
implemented" section). BF16WeightStore below is today's actual, only
behavior -- the frozen base kept exactly as loaded, no change to any
existing forward path. It exists so TrainableModel.footprint_bytes()
(nodes/model/handle.py) has something real to report through, and so a
future NF4WeightStore (QLoRA-style 4-bit quantized storage, Dettmers et
al. arXiv:2305.14314 -- see this design doc section 3.3 for the full
citation and the diffusion-specific quality caveat that needs checking
before adopting it) can exist later without restructuring TrainableModel
again. NF4WeightStore is deliberately NOT built here -- real, substantial
systems work (a fused dequant-matmul kernel or an explicit dequantize
scratch-buffer story) and its own validation pass, scoped as its own
dedicated follow-up per the design doc's calibration for this section.
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
        dequantized tensor each call (a future NF4WeightStore) or just
        return the stored tensor directly (BF16WeightStore, below)."""


class BF16WeightStore(FrozenWeightStore):
    """Today's actual, only behavior -- the frozen base kept exactly as
    loaded. No change to any existing forward path. Not actually
    restricted to bf16 specifically -- footprint_bytes()/materialize()
    are correct for whatever dtype the wrapped tensor already is (this
    project's models load as bf16 today, hence the name, matching what
    this store was actually built to represent)."""

    def __init__(self, weight):
        self._weight = weight

    def footprint_bytes(self) -> int:
        return self._weight.numel() * self._weight.element_size()

    def materialize(self):
        return self._weight
