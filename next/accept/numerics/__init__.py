"""Numeric checks for the hardware: ``python -m next.accept numerics``."""

from __future__ import annotations

from .base import Finding, NumericCheck, NumericsRunner, counts, markdown, write
from .context import Context
from .hardware import DeviceMemoryPattern, GraphReplay, OpRepeatability, TransferRoundtrip
from .model import (AttentionModes, CheckpointingEquality, ComputeDtype, ChunkedGradients, FusedGegluEquality, ModelInt8, ModelOffload,
                    ModelRepeat)
from .state import AdapterFile, FrozenWeights, MergeLoss, OptimizerRoundtrip


def all_checks() -> list[NumericCheck]:
    """In the order they run: no model first, then the subject, and the check that converts the subject last."""
    return [TransferRoundtrip(), DeviceMemoryPattern(), OpRepeatability(), GraphReplay(), OptimizerRoundtrip(), FrozenWeights(), AdapterFile(),
            ModelRepeat(), ComputeDtype(), ChunkedGradients(), CheckpointingEquality(), AttentionModes(), FusedGegluEquality(), MergeLoss(),
            ModelOffload(), ModelInt8()]


__all__ = ["Context", "Finding", "NumericCheck", "NumericsRunner", "all_checks", "counts", "markdown", "write"]
