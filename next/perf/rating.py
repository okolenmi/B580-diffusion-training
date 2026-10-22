"""Ideal time from the measured ceilings, and what is recoverable."""

from __future__ import annotations

from dataclasses import dataclass

from .ceilings import Ceilings
from .cost import Cost

BACKWARD_FACTOR = {"linear": 1.0, "conv": 1.0, "attention": 2.5, "norm": 2.0, "feedforward": 2.0}
# Backward of a frozen weight computes the input gradient only, so a matmul costs one forward again; the adapter's own matmuls are small.
# Attention's backward recomputes the scores and forms three gradients (about two and a half forwards). These are estimates, and say so in the report.


@dataclass(frozen=True)
class Rated:
    ideal_ms: float
    recoverable_ms: float
    efficiency: float
    bound: str  # compute or memory: which ceiling set the ideal


class Rating:
    def __init__(self, ceilings: Ceilings):
        self.ceilings = ceilings

    def ideal_ms(self, cost: Cost) -> tuple[float, str]:
        c = self.ceilings
        rate = c.int8_tops if cost.low_precision and c.int8_tops else c.gemm_tflops
        compute = cost.flops / (rate * 1e12) * 1000 if rate else 0.0
        memory = cost.bytes / (c.copy_gbps * 1e9) * 1000 if c.copy_gbps else 0.0
        return (compute, "compute") if compute >= memory else (memory, "memory")

    def rate(self, measured_ms: float, ideal_ms: float, bound: str) -> Rated:
        recoverable = max(measured_ms - ideal_ms, 0.0)
        return Rated(ideal_ms, recoverable, ideal_ms / measured_ms if measured_ms > 0 else 0.0, bound)
