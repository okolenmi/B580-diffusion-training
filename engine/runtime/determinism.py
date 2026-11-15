"""Switches that make two runs of the same graph, data and settings produce the same numbers."""

from __future__ import annotations

import torch


class Determinism:
    """Process-wide state: deterministic kernels, one seeded global generator, and no memory-driven batch splitting.

    Seeding the global generator covers what nodes draw from it (adapter factor initialisation, dropout).
    Everything the data path draws uses its own seeded generators already.
    """

    _applied = False

    def __init__(self, enabled: bool = False, seed: int = 0):
        self.enabled, self.seed = bool(enabled), max(int(seed), 0)

    @classmethod
    def from_settings(cls, settings) -> Determinism:
        return cls(bool(settings.get("runtime.deterministic")), int(settings.get("seed.base")))

    @property
    def plans_chunks(self) -> bool:
        """Splitting a batch by predicted memory changes the order gradients add up in, and the prediction is measured."""
        return not self.enabled

    def apply(self) -> Determinism:
        if self.enabled:
            torch.use_deterministic_algorithms(True, warn_only=True)  # warn: an operation without a deterministic kernel still runs
            torch.utils.deterministic.fill_uninitialized_memory = False  # filling only costs time here; results do not depend on it
            torch.manual_seed(self.seed)
            Determinism._applied = True
        elif Determinism._applied:
            torch.use_deterministic_algorithms(False)
            Determinism._applied = False
        return self
