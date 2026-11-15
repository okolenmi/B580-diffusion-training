"""What the data pipeline needs from a family's training objective (pure declarations)."""

from __future__ import annotations

from typing import Any, Protocol


class Objective(Protocol):
    timestep_modes: tuple[str, ...]

    @property
    def t_range(self) -> tuple[Any, Any]:
        """Smallest and largest timestep a sample may take (integers for DDPM, floats for flow)."""

    def sample_t(self, n: int, mode: str, low: Any, high: Any, generator: Any = None) -> Any:
        """``n`` timesteps."""

    def noisy(self, x0: Any, noise: Any, t: Any) -> tuple[Any, Any, Any]:
        """(network input, regression target, sigma) for latents ``x0`` and timesteps ``t``."""
