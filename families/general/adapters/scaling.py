"""How an adapter's output is scaled from ``alpha`` and ``rank``."""

from __future__ import annotations

POLICIES = ("classic", "rank_stabilized")


def scaling_factor(policy: str, alpha: float, rank: int) -> float:
    """classic: alpha / rank. rank_stabilized: alpha / sqrt(rank), which keeps the update size steady as rank grows."""
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if policy == "classic":
        return alpha / rank
    if policy == "rank_stabilized":
        return alpha / (rank ** 0.5)
    raise ValueError(f"unknown scaling policy {policy!r}; choose from {POLICIES}")
