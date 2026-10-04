"""MemorySettings -- per-graph memory configuration.

A graph property, not a global setting and not a node. The graph is the
configurable object: each graph holds its own budget, allocates only from
the unallocated pool, and if it doesn't fit the run cannot start.

Bumped the graph format to 2; format 1 loads with defaults.
"""

from __future__ import annotations

from dataclasses import dataclass


#: Sentinel for "use everything that is free".
AUTO = "auto"


@dataclass(frozen=True, slots=True)
class MemorySettings:
    """Per-graph memory configuration.

    `vram_min_mb` is the least the graph can run in (refuse to start below
    it). `vram_max_mb` is the ceiling it wants; ``"auto"`` means
    "everything that is free". `strict` (default True) means exceeding the
    budget is a bug in a node, not a condition to train through.
    `ram_max_mb` is carried and validated now, enforced later.
    """

    vram_min_mb: float = 0.0
    vram_max_mb: float | str = AUTO
    strict: bool = True
    policy: str = "demand_driven"
    ram_max_mb: float | str = AUTO

    def as_dict(self) -> dict:
        return {
            "vram_min_mb": self.vram_min_mb,
            "vram_max_mb": self.vram_max_mb,
            "strict": self.strict,
            "policy": self.policy,
            "ram_max_mb": self.ram_max_mb,
        }

    @classmethod
    def from_dict(cls, raw: dict | None) -> MemorySettings:
        """Parse from a stored dict. Missing keys get defaults."""
        if raw is None:
            return cls()
        return cls(
            vram_min_mb=float(raw.get("vram_min_mb", 0.0)),
            vram_max_mb=raw.get("vram_max_mb", AUTO),
            strict=bool(raw.get("strict", True)),
            policy=str(raw.get("policy", "demand_driven")),
            ram_max_mb=raw.get("ram_max_mb", AUTO),
        )


@dataclass(frozen=True, slots=True)
class EffectiveMemory:
    """The effective memory settings for one execution.

    Computed by `effective_memory()` -- the one place defaults are
    resolved. Stored on the execution row so a restart reproduces the
    same held total from the rows.
    """

    vram_min_mb: float
    vram_max_mb: float | str
    strict: bool
    policy: str
    ram_max_mb: float | str
    #: The demand in device MB: stated beats observed (held = max of the
    #: two, as in `Reservation.held_mb`); observed = recorded peak +
    #: pillow.
    demand_mb: float | None
    #: Whether the demand came from a stated value, an observed peak, or
    #: is unknown.
    demand_source: str  # "stated" | "observed" | "unknown"

    def as_dict(self) -> dict:
        """Storage shape -- the execution row's ``memory_json`` column."""
        return {
            "vram_min_mb": self.vram_min_mb,
            "vram_max_mb": self.vram_max_mb,
            "strict": self.strict,
            "policy": self.policy,
            "ram_max_mb": self.ram_max_mb,
            "demand_mb": self.demand_mb,
            "demand_source": self.demand_source,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> EffectiveMemory:
        """Parse a stored ``memory_json`` payload back, round-tripping
        `as_dict` exactly -- including a null demand."""
        return cls(
            vram_min_mb=float(raw["vram_min_mb"]),
            vram_max_mb=raw["vram_max_mb"],
            strict=bool(raw["strict"]),
            policy=str(raw["policy"]),
            ram_max_mb=raw["ram_max_mb"],
            demand_mb=(
                None if raw.get("demand_mb") is None
                else float(raw["demand_mb"])
            ),
            demand_source=str(raw["demand_source"]),
        )


def _with_overrides(
    settings: MemorySettings, overrides: dict | None
) -> MemorySettings:
    """The graph's settings with the execution request's overrides
    applied. Unknown override keys are ignored, like every other
    tolerant decoder here; the five known keys keep the coercions
    `MemorySettings` expects."""
    if not overrides:
        return settings
    merged = settings.as_dict()
    for key, value in overrides.items():
        if key in merged:
            merged[key] = value
    return MemorySettings(
        vram_min_mb=float(merged["vram_min_mb"]),
        vram_max_mb=merged["vram_max_mb"],
        strict=bool(merged["strict"]),
        policy=str(merged["policy"]),
        ram_max_mb=merged["ram_max_mb"],
    )


def effective_memory(
    graph_settings: MemorySettings,
    request_overrides: dict | None,
    peak_record: dict[str, float] | None,
    fingerprint_key: str | None,
    capacity_mb: float | None,
    *,
    pillow_mb: float = 150.0,
) -> EffectiveMemory:
    """Compute the effective memory settings for one execution.

    The one place defaults are computed. `stated` beats `observed` (held =
    max of the two, as in `Reservation.held_mb`); `observed` = recorded
    peak + pillow. Unknown is never a zero claim.

    `request_overrides` may carry `vram_min_mb`, `vram_max_mb`, `strict`,
    `policy`, `ram_max_mb` -- the execution request may override the
    graph's settings, so one saved graph can run with different budgets.

    `peak_record` is the peak store's current contents (fingerprint key ->
    peak MB). `fingerprint_key` is this graph's fingerprint key. If the
    fingerprint is unknown or not in the record, the demand is unknown;
    with no `capacity_mb` either (no probe reading), an unknown demand
    stays ``None`` -- unknown is never a zero claim.
    """
    # -- apply request overrides ---------------------------------------------
    effective = _with_overrides(graph_settings, request_overrides)
    vram_min_mb = effective.vram_min_mb
    vram_max_mb = effective.vram_max_mb
    strict = effective.strict
    policy = effective.policy
    ram_max_mb = effective.ram_max_mb

    # -- demand: stated beats observed ---------------------------------------
    stated_mb: float | None = None
    if isinstance(vram_max_mb, (int, float)):
        stated_mb = float(vram_max_mb)

    observed_mb: float | None = None
    if fingerprint_key and peak_record and fingerprint_key in peak_record:
        observed_mb = float(peak_record[fingerprint_key]) + pillow_mb

    # Held = max of stated and observed (as in Reservation.held_mb)
    demand_mb: float | None = None
    demand_source = "unknown"
    if stated_mb is not None and observed_mb is not None:
        demand_mb = max(stated_mb, observed_mb)
        demand_source = "stated" if stated_mb >= observed_mb else "observed"
    elif stated_mb is not None:
        demand_mb = stated_mb
        demand_source = "stated"
    elif observed_mb is not None:
        demand_mb = observed_mb
        demand_source = "observed"

    # If demand is unknown, use capacity as an exploratory exclusive
    # claim. A None capacity (no probe reading) leaves the demand None
    # too: unknown, never zero -- nothing was measured, so nothing is
    # claimed.
    if demand_mb is None:
        demand_mb = capacity_mb
        demand_source = "unknown"

    return EffectiveMemory(
        vram_min_mb=vram_min_mb,
        vram_max_mb=vram_max_mb,
        strict=strict,
        policy=policy,
        ram_max_mb=ram_max_mb,
        demand_mb=demand_mb,
        demand_source=demand_source,
    )
