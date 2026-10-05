"""MemorySettings -- per-graph memory configuration.

A graph property, not a global setting and not a node. The graph is the
configurable object: each graph holds its own budget, allocates only from
the unallocated pool, and if it doesn't fit the run cannot start.

Bumped the graph format to 2; format 1 loads with defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


#: Sentinel for "use everything that is free".
AUTO = "auto"


def _budget(name: str, value, *, allow_auto: bool, zero_ok: bool) -> None:
    """Validate one memory budget number (MEM-03H-01, domain layer).

    The same conditions the API refuses, enforced here so a stored or
    hand-edited graph file cannot smuggle a bad number past the edge:
    a boolean (``True`` is an ``int`` in Python), a non-number, a
    non-finite number, a negative one -- and, for a ceiling, zero.

    ``zero_ok`` separates the two floors this pair of settings needs: a
    *minimum* of 0 means "no floor" (it is the default), a *ceiling* of
    0 says the graph may use nothing and is a mistake -- ``auto`` is how
    a caller asks for everything free.
    """
    if isinstance(value, str):
        if allow_auto and value == AUTO:
            return
        suffix = f" or {AUTO!r}" if allow_auto else ""
        raise ValueError(f"{name}: {value!r} is not a number{suffix}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: {value!r} is not a number")
    try:
        number = float(value)
    except OverflowError:
        # An int past the float range (10**400) is not a device size;
        # left uncaught it would escape as a 500 from the save edge.
        raise ValueError(
            f"{name}: {value!r} is too large to be a device size"
        ) from None
    if not math.isfinite(number):
        raise ValueError(f"{name}: {value!r} is not a finite number")
    if zero_ok:
        if number < 0:
            raise ValueError(f"{name}: {number} MB is negative")
    elif number <= 0:
        raise ValueError(f"{name}: {number} MB is not above zero")


@dataclass(frozen=True, slots=True)
class MemorySettings:
    """Per-graph memory configuration.

    `vram_min_mb` is the least the graph can run in (refuse to start below
    it). `vram_max_mb` is the ceiling it wants; ``"auto"`` means
    "everything that is free". `strict` (default True) means exceeding the
    budget is a bug in a node, not a condition to train through.
    `ram_max_mb` is carried and validated now, enforced later.
    `device` names the accelerator this graph runs on -- carried and
    validated now, and read by the child's physical check (MEM-05) to
    know which card to ask; ``for_device``'s own dispatch decides what an
    unfamiliar name means (an unknown backend answers "no such notion",
    never a fabricated number).
    """

    vram_min_mb: float = 0.0
    vram_max_mb: float | str = AUTO
    strict: bool = True
    policy: str = "demand_driven"
    ram_max_mb: float | str = AUTO
    device: str = "xpu"

    def __post_init__(self) -> None:
        """The numbers are checked on every construction, including the
        ones that arrive through ``from_dict`` (a stored or hand-edited
        graph file) or through ``_with_overrides`` (a request's merge).

        MEM-03H-01, domain layer: independent of the API's 422, so a
        bad value that never went through the endpoint still raises
        here instead of reaching the ledger.
        """
        _budget("vram_min_mb", self.vram_min_mb, allow_auto=False, zero_ok=True)
        _budget("vram_max_mb", self.vram_max_mb, allow_auto=True, zero_ok=False)
        _budget("ram_max_mb", self.ram_max_mb, allow_auto=True, zero_ok=False)
        if not isinstance(self.vram_max_mb, str) and self.vram_min_mb > self.vram_max_mb:
            raise ValueError(
                f"vram_min_mb ({self.vram_min_mb}) exceeds vram_max_mb "
                f"({self.vram_max_mb})"
            )
        # The device is a name, not a number: a bool is not one (and
        # would be one to a `str(...)` coercion), and an empty or
        # whitespace-only string dispatches to the null context, which
        # would quietly turn the physical check into no check at all.
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError(f"device: {self.device!r} is not a device name")

    def as_dict(self) -> dict:
        return {
            "vram_min_mb": self.vram_min_mb,
            "vram_max_mb": self.vram_max_mb,
            "strict": self.strict,
            "policy": self.policy,
            "ram_max_mb": self.ram_max_mb,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, raw: dict | None) -> MemorySettings:
        """Parse from a stored dict. Missing keys get defaults; present
        keys must be numbers that mean something -- ``__post_init__``
        refuses the bad ones by name rather than coercing them.
        ``device`` is absent from every graph saved before MEM-05 and
        defaults like every other tolerant decode here."""
        if raw is None:
            return cls()
        return cls(
            vram_min_mb=raw.get("vram_min_mb", 0.0),
            vram_max_mb=raw.get("vram_max_mb", AUTO),
            strict=bool(raw.get("strict", True)),
            policy=str(raw.get("policy", "demand_driven")),
            ram_max_mb=raw.get("ram_max_mb", AUTO),
            device=raw.get("device", "xpu"),
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
    #: The demand in allocator MB: stated beats observed (held = max of
    #: the two, as in `Reservation.held_mb`); observed = recorded peak +
    #: pillow. The caller that turns a demand into a device claim adds
    #: the per-process overhead (start_graph_execution), which is what
    #: makes this number the allocator's side and not the device's.
    demand_mb: float | None
    #: Whether the demand came from a stated value, an observed peak, or
    #: is unknown.
    demand_source: str  # "stated" | "observed" | "unknown"
    #: The memory fingerprint key this execution was admitted under
    #: (MEM-01). The row carries it so the watcher can file the child's
    #: reported peaks in the peak store under the *same* key admission
    #: read its past under. None when the fingerprint was unknown at
    #: admission -- the watcher then records nothing, which is the same
    #: unknown admission already decided on, never a zero peak.
    fingerprint_key: str | None

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
            "fingerprint_key": self.fingerprint_key,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> EffectiveMemory:
        """Parse a stored ``memory_json`` payload back, round-tripping
        `as_dict` exactly -- including a null demand.

        ``fingerprint_key`` was added after the first rows were written,
        so its absence is tolerated the same way every other tolerant
        decoder here tolerates it: no key, None -- unknown, not zero.
        """
        key = raw.get("fingerprint_key")
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
            fingerprint_key=None if key is None else str(key),
        )


def _with_overrides(
    settings: MemorySettings, overrides: dict | None
) -> MemorySettings:
    """The graph's settings with the execution request's overrides
    applied. Unknown override keys are ignored, like every other
    tolerant decoder here; the five known keys keep the coercions
    `MemorySettings` expects. ``device`` is deliberately not among them
    (sixth key or not): the child runs the device the *graph file*
    names, and an override would live only on the row the child never
    sees -- so a ``device`` key in overrides is dropped here exactly
    like a typo, and the API refuses it with a 422 on the way in
    (``MemoryOverridesIn`` forbids extra keys)."""
    if not overrides:
        return settings
    merged = settings.as_dict()
    merged.pop("device", None)
    for key, value in overrides.items():
        if key in merged:
            merged[key] = value
    # No coercion here on purpose: MemorySettings.__post_init__ judges
    # exactly what the request asked for (a True that float() would have
    # turned into 1.0 is a boolean, not a budget).
    return MemorySettings(
        vram_min_mb=merged["vram_min_mb"],
        vram_max_mb=merged["vram_max_mb"],
        strict=bool(merged["strict"]),
        policy=str(merged["policy"]),
        ram_max_mb=merged["ram_max_mb"],
        device=settings.device,
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
    `device` is deliberately not among them (MEM-05): it is the graph's
    own property, persisted in the graph file the child reads.

    `peak_record` is the peak store's current contents (fingerprint key ->
    peak MB). `fingerprint_key` is this graph's fingerprint key. If the
    fingerprint is unknown or not in the record, the demand is unknown;
    with no `capacity_mb` either (no probe reading), an unknown demand
    stays ``None`` -- unknown is never a zero claim. The key itself is
    carried onto the returned `EffectiveMemory` (and from there into the
    row's ``memory_json``) so the watcher has the key admission used when
    it files this run's own peak (MEM-04 #2).
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
        fingerprint_key=fingerprint_key,
    )
