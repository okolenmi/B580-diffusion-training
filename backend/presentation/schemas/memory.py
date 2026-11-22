"""Memory overrides and the pre-run memory preview."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...domain.memory_settings import AUTO


class MemoryOverridesIn(BaseModel):
    """Per-execution overrides of the graph's memory settings.

    All-optional: anything not named keeps the graph's value. Unknown
    keys are rejected (``extra="forbid"``) with the offending key in
    the 422 -- a typo in an override must not silently become "use the
    graph's default", because the caller asked to change exactly that
    value.

    MEM-05: ``device`` is deliberately not one of the overridable keys.
    The child runs the device the *graph file* names; an override would
    live only on the row, which the child never sees -- so the key is
    refused here and the domain's merge drops it too, rather than
    being accepted into a value nothing could honour.

    MEM-03H-01: the same number checks as ``MemorySettings`` -- an
    override arrives through the same endpoint and would otherwise be
    the softer way into the ledger with a bad value.
    """

    model_config = ConfigDict(extra="forbid")

    vram_min_mb: float | None = Field(None, ge=0, allow_inf_nan=False)
    vram_max_mb: float | str | None = None
    strict: bool | None = None
    policy: str | None = None
    ram_max_mb: float | str | None = None

    @field_validator("vram_min_mb", "vram_max_mb", "ram_max_mb", mode="before")
    @classmethod
    def _no_boolean_budget(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"{value!r} is a boolean, not a memory budget")
        return value

    @field_validator("vram_max_mb", "ram_max_mb", mode="before")
    @classmethod
    def _number_or_auto(cls, value: Any) -> Any:
        # None means "keep the graph's value" and must fall through to
        # the union; everything else is a value the caller is changing.
        if value is None:
            return value
        if isinstance(value, str):
            if value != AUTO:
                raise ValueError(f"{value!r} is not a number or {AUTO!r}")
            return value
        if isinstance(value, (int, float)):
            try:
                number = float(value)
            except OverflowError:
                # An int past the float range (10**400): pydantic turns
                # ValueError into the 422, but would let OverflowError
                # escape as a 500.
                raise ValueError(
                    f"{value!r} is too large to be a device size"
                ) from None
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"{value!r} is not a finite number above zero")
            return value
        return value

    @model_validator(mode="after")
    def _min_within_max(self) -> MemoryOverridesIn:
        maximum = self.vram_max_mb
        if (
            self.vram_min_mb is not None
            and isinstance(maximum, float)
            and self.vram_min_mb > maximum
        ):
            raise ValueError(
                f"vram_min_mb ({self.vram_min_mb}) exceeds vram_max_mb ({maximum})"
            )
        return self

    def as_overrides(self) -> dict:
        """Only the keys that were named -- absent means "keep the
        graph's value", and a present key carries its override value."""
        return {
            key: value
            for key, value in self.model_dump().items()
            if value is not None
        }


class MemoryPreviewOut(BaseModel):
    """What a run of this graph would need, and whether it fits now.

    Always 200 -- a preview never refuses anything, it reports. The
    nulls are the point: ``peak_mb`` null is "never measured", not
    "needs nothing", and ``device_demand_mb`` null is "no ledger yet".
    """

    fingerprint_key: str | None
    peak_mb: float | None
    demand_mb: float | None
    demand_source: str
    exploratory: bool
    device_demand_mb: float | None
    verdict: str
    reason: str | None
    capacity_mb: float | None
    free_mb: float | None
    held_mb: float | None
    foreign_reserve_mb: float | None
    holders: dict[str, dict]


__all__ = ["MemoryOverridesIn", "MemoryPreviewOut"]
