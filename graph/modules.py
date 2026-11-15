"""Display modules: what a node can show while it runs.

A node lists them in ``MODULES``; the editor and the monitor boards draw them from the
catalog. A module names the report keys it reads; it never draws anything itself.
"""

from __future__ import annotations

from dataclasses import dataclass

KINDS = ("chart", "scalar", "bars", "text", "progress")
CADENCES = ("step", "run")
SCALES = ("linear", "symlog")
X_KEYS = ("step", "t")


@dataclass(frozen=True)
class Series:
    key: str
    label: str = ""
    color: str | None = None
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Series.key must not be empty")

    def as_dict(self) -> dict:
        return {"key": self.key, "label": self.label or self.key, "color": self.color, "unit": self.unit}


@dataclass(frozen=True)
class DisplayModule:
    name: str
    kind: str
    title: str
    series: tuple[Series, ...] = ()
    pattern: str | None = None
    exclude: tuple[str, ...] = ()
    cadence: str = "step"
    x: str = "step"
    scale: str = "linear"
    value_key: str | None = None
    total_key: str | None = None
    reference_key: str | None = None
    doc: str = ""
    size: tuple[int, int] = (6, 4)

    def __post_init__(self) -> None:
        if not self.name or ":" in self.name or "/" in self.name:
            raise ValueError(f"DisplayModule name {self.name!r} must be non-empty without ':' or '/'")
        if self.kind not in KINDS:
            raise ValueError(f"DisplayModule {self.name!r}: kind must be one of {KINDS}")
        if self.cadence not in CADENCES:
            raise ValueError(f"DisplayModule {self.name!r}: cadence must be one of {CADENCES}")
        if self.scale not in SCALES:
            raise ValueError(f"DisplayModule {self.name!r}: scale must be one of {SCALES}")
        if self.x not in X_KEYS:
            raise ValueError(f"DisplayModule {self.name!r}: x must be one of {X_KEYS}")
        if self.kind in ("chart", "bars", "scalar", "text") and not (self.series or self.pattern):
            raise ValueError(f"DisplayModule {self.name!r}: a {self.kind} module needs a Series or a pattern")
        if self.kind == "progress" and not (self.value_key and self.total_key):
            raise ValueError(f"DisplayModule {self.name!r}: progress needs value_key and total_key")
        if len(self.size) != 2 or any(not isinstance(v, int) or v < 1 for v in self.size):
            raise ValueError(f"DisplayModule {self.name!r}: size must be two positive ints")

    def as_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "title": self.title,
            "series": [s.as_dict() for s in self.series], "pattern": self.pattern,
            "exclude": list(self.exclude), "cadence": self.cadence, "x": self.x,
            "scale": self.scale, "value_key": self.value_key, "total_key": self.total_key,
            "reference_key": self.reference_key, "doc": self.doc, "size": list(self.size),
        }
