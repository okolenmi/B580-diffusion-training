"""Display modules: what a node can show while it runs, and the actions a page may ask of it.

A node lists displays in ``MODULES`` and actions in ``ACTIONS``; the editor and the boards draw them from the
catalog. A module names the report keys it reads; it never draws anything itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

KINDS = ("chart", "scalar", "bars", "text", "progress", "image")
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
class ActionModule:
    """A button a page may put on a running node: "save now". The node that owns the state declares it.

    ``event`` is a key of the node's ``EVENTS_IN``; pressing the button delivers that event, built here from ``defaults`` and never
    from anything the page sends, so a page can only trigger what the node declared. ``needs`` is the event type some *other* node of
    the graph must listen for (the node's effect goes through ordinary events); when none does, the page shows the button disabled
    with ``needs_label`` as the reason.
    """

    name: str
    label: str
    event: str
    doc: str = ""
    confirm: str | None = None
    defaults: dict = field(default_factory=dict)
    needs: type | None = None
    needs_label: str = ""

    def __post_init__(self) -> None:
        if not self.name or ":" in self.name or "/" in self.name:
            raise ValueError(f"ActionModule name {self.name!r} must be non-empty without ':' or '/'")
        if not self.label or not self.event:
            raise ValueError(f"ActionModule {self.name!r} needs a label and an event")
        if self.needs is not None and not self.needs_label:
            raise ValueError(f"ActionModule {self.name!r}: `needs` also needs a `needs_label` that names what to add")

    def as_dict(self) -> dict:
        return {
            "name": self.name, "label": self.label, "event": self.event, "doc": self.doc, "confirm": self.confirm,
            "needs": self.needs.__name__ if self.needs is not None else None, "needs_label": self.needs_label,
        }


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
        if self.kind == "image" and not self.value_key:
            raise ValueError(f"DisplayModule {self.name!r}: image needs value_key, the report key that holds the list of pictures")
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
