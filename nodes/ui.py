"""UI modules: what a node (or the graph) can put on a page.

A *module* is one self-describing block a node offers to an interface. The
interface decides where it goes and how big it is; the node decides what it
is. There are two families, and a node may carry both:

* **field modules** -- one per input ``Port``, derived automatically (a node
  author writes nothing; ``Port`` hints such as ``label``/``group``/
  ``advanced`` refine how it is presented). A field is an *editable value*.
  The template builder binds a block to ``(node_id, port_name)`` and the
  page draws the same control the graph editor would, from the same port
  metadata -- so a field can never drift from the node that owns it.

* **display modules** -- declared by the node in ``Node.MODULES``. A display
  is a *read-only live view* of what the node reports while it runs: a loss
  curve, a phase-timing bar, a scalar readout. It names the report keys it
  plots; the frontend draws it by ``kind`` from a closed vocabulary. Adding
  a metric to a node is then "add a number to the report and one ``Series``
  here" -- no frontend edit and no hand-maintained key list on the other
  side (docs/design/14 problem 1).

Everything here is plain frozen data, JSON-serialisable by
``as_dict()``. Nothing in this module imports torch or anything else from
``nodes`` -- the catalog introspects classes without instantiating them, and
a descriptor that needed a device to describe itself would defeat that.

Why a closed ``kind`` vocabulary rather than free-form drawing instructions:
the backend then stays out of presentation (it says *what* the data is, not
how many pixels wide to draw it), the frontend stays in charge of how each
kind looks, and an unknown kind degrades to a labelled placeholder instead
of a blank hole. A new kind is one frontend component; a new module of an
existing kind needs no frontend change at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The closed set of display kinds the frontend knows how to draw.
#:
#: ``chart``   -- one or more series against ``x`` (step or time), with gaps
#: ``scalar``  -- the latest value of one key, big, with an optional unit
#: ``bars``    -- the latest value of many keys side by side (phase timings)
#: ``text``    -- the latest string of one key (a status line, a path)
#: ``progress``-- ``value`` out of ``total`` (step 120 of 3000)
DISPLAY_KINDS: tuple[str, ...] = ("chart", "scalar", "bars", "text", "progress")

#: How often a module's data changes. Drives the page's downsampling:
#: ``step`` modules can run to tens of thousands of points and are decimated
#: for drawing; ``run`` modules are one value per run and never are.
CADENCES: tuple[str, ...] = ("step", "run")

#: Chart axis modes. ``linear`` for magnitudes (VRAM, milliseconds);
#: ``symlog`` for a loss that spikes (see frontend/js/lib/loss_chart.js).
SCALES: tuple[str, ...] = ("linear", "symlog")

#: Report keys that always exist on a step report and are therefore the
#: legal ``x`` for a chart. ``step`` is the optimizer step; ``t`` is seconds
#: since the run started (graph-scope modules, which have no step loop).
X_KEYS: tuple[str, ...] = ("step", "t")


@dataclass(frozen=True)
class Series:
    """One plotted line / bar / value: a key of the node's report dict.

    ``key`` is an exact report key. (Keys the node only knows at run time,
    like per-phase timings, are discovered with the module's ``pattern``
    instead.) A series whose key is absent from a report is a gap, never a
    zero -- the same rule the dashboard has always had.
    """

    key: str
    label: str = ""
    color: str | None = None
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.key or not isinstance(self.key, str):
            raise TypeError(f"Series.key must be a non-empty str, got {self.key!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label or self.key,
            "color": self.color,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class DisplayModule:
    """A read-only live view a node offers (declared in ``Node.MODULES``).

    ``name`` is unique within the node and is the module's identity across
    runs and edits: a saved layout refers to ``(node_id, name)``. Renaming a
    module therefore breaks saved layouts exactly as renaming a class breaks
    saved graphs, and for the same reason it is a deliberate act.
    """

    name: str
    kind: str
    title: str
    series: tuple[Series, ...] = ()
    #: For ``bars`` (and optionally ``chart``): discover series from the
    #: report instead of listing them -- a ``fnmatch`` pattern over keys,
    #: e.g. ``*_ms`` for per-phase timings whose labels the trainer builds
    #: at run time. Discovered series get a stable palette colour.
    pattern: str | None = None
    #: Keys matched by ``pattern`` that are still excluded (a ``*_ms``
    #: pattern minus the total, which is drawn differently).
    exclude: tuple[str, ...] = ()
    cadence: str = "step"
    x: str = "step"
    scale: str = "linear"
    #: ``progress`` only: the report keys holding the value and the total.
    value_key: str | None = None
    total_key: str | None = None
    #: A key whose latest value is drawn as a dashed reference line
    #: (``vram_budget_mb`` on the VRAM chart).
    reference_key: str | None = None
    doc: str = ""
    #: Suggested footprint on a page's grid (columns, rows). A hint: the
    #: person resizes it, and a page without a grid ignores it.
    size: tuple[int, int] = (6, 4)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise TypeError(f"DisplayModule.name must be a non-empty str, got {self.name!r}")
        if ":" in self.name or "/" in self.name:
            # ':' and '/' are the separators of module ids and stream ids.
            raise ValueError(f"DisplayModule name {self.name!r} must not contain ':' or '/'")
        if self.kind not in DISPLAY_KINDS:
            raise ValueError(
                f"DisplayModule {self.name!r}: kind {self.kind!r} is not one of {DISPLAY_KINDS}"
            )
        if self.cadence not in CADENCES:
            raise ValueError(f"DisplayModule {self.name!r}: cadence {self.cadence!r} not in {CADENCES}")
        if self.scale not in SCALES:
            raise ValueError(f"DisplayModule {self.name!r}: scale {self.scale!r} not in {SCALES}")
        if self.x not in X_KEYS:
            raise ValueError(f"DisplayModule {self.name!r}: x {self.x!r} not in {X_KEYS}")
        if not all(isinstance(s, Series) for s in self.series):
            raise TypeError(f"DisplayModule {self.name!r}: series must be Series objects")
        if self.kind in ("chart", "bars", "scalar", "text") and not (self.series or self.pattern):
            raise ValueError(
                f"DisplayModule {self.name!r}: a {self.kind!r} module needs at least one "
                f"Series or a pattern -- it would have nothing to draw"
            )
        if self.kind == "progress" and not (self.value_key and self.total_key):
            raise ValueError(f"DisplayModule {self.name!r}: progress needs value_key and total_key")
        if (
            len(self.size) != 2
            or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in self.size)
        ):
            raise ValueError(f"DisplayModule {self.name!r}: size must be two positive ints")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "title": self.title,
            "series": [s.as_dict() for s in self.series],
            "pattern": self.pattern,
            "exclude": list(self.exclude),
            "cadence": self.cadence,
            "x": self.x,
            "scale": self.scale,
            "value_key": self.value_key,
            "total_key": self.total_key,
            "reference_key": self.reference_key,
            "doc": self.doc,
            "size": list(self.size),
        }


__all__ = [
    "CADENCES",
    "DISPLAY_KINDS",
    "DisplayModule",
    "SCALES",
    "Series",
    "X_KEYS",
]
