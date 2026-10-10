"""Graph-scope modules and stream identity -- what a page can bind to that is
*not* a node's own.

A template or monitor page is a board of **blocks**; each block shows one
**module**. Nodes contribute two families of module (see ``nodes/ui.py``):
editable *field* modules (one per input port) and read-only *display*
modules (declared live views). A graph has modules of its own too, because
some things a person wants on a page belong to the run, not to any node:
the memory budget they are about to spend, how far the run has got, how
much of the card it is holding.

Those graph-scope modules are defined here, as data, in the same shape as a
node's (so the frontend has exactly one renderer for both). They are not
nodes and never become one: a node would have to sit in the graph, take part
in validation and ordering, and be wired -- none of which is true of "the
memory budget".

Pure Python, no ``nodes`` import (the backend's layering). The stream id for
a node instance is therefore defined *here*, once, and the runtime that
injects the report handle and the page that subscribes to it both read it
from this module -- there is no second copy of the format to drift.
"""

from __future__ import annotations

from typing import Any

#: Where a node instance's display modules report (one stream per instance,
#: shared by all of that node's modules; see ``nodes/ui.py``).
NODE_STREAM_PREFIX = "node:"

#: Where graph-scope display modules report (memory frames, the run clock).
#: One execution is active at a time, so one stream; the supervisor clears
#: it when a run starts.
GRAPH_STREAM = "graph:run"

#: ``ref.scope`` values and ``ref.kind`` values -- the whole addressing
#: scheme of a block's target.
SCOPES: tuple[str, ...] = ("node", "graph")
KINDS: tuple[str, ...] = ("field", "display")


def node_stream_id(node_id: str) -> str:
    """The monitor-bus stream of one node instance's display modules."""
    return f"{NODE_STREAM_PREFIX}{node_id}"


# -- graph-scope field modules: the graph's own settings ---------------------
#
# Shape mirrors ``PortInfo`` on purpose (``name``, ``type``, ``default``,
# ``choices``, ``doc``, ``label``, ``unit``, ``minimum``...): the form
# renderer draws a node's port and a graph setting through the same code.
# ``allow_auto`` is the one addition -- "a number, or the word auto" -- which
# no node port needs but a memory budget does.

GRAPH_FIELDS: tuple[dict[str, Any], ...] = (
    {
        "name": "memory.vram_max_mb",
        "label": "VRAM budget",
        "type": "float",
        "allow_auto": True,
        "default": "auto",
        "unit": "MB",
        "minimum": 256,
        "step": 256,
        "group": "Memory",
        "doc": "The most device memory this run may use. 'auto' means "
               "whatever is free when the run starts (leaving head-room); a "
               "number is exact and the run is refused if the card cannot "
               "give it.",
    },
    {
        "name": "memory.vram_min_mb",
        "label": "VRAM floor",
        "type": "float",
        "default": 0,
        "unit": "MB",
        "minimum": 0,
        "step": 256,
        "group": "Memory",
        "advanced": True,
        "doc": "Refuse to start when less than this is available. 0 = no floor.",
    },
    {
        "name": "memory.strict",
        "label": "Strict budget",
        "type": "bool",
        "default": True,
        "group": "Memory",
        "advanced": True,
        "doc": "Going over the budget is a bug in a node, not something to train through.",
    },
    {
        "name": "memory.ram_max_mb",
        "label": "RAM budget",
        "type": "float",
        "allow_auto": True,
        "default": "auto",
        "unit": "MB",
        "minimum": 256,
        "group": "Memory",
        "advanced": True,
        "doc": "Host-memory ceiling. Recorded with the run; not enforced yet.",
    },
)

#: field name -> descriptor, for lookups.
GRAPH_FIELD_BY_NAME: dict[str, dict[str, Any]] = {f["name"]: f for f in GRAPH_FIELDS}


# -- graph-scope display modules ---------------------------------------------

GRAPH_DISPLAYS: tuple[dict[str, Any], ...] = (
    {
        "name": "run",
        "kind": "status",
        "title": "Run status",
        "source": "execution",
        "doc": "State of the current run, how long it has been going, and "
               "how many of its nodes have finished.",
        "size": [6, 3],
    },
    {
        "name": "memory",
        "kind": "chart",
        "title": "Device memory",
        "stream": GRAPH_STREAM,
        "series": [
            {"key": "reserved_mb", "label": "reserved", "color": "#6c8cff", "unit": "MB"},
            {"key": "allocated_mb", "label": "allocated", "color": "#4caf50", "unit": "MB"},
            {"key": "peak_mb", "label": "peak", "color": "#ff9800", "unit": "MB"},
        ],
        "pattern": None,
        "exclude": [],
        "cadence": "step",
        "x": "t",
        "scale": "linear",
        "value_key": None,
        "total_key": None,
        "reference_key": "budget_mb",
        "doc": "What this run's process is holding, once a second, against "
               "its budget (dashed).",
        "size": [12, 5],
    },
    {
        "name": "memory_now",
        "kind": "scalar",
        "title": "VRAM in use",
        "stream": GRAPH_STREAM,
        "series": [{"key": "reserved_mb", "label": "reserved", "color": None, "unit": "MB"}],
        "pattern": None,
        "exclude": [],
        "cadence": "step",
        "x": "t",
        "scale": "linear",
        "value_key": None,
        "total_key": None,
        "reference_key": "budget_mb",
        "doc": "The latest reserved-memory reading.",
        "size": [3, 2],
    },
)

GRAPH_DISPLAY_BY_NAME: dict[str, dict[str, Any]] = {d["name"]: d for d in GRAPH_DISPLAYS}


def wired_ports(edges, node_id: str) -> frozenset[str]:
    """Input ports of ``node_id`` that an edge feeds (their value comes
    from upstream, so a form must not offer to edit them)."""
    return frozenset(e.to_port for e in edges if e.to_node == node_id)
