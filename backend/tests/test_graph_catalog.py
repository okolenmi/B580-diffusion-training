"""Graph catalog tests -- fixture-node introspection (M4).

Ports, presets, display names, domain grouping, diagnostics: everything
the palette reads, off the fixture classes in ``support`` (deterministic,
no nodes/ walk). Discovery itself is covered by
``test_graph_discovery.py``.

Run directly: python backend/tests/test_graph_catalog.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.errors import (
    NodeClassNotFoundError,
    NodeDiagnosticsError,
)
from backend.infrastructure.graph.catalog import DiscoveredGraphCatalog
from backend.presentation.schemas import graph_node_out
from backend.tests.support import (
    FIXTURE_NODES,
    check,
    finish,
    fixture_graph_registry,
)

registry = fixture_graph_registry()
catalog = DiscoveredGraphCatalog(registry)
snapshot = catalog.snapshot()

# -- shape of the snapshot -------------------------------------------------
check(
    len(snapshot.nodes) == len(FIXTURE_NODES),
    f"every fixture node introspected ({len(snapshot.nodes)}/{len(FIXTURE_NODES)})",
)
check(not snapshot.load_errors, "fixture scan reports no load errors")
check(list(snapshot.domains) == ["tests"], "module path derives the domain")

by_class = {node.class_name: node for node in snapshot.nodes}

# -- display names ---------------------------------------------------------
check(by_class["SumNode"].display_name == "Sum", "Node suffix stripped: Sum")
check(
    by_class["PresetChoiceNode"].display_name == "Preset Choice",
    "CamelCase split for the palette label",
)
check(by_class["BadDiagnosticsNode"].display_name == "Bad Diagnostics",
      "acronym-free name splits on capitals")

# -- ports -----------------------------------------------------------------
sum_inputs = {p.name: p for p in by_class["SumNode"].inputs}
check(
    sum_inputs["a"].required and sum_inputs["a"].type == "float"
    and sum_inputs["a"].default is None and sum_inputs["a"].default_repr is None,
    "required input: no default to report",
)
check(
    by_class["SumNode"].outputs[0].name == "sum"
    and by_class["SumNode"].outputs[0].type == "float",
    "declared outputs come from OUTPUTS",
)

scale_inputs = {p.name: p for p in by_class["ScaleNode"].inputs}
factor = scale_inputs["factor"]
check(
    not factor.required and factor.default == 2.0 and factor.default_repr == "2.0",
    "optional float default: JSON value + repr",
)
mode = scale_inputs["mode"]
check(mode.choices == ("mul", "div"), "choices exposed for the dropdown")
check(mode.default == "mul" and mode.default_repr == "'mul'",
      "string default: JSON value + repr")

path_port = by_class["PathNode"].inputs[0]
check(
    path_port.path_kind == "checkpoint" and path_port.default is None
    and path_port.default_repr == "None",
    "Path default None: JSON null plus the repr",
)
check(
    by_class["ObjectNode"].outputs[0].type == "any"
    and by_class["ObjectNode"].outputs[0].type_mro == ("Any",),
    "typing.Any renders as 'any'",
)
check(
    by_class["EmitSubHandleNode"].outputs[0].type_mro[0] == "SubHandle"
    and "Handle" in by_class["EmitSubHandleNode"].outputs[0].type_mro,
    "type_mro exposes the real inheritance chain",
)

# -- presets / diagnostics flags -------------------------------------------
check(
    by_class["PresetChoiceNode"].presets is not None
    and [p.name for p in by_class["PresetChoiceNode"].presets] == ["identity"],
    "dynamic node carries its presets",
)
check(
    by_class["SumNode"].presets is None and by_class["SumNode"].node_kind == "static",
    "static node: presets null, kind static",
)
check(
    by_class["DiagnosingNode"].has_diagnostics
    and not by_class["SumNode"].has_diagnostics,
    "has_diagnostics distinguishes overridden from inherited",
)

# -- wire safety -------------------------------------------------------------
wire = graph_node_out(by_class["ScaleNode"]).model_dump()
json.dumps(wire)
check(wire["inputs"][1]["default"] == 2.0 and wire["inputs"][1]["default_repr"] == "2.0",
      "the whole NodeInfo serializes to JSON")

# -- diagnostics endpoint behavior ------------------------------------------
messages = catalog.diagnostics("DiagnosingNode", {"path": "ckpt.pt"})
check(messages == {"path": ["looked at 'ckpt.pt'"]},
      "overridden diagnostics returns its lines")

try:
    catalog.diagnostics("NoSuchNode", {})
    check(False, "unknown class raises NodeClassNotFoundError")
except NodeClassNotFoundError:
    check(True, "unknown class raises NodeClassNotFoundError")

# The port contract lets the node's own exception propagate (the use
# case wraps it into a 400); the adapter itself must not swallow it.
try:
    catalog.diagnostics("BadDiagnosticsNode", {})
    check(False, "a raising diagnostics() propagates from the adapter")
except ValueError:
    check(True, "a raising diagnostics() propagates from the adapter")

from backend.application.use_cases import NodeDiagnostics

wrapped = NodeDiagnostics(catalog=catalog)
try:
    wrapped.execute("BadDiagnosticsNode", {})
    check(False, "the use case wraps the raise into node_diagnostics_failed")
except NodeDiagnosticsError:
    check(True, "the use case wraps the raise into node_diagnostics_failed")

# -- refresh picks up a newly added class ------------------------------------
_calls = {"n": 0}


def scan_then_grow():
    """First scan: the fixtures. After a refresh: one extra class."""
    _calls["n"] += 1
    classes = dict(FIXTURE_NODES)
    if _calls["n"] > 1:
        from backend.tests.support import SumNode  # same object, new key

        classes["FreshNode"] = type("FreshNode", (SumNode,), {"__module__": SumNode.__module__})
    return classes, ()


from backend.infrastructure.graph.discovery import NodeRegistry as _Registry

refreshing = DiscoveredGraphCatalog(_Registry(scan=scan_then_grow))
check(len(refreshing.snapshot().nodes) == len(FIXTURE_NODES),
      "cached scan before refresh")
check(
    len(refreshing.snapshot().nodes) == len(FIXTURE_NODES),
    "second snapshot serves the cache (no rescan)",
)
check(
    len(refreshing.snapshot(refresh=True).nodes) == len(FIXTURE_NODES) + 1,
    "refresh=True picks up newly added node classes",
)

finish()
