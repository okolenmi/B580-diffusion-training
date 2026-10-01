"""Graph discovery tests -- the real nodes/ walk (M4).

Auto-discovery replaces the legacy hand import list
(``server/nodegraph_registry.py``), so the parity claim is checked
against reality: every concrete Node under ``nodes/`` is found, zero
modules fail to import, and the registry caches/refreshes correctly.

Run directly: python backend/tests/test_graph_discovery.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.infrastructure.graph.catalog import DiscoveredGraphCatalog
from backend.infrastructure.graph.discovery import NodeRegistry, scan_nodes
from backend.presentation.schemas import graph_catalog_out, graph_node_out
from backend.tests.support import check, concrete_node_classes, finish
from nodes.core import Node

classes, errors = scan_nodes()

check(not errors, f"no module failed to import ({[e.module for e in errors]})")

# Derived from nodes/'s own class definitions rather than hardcoded: the
# palette is supposed to be every concrete Node subclass, so that is the
# invariant worth asserting. A literal number here only ever needed
# editing when someone legitimately added or retired a node, and caught
# nothing the structural checks below didn't catch directly.
expected = concrete_node_classes()
check(
    set(classes) == expected,
    f"every concrete Node subclass is discovered and nothing else "
    f"(discovered {len(classes)}, expected {len(expected)}; "
    f"missing {sorted(expected - set(classes))}, extra {sorted(set(classes) - expected)})",
)
check(len(classes) >= 30, f"palette is non-trivially populated ({len(classes)} classes)")
check(
    all(isinstance(c, type) and issubclass(c, Node) for c in classes.values()),
    "every discovered class is a Node subclass",
)
check(
    all(c.__module__.startswith("nodes.") for c in classes.values()),
    "every discovered class is defined under nodes.",
)
check(
    not any("smoke_tests" in c.__module__ for c in classes.values()),
    "smoke test helpers are not part of the palette",
)

# The legacy hand list's key claim: a node class added to nodes/ shows
# up without touching any registry file. Two known names from both
# ends of the project must be there.
check("ComposedCAMEOptimizerNode" in classes, "composed optimizer discovered")
check(
    any(name.endswith("MonitorNode") for name in classes),
    "monitor node discovered",
)

# Registry: cache is stable, refresh re-scans (swap is atomic).
calls = {"n": 0}


def counting_scan():
    calls["n"] += 1
    return dict(classes), errors


registry = NodeRegistry(scan=counting_scan)
first, _ = registry.load()
second, _ = registry.load()
check(first is second and calls["n"] == 1, "scan runs once, cached afterwards")
third, _ = registry.load(refresh=True)
check(calls["n"] == 2 and third is not first, "refresh re-walks nodes/")

# Catalog over the real scan: full snapshot, wire-serializable.
snapshot = DiscoveredGraphCatalog(registry).snapshot()
check(
    len(snapshot.nodes) == len(classes) and not snapshot.load_errors,
    "snapshot covers every class with no load errors",
)
payload = graph_catalog_out(snapshot).model_dump()
json.dumps(payload)  # raises if anything escaped JSON safety
check(payload["count"] == len(expected), f"catalog payload counts the same set ({payload['count']})")
check(payload["load_errors"] == [], "no load errors reported")
check(
    payload["domains"] and all(domain for domain in payload["domains"]),
    "domains derived from module paths (every group non-empty)",
)
for domain, nodes in payload["domains"].items():
    check(
        nodes == sorted(nodes, key=lambda n: (n["display_name"], n["class_name"])),
        f"domain {domain!r} sorted by display name (stable palette)",
    )

# One node's wire shape, end to end from the real class.
sample = next(n for n in snapshot.nodes if n.class_name == "ComposedCAMEOptimizerNode")
wire = graph_node_out(sample).model_dump()
json.dumps(wire)
check(wire["node_kind"] == "static" and wire["presets"] is None,
      "static node reports presets=null")
check(
    wire["inputs"] and all(
        "name" in p and "type" in p and "required" in p for p in wire["inputs"]
    ),
    "ports carry identity fields",
)

finish()
