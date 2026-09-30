"""Graph runtime tests -- every validation issue code + execution (M4).

The validator is the run endpoint's authority, so each issue code in
``docs/design/backend/05-graph-runtime.md`` section 4 gets a graph that
triggers it, plus the executor's paths: wiring, defaults, cancellation,
node failure, non-JSON output, callback isolation, memory release.

Run directly: python backend/tests/test_graph_runtime.py
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.domain.graph import GraphDefinition, GraphEdgeSpec, GraphNodeSpec
from backend.infrastructure.graph.runtime import ReflectedGraphRuntime
from backend.tests.support import check, finish, fixture_graph_registry

releases = {"n": 0}


def release() -> None:
    releases["n"] += 1


runtime = ReflectedGraphRuntime(fixture_graph_registry(), memory_releaser=release)


def node(node_id: str, class_name: str, **params) -> GraphNodeSpec:
    return GraphNodeSpec(id=node_id, class_name=class_name, params=params)


def edge(from_node, from_port, to_node, to_port) -> GraphEdgeSpec:
    return GraphEdgeSpec(
        from_node=from_node, from_port=from_port,
        to_node=to_node, to_port=to_port,
    )


def make(*nodes, edges=()) -> GraphDefinition:
    return GraphDefinition(nodes=tuple(nodes), edges=tuple(edges))


def codes(issues) -> list[str]:
    return [issue.code for issue in issues]


def issue_codes(graph) -> list[str]:
    return codes(runtime.validate(graph))


# ==========================================================================
# Validation -- one graph per issue code
# ==========================================================================

valid = make(
    node("v", "ScaleNode", value=3.0, factor=5.0),
    node("s", "SumNode", b=1.0),
    edges=(edge("v", "scaled", "s", "a"),),
)
check(runtime.validate(valid) == (), "a correct graph has zero issues")

check(
    issue_codes(make(node("", "SumNode", a=1.0, b=1.0)))
    == ["invalid_node_id"],
    "invalid_node_id: empty node id",
)

dup = issue_codes(
    make(node("x", "SumNode", a=1.0, b=1.0), node("x", "SumNode", a=1.0, b=1.0))
)
check("duplicate_node_id" in dup, "duplicate_node_id: reused id")

check(
    issue_codes(make(node("n", "NoSuchNode"))) == ["unknown_class"],
    "unknown_class: unregistered class name",
)

picky = issue_codes(make(node("p", "PickyNode", x=1.0)))
check(
    picky == ["shape_resolution_failed"],
    "shape_resolution_failed: resolve_inputs raised",
)

missing = issue_codes(make(node("s", "SumNode")))
check(
    missing == ["missing_required_input", "missing_required_input"],
    "missing_required_input: one per unprovided required port",
)
check(
    {i.param for i in runtime.validate(make(node("s", "SumNode")))} == {"a", "b"},
    "missing_required_input localizes the param",
)

choice = issue_codes(make(node("v", "ScaleNode", value=1.0, mode="wrap")))
check(choice == ["invalid_choice"], "invalid_choice: value outside Port.choices")

# Wire-safe type matrix: int feeds float; bool never feeds float/int;
# list feeds tuple-shaped ports; Path accepts strings, not ints.
check(
    issue_codes(make(node("v", "ScaleNode", value=2, factor=3))) == [],
    "int satisfies a float port (JSON has one number type)",
)
bool_arg = runtime.validate(make(node("v", "ScaleNode", value=True, factor=1.0)))
check(
    codes(bool_arg) == ["type_mismatch"],
    "bool does not satisfy a float port",
)
check(
    issue_codes(make(node("l", "LabelNode", text=5))) == ["type_mismatch"],
    "int does not satisfy a str port",
)
check(
    issue_codes(make(node("p", "PathNode", path="a/b.pt"))) == [],
    "str satisfies a Path port (editors send strings)",
)
check(
    issue_codes(make(node("p", "PathNode", path=7))) == ["type_mismatch"],
    "int does not satisfy a Path port",
)
check(
    issue_codes(make(node("v", "ScaleNode", value=1.0, factor=None))) == [],
    "None passes (it means use-the-default for an optional port)",
)
typed = issue_codes(make(node("e", "EmitHandleNode")))
check(
    typed == [],
    "class-typed ports are presence-checked, never judged",
)

check(
    issue_codes(make(node("s", "SumNode", a=1.0, b=1.0, c=1.0)))
    == ["unknown_param"],
    "unknown_param: key with no matching input",
)

edges_matrix = issue_codes(
    make(
        node("s", "SumNode", a=1.0, b=1.0),
        edges=(edge("s", "sum", "ghost", "x"),),
    )
)
check(edges_matrix == ["edge_unknown_node"], "edge_unknown_node: missing endpoint")

check(
    issue_codes(
        make(
            node("l", "LabelNode", text="x"),
            node("s", "SumNode", a=1.0, b=1.0),
            edges=(edge("l", "nope", "s", "a"),),
        )
    )
    == ["unknown_output_port"],
    "unknown_output_port",
)
check(
    issue_codes(
        make(
            node("v", "ScaleNode", value=1.0),
            node("s", "SumNode", a=1.0, b=1.0),
            edges=(edge("v", "scaled", "s", "nope"),),
        )
    )
    == ["unknown_input_port"],
    "unknown_input_port",
)

check(
    issue_codes(
        make(
            node("l", "LabelNode", text="x"),
            node("s", "SumNode", b=1.0),
            edges=(edge("l", "label", "s", "a"),),
        )
    )
    == ["incompatible_types"],
    "incompatible_types: str output into float input",
)
check(
    issue_codes(
        make(
            node("h2", "EmitSubHandleNode"),
            node("h", "TakeHandleNode"),
            edges=(edge("h2", "handle", "h", "handle"),),
        )
    )
    == [],
    "issubclass: a Handle-subclass output satisfies a Handle input",
)

cyclic = issue_codes(
    make(
        node("s", "SumNode", a=1.0, b=1.0),
        node("v", "ScaleNode", value=1.0),
        edges=(
            edge("s", "sum", "v", "value"),
            edge("v", "scaled", "s", "a"),
        ),
    )
)
check(cyclic == ["cycle"], "cycle: mutually dependent nodes")

# Ordering contract: nodes in submission order, then edges by index.
ordered = runtime.validate(
    make(
        node("", "NoSuchNode"),
        node("p", "PickyNode"),
        node("s", "SumNode", b=1.0),
        edges=(edge("s", "ghost", "p", "x"),),
    )
)
check(
    codes(ordered)
    == [
        "invalid_node_id",
        "unknown_class",          # node 1 (after its own id issue)
        "shape_resolution_failed",  # node 2
        "missing_required_input",   # node 3: only 'a' (b is a param)
        "unknown_output_port",      # edge 0 (inputs side skipped: shape failed)
    ],
    f"deterministic issue order (got {codes(ordered)})",
)
check(
    all(i.severity == "error" for i in ordered),
    "every emitted issue is an error (warnings never block)",
)

# ==========================================================================
# Execution
# ==========================================================================

outcome = runtime.execute(valid, cancel_event=threading.Event())
check(
    outcome.error is None and len(outcome.results) == 2,
    "valid graph runs clean with one result per node",
)
by_id = {r.node_id: r for r in outcome.results}
check(by_id["v"].ok and by_id["v"].outputs["scaled"] == 15.0,
      "ScaleNode produced 3*5")
check(by_id["s"].ok and by_id["s"].outputs["sum"] == 16.0,
      "edge wired v.scaled into SumNode.a (15+1)")
check(
    all(r.duration_ms >= 0.0 for r in outcome.results),
    "per-node timings recorded",
)

# Execution order follows dependencies (ScaleNode first, SumNode second).
check(
    [r.node_id for r in outcome.results] == ["v", "s"],
    "topological order: dependency runs first",
)

seen: list[str] = []
outcome = runtime.execute(
    valid,
    cancel_event=threading.Event(),
    on_node_done=lambda result: seen.append(result.node_id),
)
check(seen == ["v", "s"], "on_node_done fires per node in execution order")

# Defaults apply when a param is absent.
defaults = runtime.execute(
    make(node("v", "ScaleNode", value=4.0)),
    cancel_event=threading.Event(),
)
check(
    defaults.error is None and defaults.results[0].outputs["scaled"] == 8.0,
    "optional default (factor=2.0) applied by build()",
)

# Empty graph is a clean, instant finish.
empty = runtime.execute(make(), cancel_event=threading.Event())
check(empty.error is None and empty.results == (), "empty graph finishes cleanly")

# Pre-set cancel event: nothing runs, no error.
cancel_event = threading.Event()
cancel_event.set()
cancelled = runtime.execute(valid, cancel_event=cancel_event)
check(
    cancelled.error is None and cancelled.results == (),
    "cancellation before the first node reports no results, no error",
)

# A node's build() raising ends the run as an error, with the node in
# results. Ids chosen so the failing node sorts first in the topo tie-break.
boom = runtime.execute(
    make(
        node("boom", "BoomNode"),
        node("later", "SumNode", a=1.0, b=1.0),
    ),
    cancel_event=threading.Event(),
)
check(
    boom.error is not None and "boom" in boom.error,
    f"node failure surfaces in the outcome (got {boom.error!r})",
)
check(
    len(boom.results) == 1 and not boom.results[0].ok
    and boom.results[0].node_id == "boom",
    "failing node is reported with ok=False; later nodes never ran",
)

# Defense in depth: an invalid graph never executes.
invalid_run = runtime.execute(
    make(node("s", "NoSuchNode")), cancel_event=threading.Event()
)
check(
    invalid_run.error is not None
    and invalid_run.error.startswith("graph invalid:")
    and invalid_run.results == (),
    "execute() re-validates first and refuses to run",
)

# Non-JSON outputs become _type/_repr summaries (JSON-safe for reports).
obj = runtime.execute(
    make(node("o", "ObjectNode")),
    cancel_event=threading.Event(),
)
described = obj.results[0].outputs["obj"]
check(
    isinstance(described, dict) and described.get("_type") == "object",
    f"non-JSON output summarized (got {described!r})",
)
json.dumps(obj.results[0].outputs)

# A raising on_node_done callback must not abort the run.
def bad_callback(result):
    raise RuntimeError("callback exploded")


callback_run = runtime.execute(
    valid, cancel_event=threading.Event(), on_node_done=bad_callback
)
check(
    callback_run.error is None and len(callback_run.results) == 2,
    "callback exceptions are logged and swallowed",
)

# release_memory delegates to the injected releaser.
runtime.release_memory()
check(releases["n"] == 1, "release_memory calls the injected releaser")

finish()
