"""Template tests -- modules, documents, binding, API, runtime wiring.

What this covers (all on fixture nodes; nothing here touches a GPU):

  A  nodes/ui.py descriptors and the Port/Node declaration rules
  B  the catalog carries port hints and a node's modules
  C  domain: template document shape validation (every problem, with a path)
  D  DescribeGraphModules: fields vs wires, streams, the monitor-wire rule
  E  ResolveTemplate: block statuses (ok / wired / missing_*)
  F  template_binding: only exposed fields can be set
  G  the runtime hands a declaring node a stream handle (and a wired
     monitor still wins); the supervisor clears streams / republishes memory
  H  the HTTP API end to end: CRUD, resolve, modules, run

Run directly: python backend/tests/test_templates.py
"""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.core import Node, Port
from nodes.monitor.handle import MonitorHandle
from nodes.ui import DisplayModule, Series

from backend.application.graph_supervisor import GraphExecutionSupervisor
from backend.application.template_binding import apply_values, exposed_fields
from backend.domain.graph import GraphDefinition, GraphEdgeSpec, GraphNodeSpec
from backend.domain.modules import GRAPH_STREAM, node_stream_id
from backend.domain.template import normalize_template_doc, validate_template_doc
from backend.infrastructure.graph.catalog import DiscoveredGraphCatalog
from backend.infrastructure.graph.discovery import NodeRegistry
from backend.infrastructure.graph.runtime import ReflectedGraphRuntime
from backend.presentation.app import create_app
from backend.tests.support import (
    FIXTURE_NODES,
    RecordingMonitorBus,
    asgi_request,
    build_services,
    check,
    finish,
    wait_until,
)

API = "/api/v1/templates"


# --------------------------------------------------------------------------
# Fixture nodes of our own (FIXTURE_NODES is shared by many suites)
# --------------------------------------------------------------------------

class WidgetNode(Node):
    """Reports three steps of 'loss' through its monitor handle."""

    INPUTS = {
        "steps": Port(name="steps", type=int, required=False, default=3,
                      label="Steps", group="Run", unit="steps", minimum=1, maximum=100, step=1),
        "tag": Port(name="tag", type=str, required=False, default="x", advanced=True),
        "monitor": Port(name="monitor", type=MonitorHandle, required=False, default=None),
    }
    OUTPUTS = {"done": Port(name="done", type=int)}
    MODULES = (
        DisplayModule(
            name="loss", kind="chart", title="Loss", scale="symlog",
            series=(Series("loss", "loss", "#6c8cff"),),
        ),
        DisplayModule(name="progress", kind="progress", title="Progress",
                      value_key="step", total_key="total_steps"),
    )

    def build(self, steps=3, tag="x", monitor=None):
        if monitor is not None:
            for i in range(steps):
                monitor.report({"step": i + 1, "total_steps": steps, "loss": 1.0 / (i + 1)})
        return {"done": steps}


class FakeMonitorNode(Node):
    """Stands in for TrainingProgressMonitorNode: has a monitor_id input."""

    INPUTS = {"monitor_id": Port(name="monitor_id", type=str)}
    OUTPUTS = {"monitor": Port(name="monitor", type=MonitorHandle)}

    def build(self, monitor_id):
        from nodes.monitor.handle import LiveMonitorHandle
        return {"monitor": LiveMonitorHandle(monitor_id, self.context.monitor_bus)}


def registry() -> NodeRegistry:
    classes = dict(FIXTURE_NODES)
    classes["WidgetNode"] = WidgetNode
    classes["FakeMonitorNode"] = FakeMonitorNode
    return NodeRegistry(scan=lambda: (classes, ()))


def raises(exc_type, fn) -> bool:
    try:
        fn()
    except exc_type:
        return True
    except Exception:  # noqa: BLE001 -- wrong type is a failure, reported by the caller
        return False
    return False


# ==========================================================================
# A. nodes/ui.py + Port/Node rules
# ==========================================================================
print("-- A: descriptors and declaration rules --")

check(raises(ValueError, lambda: DisplayModule(name="a", kind="blink", title="t", series=(Series("k"),))),
      "an unknown kind is refused at declaration")
check(raises(ValueError, lambda: DisplayModule(name="a", kind="chart", title="t")),
      "a chart with nothing to draw is refused")
check(raises(ValueError, lambda: DisplayModule(name="a:b", kind="chart", title="t", series=(Series("k"),))),
      "':' in a module name is refused (it separates ids)")
check(raises(ValueError, lambda: DisplayModule(name="p", kind="progress", title="t")),
      "progress without value_key/total_key is refused")
check(raises(ValueError, lambda: DisplayModule(name="a", kind="chart", title="t",
                                               series=(Series("k"),), size=(0, 3))),
      "a zero-sized module is refused")
check(raises(ValueError, lambda: DisplayModule(name="a", kind="chart", title="t",
                                               series=(Series("k"),), scale="log")),
      "an unknown scale is refused")

loss = WidgetNode.MODULES[0].as_dict()
check(loss["series"][0] == {"key": "loss", "label": "loss", "color": "#6c8cff", "unit": ""}
      and loss["scale"] == "symlog" and loss["size"] == [6, 4],
      f"as_dict is JSON-shaped (got {loss})")

def _dup():
    class Dup(Node):
        OUTPUTS = {"o": Port(name="o", type=int)}
        MODULES = (
            DisplayModule(name="a", kind="scalar", title="t", series=(Series("k"),)),
            DisplayModule(name="a", kind="scalar", title="u", series=(Series("k"),)),
        )
        def build(self): return {"o": 1}
check(raises(TypeError, _dup), "duplicate module names on one node are refused at class creation")

def _not_tuple():
    class Bad(Node):
        OUTPUTS = {"o": Port(name="o", type=int)}
        MODULES = ["nope"]
        def build(self): return {"o": 1}
check(raises(TypeError, _not_tuple), "MODULES must be a tuple of DisplayModule")

check(raises(ValueError, lambda: Port(name="p", type=int, minimum=5, maximum=1)),
      "minimum above maximum is refused")
check(raises(ValueError, lambda: Port(name="p", type=int, step=0)), "a zero step is refused")
check(raises(TypeError, lambda: Port(name="p", type=int, minimum="1")), "a non-numeric bound is refused")
check(Port(name="p", type=int).advanced is False and Port(name="p", type=int).label == "",
      "hints default to 'not stated'")

from nodes.train.node import TrainerNode  # noqa: E402
names = [m.name for m in TrainerNode.MODULES]
check(names == ["progress", "loss", "lr", "grad_norm", "vram", "timing",
                  "loss_now", "loss_by_t", "vram_peak"],
      f"the trainer declares its live views (got {names})")
vram = next(m for m in TrainerNode.MODULES if m.name == "vram")
check(vram.reference_key == "vram_budget_mb" and "vram_reserved_mb" in [s.key for s in vram.series],
      "the VRAM view names the budget reference and the reserved series")

# ==========================================================================
# B. catalog
# ==========================================================================
print("-- B: the catalog carries hints and modules --")
catalog = DiscoveredGraphCatalog(registry())
snap = {n.class_name: n for n in catalog.snapshot().nodes}
w = snap["WidgetNode"]
steps = next(p for p in w.inputs if p.name == "steps")
check(steps.label == "Steps" and steps.group == "Run" and steps.unit == "steps"
      and (steps.minimum, steps.maximum, steps.step) == (1, 100, 1),
      "port hints reach PortInfo")
check(next(p for p in w.inputs if p.name == "tag").advanced is True, "advanced reaches PortInfo")
check([m["name"] for m in w.modules] == ["loss", "progress"], "modules reach NodeInfo as dicts")
check(snap["SumNode"].modules == (), "a node with no live views has none")
from backend.presentation.schemas import graph_node_out  # noqa: E402
out = graph_node_out(w)
check(out.modules[0]["kind"] == "chart" and out.inputs[0].label == "Steps",
      "the wire schema carries both")

# ==========================================================================
# C. document shape
# ==========================================================================
print("-- C: template document shape --")

def block(i="b1", node="n1", name="steps", kind="field", x=0, y=0, w=6, h=2, scope="node"):
    ref = {"scope": scope, "kind": kind, "name": name}
    if scope == "node":
        ref["node"] = node
    return {"id": i, "ref": ref, "x": x, "y": y, "w": w, "h": h}

GRAPH = {"nodes": [{"id": "n1", "class_name": "WidgetNode", "params": {"steps": 2}}], "edges": []}

def doc(blocks=None, **extra):
    return {"kind": "template", "columns": 12, "graph": GRAPH,
            "blocks": blocks if blocks is not None else [block()], **extra}

check(validate_template_doc(doc()) == [], "a well-formed page has no problems")
check(validate_template_doc({"kind": "monitor", "blocks": []}) == [],
      "a monitor window needs no graph snapshot")
check(any("graph" in p for p in validate_template_doc({"kind": "template", "blocks": []})),
      "a template without a graph snapshot is refused")
check(validate_template_doc("x") == ["template: must be an object"], "a non-object is refused")

bad = validate_template_doc(doc(blocks=[
    block("a", x=10, w=6),                                   # runs past the grid
    block("a"),                                              # duplicate id
    {"id": "c", "ref": {"scope": "graph", "kind": "field", "name": "memory.vram_max_mb", "node": "n1"},
     "x": 0, "y": 0, "w": 1, "h": 1},                        # graph block with a node
    {"id": "bad id!", "ref": {"scope": "moon", "kind": "x", "name": ""}, "x": -1, "y": 0, "w": 0, "h": 1},
    {"id": "e", "ref": {"scope": "node", "kind": "field", "name": "f"}, "x": 0, "y": 0, "w": 1, "h": 1},
]))
joined = "\n".join(bad)
for needle in ("runs past the 12-column grid", "duplicate id", "a graph block has no node",
               "ref.scope", "ref.kind", "ref.name", "needs a node id", ".id:", "x: must be an integer >= 0",
               "w: must be an integer >= 1"):
    check(needle in joined, f"reported: {needle}")
check(len(bad) >= 9, f"every problem is collected, not just the first (got {len(bad)})")

check(any("columns" in p for p in validate_template_doc(doc(columns=99))), "columns out of range")
check(any("collapsed" in p for p in validate_template_doc(
    doc(blocks=[{**block(), "collapsed": "yes"}]))), "collapsed must be boolean")
check(any("options" in p for p in validate_template_doc(
    doc(blocks=[{**block(), "options": {"k": "x" * 5000}}]))), "options size is capped")
check(any("blocks: at most" in p for p in validate_template_doc(
    doc(blocks=[block(f"b{i}") for i in range(201)]))), "block count is capped")

norm = normalize_template_doc(doc(extra_key=1))
check(norm["format"] == 1 and norm["extra_key"] == 1 and norm["blocks"][0]["collapsed"] is False
      and norm["blocks"][0]["options"] == {}, "normalise stamps format, fills defaults, keeps unknown keys")

# ==========================================================================
# D. DescribeGraphModules
# ==========================================================================
print("-- D: what a graph offers --")
from backend.application.use_cases.graph_modules import (  # noqa: E402
    DescribeGraphModules, ResolveTemplate, is_settable,
)
describe = DescribeGraphModules(catalog=catalog)

g = GraphDefinition(
    nodes=(
        GraphNodeSpec("w1", "WidgetNode", {"steps": 5}),
        GraphNodeSpec("sc", "ScaleNode", {"value": 2.0}),
        GraphNodeSpec("su", "SumNode", {"b": 1.0}),
        GraphNodeSpec("ghost", "NoSuchNode", {}),
    ),
    edges=(GraphEdgeSpec("sc", "scaled", "su", "a"),),
)
m = describe.execute(g)
by_id = {n["id"]: n for n in m.nodes}
wfields = {f["name"]: f for f in by_id["w1"]["fields"]}
check(set(wfields) == {"steps", "tag"}, f"handle-typed 'monitor' is not a field (got {sorted(wfields)})")
check(wfields["steps"]["value"] == 5 and wfields["steps"]["has_value"], "current value is reported")
check(wfields["tag"]["has_value"] is False, "an unset param says so")
sfields = {f["name"]: f for f in by_id["su"]["fields"]}
check(sfields["a"]["wired"] is True and sfields["b"]["wired"] is False,
      "a port fed by an edge is flagged wired")
check([d["name"] for d in by_id["w1"]["displays"]] == ["loss", "progress"], "displays listed")
check(by_id["w1"]["displays"][0]["stream"] == node_stream_id("w1") == "node:w1",
      "each display carries its stream")
check(by_id["ghost"]["known"] is False and by_id["ghost"]["fields"] == [], "an unknown class is kept, marked")
check({f["name"] for f in m.graph_fields} >= {"memory.vram_max_mb", "memory.vram_min_mb"},
      "the graph's own settings are offered")
check(next(f for f in m.graph_fields if f["name"] == "memory.vram_max_mb")["value"] == "auto",
      "graph setting current value comes from the graph")
check({d["name"] for d in m.graph_displays} >= {"run", "memory"}, "graph-scope live views are offered")

check(is_settable(next(p for p in snap["ScaleNode"].inputs if p.name == "mode")), "choices => settable")
check(not is_settable(next(p for p in snap["WidgetNode"].inputs if p.name == "monitor")),
      "a handle type is not settable")

# the old route: a wired monitor node's id is where reports go
g2 = GraphDefinition(
    nodes=(GraphNodeSpec("m", "FakeMonitorNode", {"monitor_id": "my-stable-id"}),
           GraphNodeSpec("w1", "WidgetNode", {})),
    edges=(GraphEdgeSpec("m", "monitor", "w1", "monitor"),),
)
check(describe.execute(g2).nodes[1]["stream"] == "my-stable-id",
      "a wired monitor node's id is the stream (old route keeps working)")

# ==========================================================================
# E. ResolveTemplate
# ==========================================================================
print("-- E: block statuses --")
resolver = ResolveTemplate(describe=describe, runtime=ReflectedGraphRuntime(registry(), memory_releaser=lambda: None))

full_graph = {
    "nodes": [
        {"id": "w1", "class_name": "WidgetNode", "params": {}},
        {"id": "sc", "class_name": "ScaleNode", "params": {"value": 1.0}},
        {"id": "su", "class_name": "SumNode", "params": {"b": 1.0}},
        {"id": "ghost", "class_name": "NoSuchNode", "params": {}},
    ],
    "edges": [{"from_node": "sc", "from_port": "scaled", "to_node": "su", "to_port": "a"}],
}
blocks = [
    block("ok", "w1", "steps"),
    block("wired", "su", "a"),
    block("nonode", "zz", "steps"),
    block("noclass", "ghost", "x"),
    block("nofield", "w1", "nope"),
    block("disp", "w1", "loss", kind="display"),
    block("nodisp", "w1", "zzz", kind="display"),
    block("gf", scope="graph", name="memory.vram_max_mb"),
    block("gfbad", scope="graph", name="memory.nope"),
    block("gd", scope="graph", name="memory", kind="display"),
]
res = resolver.execute({"kind": "template", "graph": full_graph, "blocks": blocks})
status = {b["id"]: b["status"] for b in res.blocks}
check(status == {"ok": "ok", "wired": "wired", "nonode": "missing_node", "noclass": "unknown_class",
                 "nofield": "missing_module", "disp": "ok", "nodisp": "missing_module",
                 "gf": "ok", "gfbad": "missing_module", "gd": "ok"},
      f"statuses (got {status})")
check(all(b["detail"] for b in res.blocks if b["status"] != "ok"), "every non-ok status says why")
check(any(i["code"] == "unknown_class" for i in res.issues), "graph validation issues ride along")
check(raises(Exception, lambda: resolver.execute({"kind": "template", "blocks": []})),
      "a malformed draft is refused before resolving")
mon = resolver.execute({"kind": "monitor", "blocks": [block("d", "w1", "loss", kind="display")]},
                       graph=full_graph)
check(mon.blocks[0]["status"] == "ok" and mon.issues == (),
      "a monitor window resolves against a graph passed in, and does not validate it")

# ==========================================================================
# F. binding
# ==========================================================================
print("-- F: only exposed fields can be set --")
d_f = {"kind": "template", "graph": full_graph,
       "blocks": [block("a", "w1", "steps"), block("s", "su", "a"),
                  block("m", scope="graph", name="memory.vram_max_mb")]}
nf, gf = exposed_fields(d_f)
check(nf == {("w1", "steps"), ("su", "a")} and gf == {"vram_max_mb"}, "exposed fields read from blocks")
base = GraphDefinition.from_dict(full_graph)

new, probs = apply_values(base, d_f, {"nodes": {"w1": {"steps": 7}}, "memory": {"vram_max_mb": 8000}})
check(probs == [] and new.node("w1").params["steps"] == 7 and new.memory.vram_max_mb == 8000,
      "an exposed field is applied")
check(base.node("w1").params == {} and base.memory.vram_max_mb == "auto", "the input graph is untouched")
_, probs = apply_values(base, d_f, {"nodes": {"w1": {"tag": "hax"}}})
check(len(probs) == 1 and "does not expose" in probs[0], f"a hidden field is refused (got {probs})")
_, probs = apply_values(base, d_f, {"nodes": {"su": {"a": 1.0}}})
check(len(probs) == 1 and "fed by another node" in probs[0], "a wired field is refused")
_, probs = apply_values(base, d_f, {"nodes": {"nope": {"x": 1}}})
check("no such node" in probs[0], "an unknown node is refused")
_, probs = apply_values(base, d_f, {"memory": {"strict": False}})
check("does not expose" in probs[0], "a hidden graph setting is refused")
_, probs = apply_values(base, d_f, {"memory": {"vram_max_mb": -5}})
check(len(probs) == 1 and "vram_max_mb" in probs[0], f"a bad memory number is refused by name (got {probs})")
_, probs = apply_values(base, d_f, {"bogus": {}, "nodes": {"w1": {"tag": 1}}})
check(len(probs) == 2, "every problem is collected")
new, _ = apply_values(GraphDefinition.from_dict(GRAPH), d_f | {"graph": GRAPH},
                      {"nodes": {"n1": {}}})
cleared, _ = apply_values(new, d_f, {"nodes": {"w1": {"steps": None}}}) if False else (None, None)
withp = GraphDefinition.from_dict({"nodes": [{"id": "w1", "class_name": "WidgetNode", "params": {"steps": 9}}], "edges": []})
cleared, probs = apply_values(withp, d_f, {"nodes": {"w1": {"steps": None}}})
check(probs == [] and "steps" not in cleared.node("w1").params, "None clears the param (default applies)")

# ==========================================================================
# G. runtime + supervisor
# ==========================================================================
print("-- G: the runtime wires the stream; the supervisor clears and republishes --")
rec = RecordingMonitorBus()
rt = ReflectedGraphRuntime(registry(), memory_releaser=lambda: None, monitor_bus=rec)
out = rt.execute(GraphDefinition.from_dict({"nodes": [{"id": "w1", "class_name": "WidgetNode", "params": {"steps": 3}}], "edges": []}),
                 cancel_event=threading.Event())
check(out.error is None, f"graph ran (got {out.error})")
check([r[0] for r in rec.reports] == ["node:w1"] * 3, f"no monitor node needed: reports land on node:w1 (got {rec.reports})")
check(rec.reports[0][1] == {"step": 1, "total_steps": 3, "loss": 1.0}, "the node's own dict is the frame")

rec2 = RecordingMonitorBus()
rt2 = ReflectedGraphRuntime(registry(), memory_releaser=lambda: None, monitor_bus=rec2)
out = rt2.execute(GraphDefinition.from_dict({
    "nodes": [{"id": "m", "class_name": "FakeMonitorNode", "params": {"monitor_id": "mine"}},
              {"id": "w1", "class_name": "WidgetNode", "params": {"steps": 2}}],
    "edges": [{"from_node": "m", "from_port": "monitor", "to_node": "w1", "to_port": "monitor"}]}),
    cancel_event=threading.Event())
check({r[0] for r in rec2.reports} == {"mine"}, f"a wired monitor wins (got {[r[0] for r in rec2.reports]})")

rt3 = ReflectedGraphRuntime(registry(), memory_releaser=lambda: None)  # no bus at all
out = rt3.execute(GraphDefinition.from_dict({"nodes": [{"id": "w1", "class_name": "WidgetNode", "params": {}}], "edges": []}),
                  cancel_event=threading.Event())
check(out.error is None and out.results[0].outputs["done"] == 3, "no bus: the node still runs (monitor stays None)")

sup = GraphExecutionSupervisor.__new__(GraphExecutionSupervisor)
bus = RecordingMonitorBus()
sup._monitor_bus = bus
sup._republish_memory({"kind": "memory", "t": 12.5, "reserved_mb": 100.0, "allocated_mb": 50.0,
                       "peak_mb": 120.0, "budget_mb": 9000.0, "backstop": "enforced", "event": None})
check(bus.reports == [(GRAPH_STREAM, {"t": 12.5, "reserved_mb": 100.0, "allocated_mb": 50.0,
                                      "peak_mb": 120.0, "budget_mb": 9000.0, "event": None})],
      f"memory frames are republished on the graph stream, minus noise (got {bus.reports})")
sup._clear_streams(GraphDefinition.from_dict(full_graph))
check(bus.cleared[0] == GRAPH_STREAM and "node:w1" in bus.cleared and "node:su" in bus.cleared,
      f"a starting run clears the graph stream and every node's (got {bus.cleared})")
sup._monitor_bus = None
sup._republish_memory({"t": 1})
sup._clear_streams(GraphDefinition())
check(True, "no bus: both are no-ops")

# ==========================================================================
# H. API
# ==========================================================================
print("-- H: HTTP API --")
root = Path(tempfile.mkdtemp(prefix="backend-templates-"))
services = build_services(project_root=root, graph_registry=registry())
app = create_app(services)

def put(name, document, description=""):
    return asgi_request(app, f"{API}/{name}", method="PUT",
                        json_body={"description": description, "doc": document})

status_, _, body = put("train basic", doc(), "starter")
check(status_ == 201 and body["name"] == "train basic" and body["kind"] == "template"
      and body["block_count"] == 1 and body["node_count"] == 1, f"first save is 201 (got {status_} {body})")
check(body["doc"]["format"] == 1 and body["doc"]["graph"]["format"] == 2,
      "the document and its graph snapshot are stamped")
status_, _, body = put("train basic", doc(blocks=[block(), block("b2", "n1", "tag")]))
check(status_ == 200 and body["block_count"] == 2, "replace is 200")

status_, _, body = put("broken", {"kind": "template", "graph": GRAPH,
                                  "blocks": [block("a", x=11, w=5), block("a")]})
check(status_ == 422 and body["error"]["code"] == "template_invalid"
      and len(body["error"]["details"]) >= 2, f"every problem is in details (got {status_} {body})")
status_, _, body = put("   ", doc())
check(status_ == 422 and body["error"]["code"] == "invalid_query", "an empty name is refused")
status_, _, body = put("badmem", doc() | {"graph": {**GRAPH, "memory": {"vram_max_mb": -1}}})
check(status_ == 422, "the graph snapshot's memory block is checked by value, like the library")

put("watch", {"kind": "monitor", "blocks": [block("d", "n1", "loss", kind="display")]})
status_, _, body = asgi_request(app, API)
check(status_ == 200 and body["count"] == 2 and {t["name"] for t in body["templates"]} == {"train basic", "watch"},
      "list returns both")
status_, _, body = asgi_request(app, f"{API}?kind=monitor")
check(body["count"] == 1 and body["templates"][0]["name"] == "watch", "list filters by kind")
status_, _, body = asgi_request(app, f"{API}?kind=zzz")
check(status_ == 422, "an unknown kind filter is refused")
status_, _, body = asgi_request(app, f"{API}/train%20basic")
check(status_ == 200 and len(body["doc"]["blocks"]) == 2, "get returns the stored document")
status_, _, body = asgi_request(app, f"{API}/nope")
check(status_ == 404 and body["error"]["code"] == "template_not_found", "unknown name is 404")

status_, _, body = asgi_request(app, f"{API}/modules", method="POST", json_body={
    "nodes": [{"id": "w1", "class_name": "WidgetNode", "params": {"steps": 4}}], "edges": []})
check(status_ == 200 and body["nodes"][0]["displays"][0]["stream"] == "node:w1"
      and body["graph_fields"][0]["name"].startswith("memory."), "modules endpoint")

status_, _, body = asgi_request(app, f"{API}/resolve", method="POST", json_body={"doc": {
    "kind": "template", "graph": full_graph, "blocks": [block("ok", "w1", "steps"), block("gone", "zz", "steps")]}})
check(status_ == 200 and [b["status"] for b in body["blocks"]] == ["ok", "missing_node"],
      f"resolve endpoint (got {status_} {body})")
status_, _, body = asgi_request(app, f"{API}/resolve", method="POST", json_body={"doc": {"kind": "template", "blocks": []}})
check(status_ == 422 and body["error"]["code"] == "template_invalid", "resolve refuses a malformed draft")

# run: a sum graph, exposing 'b' only
SUM_GRAPH = {"nodes": [{"id": "s", "class_name": "SumNode", "params": {"a": 1.0, "b": 2.0}}], "edges": []}
put("sum", {"kind": "template", "graph": SUM_GRAPH,
            "blocks": [block("fb", "s", "b")]})
status_, _, body = asgi_request(app, f"{API}/sum/run", method="POST", json_body={"values": {"nodes": {"s": {"a": 99.0}}}})
check(status_ == 422 and body["error"]["code"] == "template_invalid"
      and "does not expose" in str(body["error"]["details"]), f"a hidden field cannot be set (got {status_} {body})")
status_, _, body = asgi_request(app, f"{API}/sum/run", method="POST", json_body={"values": {"nodes": {"s": {"b": 40.0}}}})
check(status_ == 201 and body["status"] in ("queued", "running", "finished"), f"run starts (got {status_} {body})")
eid = body["execution_id"]
final = {}
def done():
    s, _, b = asgi_request(app, f"/api/v1/graphs/executions/{eid}")
    final.update(b)
    return b.get("status") in ("finished", "error", "stopped")
check(wait_until(done, timeout=30), "the run reaches a terminal state")
check(final.get("status") == "finished", f"run finished (got {final.get('status')} {final.get('error')})")
res_sum = final["results"][0]["outputs"]["sum"] if final.get("results") else None
check(res_sum == 41.0, f"the exposed value was used (a=1 saved, b=40 typed => 41; got {res_sum})")

status_, _, body = asgi_request(app, f"{API}/watch/run", method="POST", json_body={})
check(status_ == 422 and "window" in body["error"]["message"], "a monitor window cannot be run")
status_, _, body = asgi_request(app, f"{API}/nope/run", method="POST", json_body={})
check(status_ == 404, "running an unknown template is 404")
status_, _, body = asgi_request(app, f"{API}/watch", method="DELETE")
check(status_ == 200 and body == {"deleted": True}, "delete")
status_, _, body = asgi_request(app, f"{API}/watch", method="DELETE")
check(status_ == 404, "delete again is 404")

# a container built without template services answers honestly
import dataclasses  # noqa: E402
bare = dataclasses.replace(services, templates=None)
status_, _, body = asgi_request(create_app(bare), API)
check(status_ == 503 and body["error"]["code"] == "templates_unavailable", "no template services => 503")

finish()
