"""Template tests -- modules, documents, binding, API, runtime wiring.

What this covers (all on fixture nodes; nothing here touches a GPU):

  A  (the declaration rules of display modules are checked in next/tests)
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

from backend.application.graph_supervisor import GraphExecutionSupervisor
from backend.application.template_binding import apply_values, exposed_fields
from backend.domain.graph import Edge, GraphDefinition, NodeSpec
from backend.domain.modules import GRAPH_STREAM, node_stream_id
from backend.domain.template import normalize_template_doc, validate_template_doc
from backend.infrastructure.graph.contract_catalog import ContractNodeCatalog
from backend.infrastructure.graph.runtime import ContractGraphRuntime
from backend.presentation.app import create_app
from backend.tests.support import (
    RecordingMonitorBus,
    asgi_request,
    build_services,
    check,
    finish,
    fixture_tree_registry,
    uid,
    wait_until,
)
from next.graph.document import GraphDoc

API = "/api/v1/templates"

# Node uids: graphs name their nodes by uid, and a short test name maps to a valid one.
W1, SC, SU, GHOST, ZZ, N1, S, M = (uid(n) for n in ("w1", "sc", "su", "ghost", "zz", "n1", "s", "m"))


def registry():
    """The new tree's registry over the shared fixture nodes (WidgetNode lives there)."""
    return fixture_tree_registry()


def raises(exc_type, fn) -> bool:
    try:
        fn()
    except exc_type:
        return True
    except Exception:  # noqa: BLE001 -- wrong type is a failure, reported by the caller
        return False
    return False


# ==========================================================================
# B. the catalog carries a node's hints and live views
# ==========================================================================
print("-- B: the catalog carries hints and modules --")
catalog = ContractNodeCatalog(registry())
snap = {n["class_name"]: n for n in catalog.catalog()["nodes"]}
w = snap["WidgetNode"]
steps = next(p for p in w["inputs"] if p["name"] == "steps")
check(steps["label"] == "Steps" and steps["group"] == "run" and steps["unit"] == "steps"
      and (steps["minimum"], steps["maximum"], steps["step"]) == (1, 100, 1),
      "port hints reach the catalog")
check(next(p for p in w["inputs"] if p["name"] == "tag")["advanced"] is True, "advanced reaches the catalog")
check([m["name"] for m in w["modules"]] == ["loss", "progress"], "modules reach the catalog as dicts")
check(snap["SumNode"]["modules"] == [], "a node with no live views has none")

# ==========================================================================
# C. document shape
# ==========================================================================
print("-- C: template document shape --")

def block(i="b1", node=N1, name="steps", kind="field", x=0, y=0, w=6, h=2, scope="node"):
    ref = {"scope": scope, "kind": kind, "name": name}
    if scope == "node":
        ref["node"] = node
    return {"id": i, "ref": ref, "x": x, "y": y, "w": w, "h": h}

GRAPH = {"format": 3, "nodes": [{"uid": N1, "class_name": "WidgetNode", "params": {"steps": 2}}], "edges": []}

def doc(blocks=None, **extra):
    return {"kind": "template", "columns": 12, "graph": GRAPH,
            "blocks": blocks if blocks is not None else [block()], **extra}

check(validate_template_doc(doc()) == [], "a well-formed page has no problems")
check(validate_template_doc({"kind": "monitor", "blocks": []}) == [],
      "a monitor window needs no graph snapshot")
check(any("graph" in p for p in validate_template_doc({"kind": "template", "blocks": []})),
      "a template without a graph snapshot is refused")
check(validate_template_doc("x") == ["template: must be an object"], "a non-object is refused")

note = {"id": "n1", "ref": {"scope": "note", "kind": "note", "name": "note1"}, "x": 0, "y": 0, "w": 6, "h": 2,
        "collapsed": False, "title": None, "options": {"text": "Set the learning rate first."}}
check(validate_template_doc(doc(blocks=[note])) == [], "a note block is a well-formed block")
check(validate_template_doc(doc(blocks=[{**note, "ref": {**note["ref"], "node": "n1"}}])) != [], "a note has no node")
check(validate_template_doc(doc(blocks=[{**note, "ref": {**note["ref"], "kind": "display"}}])) != [], "a note's scope and kind go together")
check(validate_template_doc(doc(blocks=[{**note, "ref": {"scope": "node", "kind": "note", "name": "x", "node": "n1"}}])) != [],
      "the note kind belongs to the note scope only")

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
    DescribeGraphModules, ResolveTemplate,
)
describe = DescribeGraphModules(catalog=catalog)

g = GraphDefinition(GraphDoc(
    nodes=(
        NodeSpec(W1, "WidgetNode", {"steps": 5}),
        NodeSpec(SC, "ScaleNode", {"value": 2.0}),
        NodeSpec(SU, "SumNode", {"b": 1.0}),
        NodeSpec(GHOST, "NoSuchNode", {}),
    ),
    edges=(Edge(SC, "scaled", SU, "a"),),
))
m = describe.execute(g)
by_id = {n["id"]: n for n in m.nodes}
wfields = {f["name"]: f for f in by_id[W1]["fields"]}
check(set(wfields) == {"steps", "tag"}, f"the widget's own ports are its fields (got {sorted(wfields)})")
check(wfields["steps"]["value"] == 5 and wfields["steps"]["has_value"], "current value is reported")
check(wfields["tag"]["has_value"] is False, "an unset param says so")
sfields = {f["name"]: f for f in by_id[SU]["fields"]}
check(sfields["a"]["wired"] is True and sfields["b"]["wired"] is False,
      "a port fed by an edge is flagged wired")
check([d["name"] for d in by_id[W1]["displays"]] == ["loss", "progress"], "displays listed")
check(by_id[W1]["displays"][0]["stream"] == node_stream_id(W1) == f"node:{W1}",
      "each display carries its stream")
check(by_id[GHOST]["known"] is False and by_id[GHOST]["fields"] == [], "an unknown class is kept, marked")
check({f["name"] for f in m.graph_fields} >= {"memory.vram_max_mb", "memory.vram_min_mb"},
      "the graph's own settings are offered")
check(next(f for f in m.graph_fields if f["name"] == "memory.vram_max_mb")["value"] == "auto",
      "graph setting current value comes from the graph")
check({d["name"] for d in m.graph_displays} >= {"run", "memory"}, "graph-scope live views are offered")

scale_fields = {f["name"] for f in by_id[SC]["fields"]}
check({"value", "factor", "mode"} <= scale_fields, "settable ports are fields (choices included)")

# ==========================================================================
# E. ResolveTemplate
# ==========================================================================
print("-- E: block statuses --")
resolver = ResolveTemplate(describe=describe, runtime=ContractGraphRuntime(fixture_tree_registry(), releaser=lambda: None))

full_graph = {
    "format": 3,
    "nodes": [
        {"uid": W1, "class_name": "WidgetNode", "params": {}},
        {"uid": SC, "class_name": "ScaleNode", "params": {"value": 1.0}},
        {"uid": SU, "class_name": "SumNode", "params": {"b": 1.0}},
        {"uid": GHOST, "class_name": "NoSuchNode", "params": {}},
    ],
    "edges": [{"from_node": SC, "from_port": "scaled", "to_node": SU, "to_port": "a"}],
}
blocks = [
    block("ok", W1, "steps"),
    block("wired", SU, "a"),
    block("nonode", ZZ, "steps"),
    block("noclass", GHOST, "x"),
    block("nofield", W1, "nope"),
    block("disp", W1, "loss", kind="display"),
    block("nodisp", W1, "zzz", kind="display"),
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
mon = resolver.execute({"kind": "monitor", "blocks": [block("d", W1, "loss", kind="display")]},
                       graph=full_graph)
check(mon.blocks[0]["status"] == "ok" and mon.issues == (),
      "a monitor window resolves against a graph passed in, and does not validate it")

# ==========================================================================
# F. binding
# ==========================================================================
print("-- F: only exposed fields can be set --")
d_f = {"kind": "template", "graph": full_graph,
       "blocks": [block("a", W1, "steps"), block("s", SU, "a"),
                  block("m", scope="graph", name="memory.vram_max_mb")]}
nf, gf = exposed_fields(d_f)
check(nf == {(W1, "steps"), (SU, "a")} and gf == {"vram_max_mb"}, "exposed fields read from blocks")
base = GraphDefinition.from_dict(full_graph)

new, probs = apply_values(base, d_f, {"nodes": {W1: {"steps": 7}}, "memory": {"vram_max_mb": 8000}})
check(probs == [] and new.node(W1).params["steps"] == 7 and new.memory.vram_max_mb == 8000,
      "an exposed field is applied")
check(base.node(W1).params == {} and base.memory.vram_max_mb == "auto", "the input graph is untouched")
_, probs = apply_values(base, d_f, {"nodes": {W1: {"tag": "hax"}}})
check(len(probs) == 1 and "does not expose" in probs[0], f"a hidden field is refused (got {probs})")
_, probs = apply_values(base, d_f, {"nodes": {SU: {"a": 1.0}}})
check(len(probs) == 1 and "fed by another node" in probs[0], "a wired field is refused")
_, probs = apply_values(base, d_f, {"nodes": {uid("nope"): {"x": 1}}})
check("no such node" in probs[0], "an unknown node is refused")
_, probs = apply_values(base, d_f, {"memory": {"strict": False}})
check("does not expose" in probs[0], "a hidden graph setting is refused")
_, probs = apply_values(base, d_f, {"memory": {"vram_max_mb": -5}})
check(len(probs) == 1 and "vram_max_mb" in probs[0], f"a bad memory number is refused by name (got {probs})")
_, probs = apply_values(base, d_f, {"bogus": {}, "nodes": {W1: {"tag": 1}}})
check(len(probs) == 2, "every problem is collected")
new, _ = apply_values(GraphDefinition.from_dict(GRAPH), d_f | {"graph": GRAPH},
                      {"nodes": {N1: {}}})
withp = GraphDefinition.from_dict({"format": 3, "nodes": [{"uid": W1, "class_name": "WidgetNode", "params": {"steps": 9}}], "edges": []})
cleared, probs = apply_values(withp, d_f, {"nodes": {W1: {"steps": None}}})
check(probs == [] and "steps" not in cleared.node(W1).params, "None clears the param (default applies)")

# ==========================================================================
# G. runtime + supervisor
# ==========================================================================
print("-- G: the runtime wires the stream; the supervisor clears and republishes --")
from backend.infrastructure.graph_task_worker import _EventReporter  # noqa: E402


class _Writer:
    def __init__(self) -> None:
        self.frames: list[tuple[str, dict]] = []

    def monitor(self, stream: str, frame: dict) -> None:
        self.frames.append((stream, frame))


writer = _Writer()
_EventReporter(writer).report("node:x", {"step": 1})
check(writer.frames == [("node:x", {"step": 1})], "every stream frame of a run is written to the event file")

writer = _Writer()
out = ContractGraphRuntime(registry(), reporter=_EventReporter(writer), releaser=lambda: None).execute(
    GraphDefinition.from_dict({"format": 3, "nodes": [{"uid": W1, "class_name": "WidgetNode", "params": {"steps": 3}}]}),
    cancel_event=threading.Event())
widget_frames = [frame for stream, frame in writer.frames if stream == f"node:{W1}"]
check(out.error is None and [f["step"] for f in widget_frames] == [1, 2, 3],
      f"a node with live views reports on node:<uid> (got {out.error}, {widget_frames})")
check(widget_frames[0] == {"step": 1, "total_steps": 3, "loss": 1.0}, "the node's own dict is the frame")

sup = GraphExecutionSupervisor.__new__(GraphExecutionSupervisor)
bus = RecordingMonitorBus()
sup._monitor_bus = bus
sup._republish_memory({"kind": "memory", "t": 12.5, "reserved_mb": 100.0, "allocated_mb": 50.0,
                       "peak_mb": 120.0, "budget_mb": 9000.0, "backstop": "enforced", "event": None})
check(bus.reports == [(GRAPH_STREAM, {"t": 12.5, "reserved_mb": 100.0, "allocated_mb": 50.0,
                                      "peak_mb": 120.0, "budget_mb": 9000.0, "event": None})],
      f"memory frames are republished on the graph stream, minus noise (got {bus.reports})")
sup._clear_streams(GraphDefinition.from_dict(full_graph))
check(bus.cleared[0] == GRAPH_STREAM and f"node:{W1}" in bus.cleared and f"node:{SU}" in bus.cleared,
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
check(body["doc"]["format"] == 1 and body["doc"]["graph"]["format"] == 3,
      "the document and its graph snapshot are stamped")
status_, _, body = put("train basic", doc(blocks=[block(), block("b2", N1, "tag")]))
check(status_ == 200 and body["block_count"] == 2, "replace is 200")

status_, _, body = put("broken", {"kind": "template", "graph": GRAPH,
                                  "blocks": [block("a", x=11, w=5), block("a")]})
check(status_ == 422 and body["error"]["code"] == "template_invalid"
      and len(body["error"]["details"]) >= 2, f"every problem is in details (got {status_} {body})")
status_, _, body = put("   ", doc())
check(status_ == 422 and body["error"]["code"] == "invalid_query", "an empty name is refused")
status_, _, body = put("badmem", doc() | {"graph": {**GRAPH, "settings": {"memory.vram_max_mb": -1}}})
check(status_ == 422, "the graph snapshot's memory settings are checked by value, like the library")

put("watch", {"kind": "monitor", "blocks": [block("d", N1, "loss", kind="display")]})
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

status_, _, body = asgi_request(app, f"{API}/modules", method="POST", json_body={"graph": {
    "format": 3, "nodes": [{"uid": W1, "class_name": "WidgetNode", "params": {"steps": 4}}], "edges": []}})
check(status_ == 200 and body["nodes"][0]["displays"][0]["stream"] == f"node:{W1}"
      and body["graph_fields"][0]["name"].startswith("memory."), "modules endpoint")

status_, _, body = asgi_request(app, f"{API}/resolve", method="POST", json_body={"doc": {
    "kind": "template", "graph": full_graph, "blocks": [block("ok", W1, "steps"), block("gone", ZZ, "steps")]}})
check(status_ == 200 and [b["status"] for b in body["blocks"]] == ["ok", "missing_node"],
      f"resolve endpoint (got {status_} {body})")
status_, _, body = asgi_request(app, f"{API}/resolve", method="POST", json_body={"doc": {"kind": "template", "blocks": []}})
check(status_ == 422 and body["error"]["code"] == "template_invalid", "resolve refuses a malformed draft")

# run: a sum graph, exposing 'b' only
SUM_GRAPH = {"format": 3, "nodes": [{"uid": S, "class_name": "SumNode", "params": {"a": 1.0, "b": 2.0}}], "edges": []}
put("sum", {"kind": "template", "graph": SUM_GRAPH,
            "blocks": [block("fb", S, "b")]})
status_, _, body = asgi_request(app, f"{API}/sum/run", method="POST", json_body={"values": {"nodes": {S: {"a": 99.0}}}})
check(status_ == 422 and body["error"]["code"] == "template_invalid"
      and "does not expose" in str(body["error"]["details"]), f"a hidden field cannot be set (got {status_} {body})")
status_, _, body = asgi_request(app, f"{API}/sum/run", method="POST", json_body={"values": {"nodes": {S: {"b": 40.0}}}})
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
