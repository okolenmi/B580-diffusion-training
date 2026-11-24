"""App and board tests -- modules, documents, binding, API, runtime wiring.

What this covers (all on fixture nodes; nothing here touches a GPU):

  B  the catalog carries port hints and a node's modules
  C  domain: board / container / app document shape (every problem, with a path), versions by graph hash
  D  DescribeGraphModules: fields vs wires, streams, the monitor-wire rule
  E  ResolveBoard: card statuses (ok / wired / missing_*), also through nested containers
  F  app_binding: only exposed fields can be set, at any nesting depth
  G  the runtime hands a declaring node a stream handle (and a wired
     monitor still wins); the supervisor clears streams / republishes memory
  H  the HTTP API end to end: app CRUD and versions, diff, boards, run and the origin it records
  I  artifacts: the pictures a run leaves for a page are served, nothing else is
  J  actions: a page's "save now" reaches a running node through the command inbox and is answered
  K  an app's runs: the list filtered by app (the history card), a run queued behind a live one (the queue card) and cancelled

Run directly: python backend/tests/test_apps.py
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.execution_recorder import ExecutionRecorder
from backend.application.app_binding import apply_values, exposed_fields
from backend.domain.graph import Edge, GraphDefinition, NodeSpec
from backend.domain.modules import GRAPH_STREAM, node_stream_id, run_stream_id
from backend.domain.app import graph_hash, normalize_app_doc, validate_app_doc
from backend.domain.board import iter_cards, normalize_cards, validate_board_doc
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
from graph.document import GraphDoc

API = "/api/v1/apps"
BOARDS = "/api/v1/boards"

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
check([m["name"] for m in w["modules"]] == ["loss", "progress", "status", "result"], "modules reach the catalog as dicts")
check([m["kind"] for m in w["modules"]][2:] == ["line", "result"], "a sentence and a result are module kinds the catalog carries")
check(snap["SumNode"]["modules"] == [], "a node with no live views has none")

# ==========================================================================
# C. document shape
# ==========================================================================
print("-- C: board, container and app documents --")

def block(i="b1", node=N1, name="steps", kind="field", x=0, y=0, w=6, h=2, scope="node"):
    ref = {"scope": scope, "kind": kind, "name": name}
    if scope == "node":
        ref["node"] = node
    return {"id": i, "ref": ref, "x": x, "y": y, "w": w, "h": h}

def section(i, *tabs, columns=12, x=0, y=0, w=12, h=4):
    """A container card; each tab is (id, [cards])."""
    return {"id": i, "x": x, "y": y, "w": w, "h": h,
            "container": {"columns": columns, "tabs": [{"id": t, "title": t.title(), "blocks": cards} for t, cards in tabs]}}

GRAPH = {"format": 3, "nodes": [{"uid": N1, "class_name": "WidgetNode", "params": {"steps": 2}}], "edges": []}

def board(blocks=None, **extra):
    return {"mode": "page", "columns": 12, "blocks": blocks if blocks is not None else [block()], **extra}

def app_doc(cards=None, graph=GRAPH, **extra):
    return {"graph": graph, "root": {"columns": 12, "tabs": [{"id": "main", "title": "Main",
            "blocks": cards if cards is not None else [block()]}]}, **extra}

check(validate_board_doc(board()) == [], "a well-formed board has no problems")
check(validate_board_doc({"blocks": []}) == [], "mode and columns default")
check(any("mode" in p for p in validate_board_doc({"mode": "template", "blocks": []})), "an unknown mode is refused")
check(validate_board_doc("x") == ["board: must be an object"], "a non-object is refused")
check(any("not allowed in a flat board" in p for p in validate_board_doc(board([section("s", ("t", []))]))),
      "a view has no containers")

note = {"id": "n1", "ref": {"scope": "note", "kind": "note", "name": "note1"}, "x": 0, "y": 0, "w": 6, "h": 2,
        "collapsed": False, "title": None, "options": {"text": "Set the learning rate first."}}
check(validate_board_doc(board([note])) == [], "a note block is a well-formed block")
check(validate_board_doc(board([{**note, "ref": {**note["ref"], "node": "n1"}}])) != [], "a note has no node")
check(validate_board_doc(board([{**note, "ref": {**note["ref"], "kind": "display"}}])) != [], "a note's scope and kind go together")
check(validate_board_doc(board([{**note, "ref": {"scope": "node", "kind": "note", "name": "x", "node": "n1"}}])) != [],
      "the note kind belongs to the note scope only")

bad = validate_board_doc(board([
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

check(any("columns" in p for p in validate_board_doc(board(columns=99))), "columns out of range")
check(any("collapsed" in p for p in validate_board_doc(board([{**block(), "collapsed": "yes"}]))), "collapsed must be boolean")
check(any("options" in p for p in validate_board_doc(board([{**block(), "options": {"k": "x" * 5000}}]))), "options size is capped")
check(any("at most 200 cards" in p for p in validate_board_doc(board([block(f"b{i}") for i in range(201)]))), "card count is capped")
check(any("style" in p for p in validate_board_doc(board([{**block(), "style": "neon"}]))), "a card style is one of the known ones")
check(validate_board_doc(board([{**block(), "style": "outlined"}])) == [], "a known card style passes")

check(validate_app_doc(app_doc()) == [], "a well-formed app has no problems")
check(any("graph" in p for p in validate_app_doc({"root": app_doc()["root"]})), "an app without a graph snapshot is refused")
check(any("root" in p for p in validate_app_doc({"graph": GRAPH})), "an app without a page is refused")
check(any("style.accent" in p for p in validate_app_doc(app_doc(style={"accent": "plaid"}))), "the accent is one of the palette")
check(any("uid" in p for p in validate_app_doc(app_doc(graph={"nodes": [{"class_name": "X"}]}))), "snapshot nodes need uids")

tabbed = section("tabs1", ("one", [block("t1")]), ("two", [block("t2", N1, "tag"), block("t3", N1, "steps", y=2)]))
check(validate_app_doc(app_doc([tabbed])) == [], "a card may hold several tabs")
check([c["id"] for c in iter_cards(app_doc([tabbed, block("top")]))] == ["t1", "t2", "t3", "top"],
      "iter_cards walks every tab in order")
check(any("duplicate id" in p for p in validate_app_doc(app_doc([tabbed, block("t2")]))), "card ids are unique across the whole document")
check(any("duplicate id" in p for p in validate_app_doc(app_doc([section("s", ("a", []), ("a", []))]))), "tab ids are unique in a container")
check(any("tabs" in p for p in validate_app_doc(app_doc([section("s")]))), "a container needs at least one tab")
check(any("tabs" in p for p in validate_app_doc(app_doc([section("s", *[(f"t{i}", []) for i in range(13)])]))), "tab count is capped")
check(any("ref" in p for p in validate_app_doc(app_doc([{**tabbed, "ref": block()["ref"]}]))), "a container card has no ref")
check(any("runs past the 6-column" in p for p in validate_app_doc(app_doc([
    section("s", ("a", [block("in", x=4, w=4)]), columns=6)]))), "inner cards obey the container's own grid")
deep = block("leaf")
for level in range(5):
    deep = section(f"d{level}", ("t", [deep]))
check(any("nest at most" in p for p in validate_app_doc(app_doc([deep]))), "containers nest only so deep")

norm = normalize_app_doc(app_doc([tabbed], extra_key=1))
check(norm["format"] == 1 and norm["extra_key"] == 1 and norm["style"] == {"accent": "blue", "density": "comfortable"}
      and norm["source_graph"] is None, "normalise stamps format, fills defaults, keeps unknown keys")
inner = norm["root"]["tabs"][0]["blocks"][0]["container"]["tabs"][0]["blocks"][0]
check(inner["collapsed"] is False and inner["options"] == {}, "normalise reaches cards inside containers")
check(normalize_cards([block()])[0]["title"] is None, "a card title defaults to none")

shuffled = {"format": 3, "edges": [], "nodes": [{"uid": N1, "class_name": "WidgetNode", "params": {"steps": 2}, "x": 99, "y": 5}]}
check(graph_hash(GRAPH) == graph_hash(shuffled), "layout and format do not change the graph hash")
check(graph_hash(GRAPH) != graph_hash({**GRAPH, "nodes": [{**GRAPH["nodes"][0], "params": {"steps": 3}}]}), "a param does")
check(graph_hash(GRAPH) != graph_hash({**GRAPH, "settings": {"memory.vram_max_mb": 4000}}), "a graph setting does")

# ==========================================================================
# D. DescribeGraphModules
# ==========================================================================
print("-- D: what a graph offers --")
from backend.application.use_cases.graph_modules import (  # noqa: E402
    DescribeGraphModules, ResolveBoard,
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
check("ui" in wfields["steps"], "a card's field carries the port's ui hints (a text area, a slider)")
check(set(wfields) == {"steps", "tag"}, f"the widget's own ports are its fields (got {sorted(wfields)})")
check(wfields["steps"]["value"] == 5 and wfields["steps"]["has_value"], "current value is reported")
check(wfields["tag"]["has_value"] is False, "an unset param says so")
sfields = {f["name"]: f for f in by_id[SU]["fields"]}
check(sfields["a"]["wired"] is True and sfields["b"]["wired"] is False,
      "a port fed by an edge is flagged wired")
check([d["name"] for d in by_id[W1]["displays"]] == ["loss", "progress", "status", "result"], "displays listed")
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
# E. ResolveBoard
# ==========================================================================
print("-- E: block statuses --")
resolver = ResolveBoard(describe=describe, runtime=ContractGraphRuntime(fixture_tree_registry(), releaser=lambda: None))

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
res = resolver.execute(board(blocks), graph=full_graph)
status = {b["id"]: b["status"] for b in res.blocks}
check(status == {"ok": "ok", "wired": "wired", "nonode": "missing_node", "noclass": "unknown_class",
                 "nofield": "missing_module", "disp": "ok", "nodisp": "missing_module",
                 "gf": "ok", "gfbad": "missing_module", "gd": "ok"},
      f"statuses (got {status})")
check(all(b["detail"] for b in res.blocks if b["status"] != "ok"), "every non-ok status says why")
check(any(i["code"] == "unknown_class" for i in res.issues), "graph validation issues ride along")
check(raises(Exception, lambda: resolver.execute({"mode": "zzz", "blocks": []}, graph=full_graph)),
      "a malformed draft is refused before resolving")
mon = resolver.execute({"mode": "monitor", "blocks": [block("d", W1, "loss", kind="display")]}, graph=full_graph)
check(mon.blocks[0]["status"] == "ok" and mon.issues == (),
      "a monitor window resolves against a graph passed in, and does not validate it")
nested = resolver.execute(app_doc([section("s", ("a", [block("in_ok", W1, "steps")]), ("b", [block("in_gone", ZZ, "steps")]))],
                                  graph=full_graph))
check({b["id"]: b["status"] for b in nested.blocks} == {"in_ok": "ok", "in_gone": "missing_node"},
      "an app resolves the cards inside its containers, against its own snapshot")

# ==========================================================================
# F. binding
# ==========================================================================
print("-- F: only exposed fields can be set --")
d_f = app_doc([block("a", W1, "steps"), section("s", ("t", [block("s", SU, "a")])),
               block("m", scope="graph", name="memory.vram_max_mb")], graph=full_graph)
nf, gf = exposed_fields(d_f)
check(nf == {(W1, "steps"), (SU, "a")} and gf == {"vram_max_mb"}, "exposed fields read from cards, containers included")
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
check(widget_frames[0] == {"step": 1, "total_steps": 3, "loss": 1.0, "status": "step 1 of 3"}, "the node's own dict is the frame")

recorder = ExecutionRecorder.__new__(ExecutionRecorder)
bus = RecordingMonitorBus()
recorder._monitor_bus = bus
recorder.republish_memory(7, {"kind": "memory", "t": 12.5, "reserved_mb": 100.0, "allocated_mb": 50.0,
                       "peak_mb": 120.0, "budget_mb": 9000.0, "backstop": "enforced", "event": None})
memory_frame = {"t": 12.5, "reserved_mb": 100.0, "allocated_mb": 50.0, "peak_mb": 120.0, "budget_mb": 9000.0, "event": None}
check(bus.reports == [(GRAPH_STREAM, memory_frame), (run_stream_id(7, GRAPH_STREAM), memory_frame)],
      f"memory frames are republished on the graph stream and the run's own, minus noise (got {bus.reports})")
recorder.clear_streams(GraphDefinition.from_dict(full_graph))
check(bus.cleared[0] == GRAPH_STREAM and f"node:{W1}" in bus.cleared and f"node:{SU}" in bus.cleared,
      f"a starting run clears the graph stream and every node's (got {bus.cleared})")
recorder._monitor_bus = None
recorder.republish_memory(7, {"t": 1})
recorder.clear_streams(GraphDefinition())
check(True, "no bus: both are no-ops")

# ==========================================================================
# H. API
# ==========================================================================
print("-- H: HTTP API --")
root = Path(tempfile.mkdtemp(prefix="backend-apps-"))
services = build_services(project_root=root, graph_registry=registry(), real_runs=True)
app = create_app(services)

def put(name, document, description=""):
    return asgi_request(app, f"{API}/{name}", method="PUT",
                        json_body={"description": description, "doc": document})

status_, _, body = put("train basic", app_doc([block(), section("s", ("t", [block("in", N1, "tag")]))]), "starter")
check(status_ == 201 and body["name"] == "train basic" and body["version"] == 1
      and body["card_count"] == 2 and body["node_count"] == 1, f"first save is 201, version 1 (got {status_} {body})")
check(body["doc"]["format"] == 1 and body["doc"]["graph"]["format"] == 3, "the document and its graph snapshot are stamped")
check(len(body["graph_hash"]) == 64, "the snapshot's hash is reported")

status_, _, body = put("train basic", app_doc([block(), block("b2", N1, "tag")]))
check(status_ == 200 and body["card_count"] == 2 and body["version"] == 1, "a layout-only replace is 200 and keeps the version")
changed = {**GRAPH, "nodes": [{**GRAPH["nodes"][0], "params": {"steps": 9}}]}
status_, _, body = put("train basic", app_doc(graph=changed))
check(status_ == 200 and body["version"] == 2, "a changed snapshot makes the next version")
status_, _, body = put("train basic", app_doc(graph=changed, style={"accent": "teal"}))
check(body["version"] == 2, "a style change alone does not")

status_, _, body = put("broken", app_doc([block("a", x=11, w=5), block("a")]))
check(status_ == 422 and body["error"]["code"] == "app_invalid" and len(body["error"]["details"]) >= 2,
      f"every problem is in details (got {status_} {body})")
status_, _, body = put("   ", app_doc())
check(status_ == 422 and body["error"]["code"] == "invalid_query", "an empty name is refused")
status_, _, body = put("badmem", app_doc(graph={**GRAPH, "settings": {"memory.vram_max_mb": -1}}))
check(status_ == 422, "the graph snapshot's memory settings are checked by value, like the library")

put("second", app_doc())
status_, _, body = asgi_request(app, API)
check(status_ == 200 and body["count"] == 2 and {t["name"] for t in body["apps"]} == {"train basic", "second"}, "list returns both")
status_, _, body = asgi_request(app, f"{API}/train%20basic")
check(status_ == 200 and len(body["doc"]["root"]["tabs"][0]["blocks"]) == 1, "get returns the stored document")
status_, _, body = asgi_request(app, f"{API}/nope")
check(status_ == 404 and body["error"]["code"] == "app_not_found", "unknown name is 404")

before = asgi_request(app, f"{API}/train%20basic")[2]
owned = {"format": 3, "nodes": [{"uid": SC, "class_name": "ScaleNode", "params": {}, "layout": {"x": 40, "y": 60, "w": 300}}],
         "edges": [], "views": [{"name": "kept"}]}
status_, _, body = asgi_request(app, f"{API}/train%20basic/graph", method="PUT", json_body={"graph": owned})
saved_doc = body["app"]["doc"] if status_ == 200 else {}
check(status_ == 200 and body["orphaned_cards"] == ["b1"] and body["app"]["version"] == before["version"] + 1,
      f"replacing the app's graph names the cards that lose their target and makes the next version (got {status_} {body.get('orphaned_cards')})")
check(saved_doc.get("root") == before["doc"]["root"] and body["app"]["description"] == before["description"],
      "the page and the description stay as they were")
check(saved_doc.get("graph", {}).get("nodes", [{}])[0].get("layout") == {"x": 40, "y": 60, "w": 300} and saved_doc["graph"].get("views") == [{"name": "kept"}],
      "the app's graph keeps the editor's layout and views")
moved = {**owned, "nodes": [{**owned["nodes"][0], "layout": {"x": 1, "y": 2}}]}
status_, _, body = asgi_request(app, f"{API}/train%20basic/graph", method="PUT", json_body={"graph": moved})
check(status_ == 200 and body["app"]["version"] == before["version"] + 1, "moving a node alone does not make a new version")
status_, _, body = asgi_request(app, f"{API}/train%20basic/graph", method="PUT", json_body={"graph": {"nodes": "x"}})
check(status_ == 422 and body["error"]["code"] == "invalid_query", "a graph that is not a document is refused")
status_, _, body = asgi_request(app, f"{API}/nope/graph", method="PUT", json_body={"graph": owned})
check(status_ == 404, "replacing the graph of an unknown app is 404")
put("train basic", before["doc"])  # later sections start from the document as it was

status_, _, body = asgi_request(app, f"{BOARDS}/modules", method="POST", json_body={"graph": {
    "format": 3, "nodes": [{"uid": W1, "class_name": "WidgetNode", "params": {"steps": 4}}], "edges": []}})
check(status_ == 200 and body["nodes"][0]["displays"][0]["stream"] == f"node:{W1}"
      and body["graph_fields"][0]["name"].startswith("memory."), "modules endpoint")

status_, _, body = asgi_request(app, f"{BOARDS}/resolve", method="POST", json_body={
    "doc": app_doc([block("ok", W1, "steps"), block("gone", ZZ, "steps")], graph=full_graph)})
check(status_ == 200 and [b["status"] for b in body["blocks"]] == ["ok", "missing_node"], f"resolve endpoint (got {status_} {body})")
status_, _, body = asgi_request(app, f"{BOARDS}/resolve", method="POST", json_body={"doc": {"graph": GRAPH}})
check(status_ == 422 and body["error"]["code"] == "board_invalid", "resolve refuses a malformed draft")
status_, _, body = asgi_request(app, f"{BOARDS}/resolve", method="POST", json_body={
    "doc": board([block("d", W1, "loss", kind="display")], mode="monitor"), "graph": full_graph})
check(status_ == 200 and body["blocks"][0]["status"] == "ok", "a view resolves against the graph sent beside it")
status_, _, body = asgi_request(app, "/api/v1/templates")
check(status_ == 404, "the old templates API is gone")

# run: a sum graph, exposing 'b' only
SUM_GRAPH = {"format": 3, "nodes": [{"uid": S, "class_name": "SumNode", "params": {"a": 1.0, "b": 2.0}}], "edges": []}
put("sum", app_doc([section("s", ("t", [block("fb", S, "b")]))], graph=SUM_GRAPH))
status_, _, body = asgi_request(app, f"{API}/sum/run", method="POST", json_body={"values": {"nodes": {S: {"a": 99.0}}}})
check(status_ == 422 and body["error"]["code"] == "app_invalid" and "does not expose" in str(body["error"]["details"]),
      f"a hidden field cannot be set (got {status_} {body})")
status_, _, body = asgi_request(app, f"{API}/sum/run", method="POST", json_body={"values": {"nodes": {S: {"b": 40.0}}}})
check(status_ == 201 and body["status"] in ("queued", "running", "finished"), f"run starts (got {status_} {body})")
check(body["origin"] == {"app": "sum", "version": 1, "values": {"nodes": {S: {"b": 40.0}}}},
      f"the run records which app and version it came from (got {body.get('origin')})")
eid = body["execution_id"]
final = {}
def done():
    s, _, b = asgi_request(app, f"/api/v1/graphs/executions/{eid}")
    final.update(b)
    return b.get("status") in ("finished", "error", "stopped")
check(wait_until(done, timeout=30), "the run reaches a terminal state")
check(final.get("status") == "finished", f"run finished (got {final.get('status')} {final.get('error')})")
check(final.get("origin", {}).get("app") == "sum", "the origin is stored with the run")
res_sum = final["results"][0]["outputs"]["sum"] if final.get("results") else None
check(res_sum == 41.0, f"the exposed value was used (a=1 saved, b=40 typed => 41; got {res_sum})")

status_, _, body = put("blank", app_doc([], graph={"format": 3, "nodes": [], "edges": []}))
check(status_ == 201 and body["node_count"] == 0 and body["card_count"] == 0, "an app can start empty: no nodes, no cards")
status_, _, body = asgi_request(app, f"{API}/blank/run", method="POST", json_body={})
check(status_ in (409, 422) and "error" in body, f"running an empty app is refused with the usual error, not a crash (got {status_} {body})")
status_, _, body = asgi_request(app, f"{API}/nope/run", method="POST", json_body={})
check(status_ == 404, "running an unknown app is 404")
status_, _, body = asgi_request(app, f"{API}/second", method="DELETE")
check(status_ == 200 and body == {"deleted": True}, "delete")
status_, _, body = asgi_request(app, f"{API}/second", method="DELETE")
check(status_ == 404, "delete again is 404")

# a container built without app services answers honestly
import dataclasses  # noqa: E402
bare = dataclasses.replace(services, apps=None)
status_, _, body = asgi_request(create_app(bare), API)
check(status_ == 503 and body["error"]["code"] == "apps_unavailable", "no app services => 503")

# ==========================================================================
# I. artifacts
# ==========================================================================
print("-- I: artifacts --")
(root / "artifacts" / "n_aaaaaa").mkdir(parents=True)
(root / "artifacts" / "n_aaaaaa" / "p1.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
(root / "artifacts" / "n_aaaaaa" / "note.txt").write_text("not a picture")
(root / "secret.png").write_bytes(b"x")
status_, headers, body = asgi_request(app, "/api/v1/artifacts/n_aaaaaa/p1.png")
check(status_ == 200 and headers.get("content-type") == "image/png", f"a picture a node left is served (got {status_} {headers.get('content-type')})")
for bad in ("n_aaaaaa/note.txt", "n_aaaaaa/missing.png", "../secret.png", "n_aaaaaa/../../secret.png"):
    status_, _, body = asgi_request(app, f"/api/v1/artifacts/{bad}")
    check(status_ == 404, f"{bad!r} is not served (got {status_})")

# ==========================================================================
# J. actions
# ==========================================================================
print("-- J: actions --")
from backend.infrastructure.command_inbox import CommandInbox, CommandOutbox  # noqa: E402
from backend.tests import fixture_nodes  # noqa: E402

with tempfile.TemporaryDirectory() as raw:
    inbox_dir = Path(raw) / "execution_1.commands"
    got: list = []
    box = CommandInbox(inbox_dir, got.append)
    outbox = CommandOutbox(inbox_dir)
    first = outbox.submit(S, "save_now")
    second = outbox.submit(S, "other")
    (inbox_dir / "junk.json").write_text("{}")
    (inbox_dir / "99999999999999999999-deadbeef.json").write_text("not json")
    check(box.poll() == 2 and [c["id"] for c in got] == [first, second] and got[0] == {"id": first, "node": S, "action": "save_now"},
          f"commands arrive whole, in the order they were sent, and nothing else is read (got {got})")
    check(sorted(p.name for p in inbox_dir.iterdir()) == ["junk.json"], "a taken command is removed; files that are not commands are left alone")

LISTEN, WAIT = uid("listen"), uid("wait")
RUN_GRAPH = {"format": 3, "nodes": [
    {"uid": LISTEN, "class_name": "CheckpointListenerNode", "params": {}},
    {"uid": WAIT, "class_name": "ActionWaitNode", "params": {}}], "edges": []}
status_, _, body = asgi_request(app, "/api/v1/graphs/run", method="POST", json_body={"graph": RUN_GRAPH})
check(status_ == 201, f"a run with an action starts (got {status_} {body})")
xid = body["execution_id"]
EXEC = f"/api/v1/graphs/executions/{xid}"


def state() -> str:
    return asgi_request(app, EXEC)[2].get("status")


check(wait_until(lambda: state() == "running" and (state_node := asgi_request(app, EXEC)[2]) is not None
                 and any(r["node_id"] == LISTEN for r in state_node["results"]), timeout=30), "the run is live and the listener is built")
status_, _, mods = asgi_request(app, f"{BOARDS}/modules", method="POST", json_body={"graph": RUN_GRAPH})
action = next(n for n in mods["nodes"] if n["id"] == WAIT)["actions"][0]
check(action["name"] == "save_now" and action["available"] is True, f"the board is offered the action, available (got {action})")
status_, _, body = asgi_request(app, f"{EXEC}/actions", method="POST", json_body={"node": WAIT, "action": "nope"})
check(status_ == 404 and body["error"]["code"] == "action_not_found", f"an action the node does not declare is 404 (got {status_} {body})")
status_, _, body = asgi_request(app, f"{EXEC}/actions", method="POST", json_body={"node": "n_zzzzzz", "action": "save_now"})
check(status_ == 404, "a node that is not in the run is 404")
status_, _, body = asgi_request(app, f"{EXEC}/actions", method="POST", json_body={"node": WAIT, "action": "save_now"})
check(status_ == 202 and body["status"] == "queued" and body["command_id"], f"the action is queued (got {status_} {body})")
check(wait_until(lambda: state() == "finished", timeout=30), f"the node handled it and the run finished (got {state()})")
check(fixture_nodes.HEARD == [(7, "manual")], f"the listener heard the node's checkpoint (got {fixture_nodes.HEARD})")
status_, _, frames = asgi_request(app, f"{EXEC}/monitor/node:{WAIT}")
answers = [f["status"] for f in frames.get("frames", []) if f.get("kind") == "action"]
check(answers[:2] == ["queued", "delivered"], f"the run answered on the node's stream (got {answers})")
status_, _, body = asgi_request(app, f"{EXEC}/actions", method="POST", json_body={"node": WAIT, "action": "save_now"})
check(status_ == 409 and body["error"]["code"] == "graph_execution_not_active", f"a finished run takes no actions (got {status_} {body})")

LONE = {"format": 3, "nodes": [{"uid": WAIT, "class_name": "ActionWaitNode", "params": {}}], "edges": []}
status_, _, body = asgi_request(app, "/api/v1/graphs/run", method="POST", json_body={"graph": LONE})
lone = body["execution_id"]
check(wait_until(lambda: asgi_request(app, f"/api/v1/graphs/executions/{lone}")[2].get("status") == "running", timeout=30), "a lone node runs")
status_, _, body = asgi_request(app, f"/api/v1/graphs/executions/{lone}/actions", method="POST", json_body={"node": WAIT, "action": "save_now"})
check(status_ == 409 and body["error"]["code"] == "action_unavailable" and "listens" in body["error"]["message"],
      f"with nothing to act on it the action is refused and says what to add (got {status_} {body})")
asgi_request(app, f"/api/v1/graphs/executions/{lone}/stop", method="POST")
wait_until(lambda: asgi_request(app, f"/api/v1/graphs/executions/{lone}")[2].get("status") in ("stopped", "finished", "error"), timeout=30)

# ==========================================================================
# K. an app's runs
# ==========================================================================
print("-- K: an app's runs --")
status_, _, body = asgi_request(app, "/api/v1/graphs/executions?app=sum")
mine = body.get("executions", [])
check(status_ == 200 and [r["execution_id"] for r in mine] == [eid], f"the list can be limited to the runs one app started (got {status_} {mine})")
check(asgi_request(app, "/api/v1/graphs/executions?app=nope")[2].get("executions") == [], "an app that never ran has an empty history")
check(any(r["origin"] is None for r in asgi_request(app, "/api/v1/graphs/executions")[2]["executions"]), "without the filter runs from the editor are listed too")
status_, _, body = asgi_request(app, f"{API}/sum/run", method="POST", json_body={"values": {}, "mode": "sideways"})
check(status_ == 422, f"an unknown run mode is refused (got {status_})")

def start_live() -> int:
    """A run that is running now: the previous one gives its memory back a moment after it ends, so a start can be planned first."""
    for _ in range(40):
        _, _, started = asgi_request(app, "/api/v1/graphs/run", method="POST", json_body={"graph": LONE})
        if started["status"] != "planned":
            return started["execution_id"]
        asgi_request(app, f"/api/v1/graphs/executions/{started['execution_id']}/stop", method="POST")
        time.sleep(0.25)
    return started["execution_id"]


busy = start_live()
check(wait_until(lambda: asgi_request(app, f"/api/v1/graphs/executions/{busy}")[2].get("status") == "running", timeout=30), "a run is live")
status_, _, body = asgi_request(app, f"{API}/sum/run", method="POST", json_body={"values": {"nodes": {S: {"b": 5.0}}}, "mode": "queue"})
waiting = body.get("execution_id")
check(status_ == 201 and body["status"] == "planned" and body["origin"]["app"] == "sum",
      f"a run queued behind a live one waits, and remembers where it came from (got {status_} {body.get('status')})")
rows = asgi_request(app, "/api/v1/graphs/executions?app=sum")[2]["executions"]
check(rows[0]["execution_id"] == waiting and rows[0]["origin"]["values"] == {"nodes": {S: {"b": 5.0}}},
      "the history lists it newest first with the values it was asked for (what Run again puts back)")
status_, _, body = asgi_request(app, f"/api/v1/graphs/executions/{waiting}/stop", method="POST")
check(status_ == 200 and asgi_request(app, f"/api/v1/graphs/executions/{waiting}")[2]["status"] == "stopped", f"a waiting run is cancelled like a running one is stopped (got {status_})")
asgi_request(app, f"/api/v1/graphs/executions/{busy}/stop", method="POST")
wait_until(lambda: asgi_request(app, f"/api/v1/graphs/executions/{busy}")[2].get("status") in ("stopped", "finished", "error"), timeout=30)

finish()
