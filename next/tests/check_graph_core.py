"""Self-checks for next/graph: declarations, document, validation, runtime, events, settings, resolve.

    python -m next.tests.check_graph_core
"""

from __future__ import annotations

import json
import threading

from ..graph import (Edge, EpochEnding, EventHub, EventIn, EventOut, Group, GraphDoc, GraphFormatError, GraphRunner,
                     Node, NodeRegistry, NodeSpec, Port, PromptsChanged, build_catalog, parse, resolve_node)
from ..graph.document import FORMAT, new_uid
from ..graph.events import (EventDeliveryError, EventHandlerError, EventLoopError, EventThreadError, StopRequested,
                            Wire)
from ..graph.settings import SettingsError, SettingsRegistry
from ..graph.validate import topological_order
from . import toy_nodes
from .support import check, raises, run

REGISTRY = NodeRegistry(packages=("next.tests.toy_nodes",))


def uid(n: int) -> str:
    return "n_" + "abcdefghijklmnop"[n] * 6


def doc(nodes, edges=(), event_edges=(), settings=None) -> GraphDoc:
    specs = tuple(NodeSpec(uid(i), cls, dict(params)) for i, (cls, params) in enumerate(nodes))
    to_edge = lambda e: Edge(uid(e[0]), e[1], uid(e[2]), e[3])  # noqa: E731
    return GraphDoc(specs, tuple(map(to_edge, edges)), tuple(map(to_edge, event_edges)), settings or {})


class Frames:
    def __init__(self):
        self.frames: list[tuple[str, dict]] = []

    def report(self, stream, frame):
        self.frames.append((stream, frame))


def codes(validation) -> set[str]:
    return {i.code for i in validation.issues}


def runner(**kw) -> GraphRunner:
    return GraphRunner(REGISTRY, **kw)


def declarations():
    # abstract-free classes are linted at definition, so build bad ones with type()
    def define(**attrs):
        return type("Bad", (Node,), {"build": lambda self, **i: {}, **attrs})
    raises(TypeError, lambda: define(
        INPUTS={f"f{i}": Port(f"f{i}", int, required=False) for i in range(9)}, OUTPUTS={"x": Port("x", int)}),
        "more than eight fields without groups is refused", "GROUPS")
    raises(TypeError, lambda: define(OUTPUTS={"x": Port("x", int)},
                                     INPUTS={"a": Port("a", int, group="nope")}), "unknown group", "unknown group")
    raises(TypeError, lambda: define(OUTPUTS={"x": Port("x", int)},
                                     INPUTS={"a": Port("a", int, visible_when=("zzz", 1))}), "visible_when typo")
    raises(TypeError, lambda: define(), "no outputs and not dynamic", "no OUTPUTS")
    raises(TypeError, lambda: define(OUTPUTS={"y": Port("x", int)}), "port name must match its key")
    raises(TypeError, lambda: define(OUTPUTS={"x": Port("x", int)},
                                     EVENTS_IN={"e": EventIn(EpochEnding, handler="missing")}), "missing handler")
    raises(TypeError, lambda: define(OUTPUTS={"x": Port("x", int)},
                                     EVENTS_OUT={"e": EventOut(int)}), "payload must be an EventType")
    raises(TypeError, lambda: define(OUTPUTS={"x": Port("x", int)}, DYNAMIC=toy_nodes.DynamicNode.DYNAMIC),
           "DYNAMIC needs depends_on ports and resolve()")
    raises(ValueError, lambda: Port("p", str, choices=("a",), default="b"), "default outside choices")
    raises(ValueError, lambda: Port("p", int, minimum=5, maximum=1), "min above max")
    check(toy_nodes.GroupedNode.GROUPS[1].collapsed, "a node with groups and ten fields is accepted")


def document():
    raw = {"format": FORMAT, "settings": {"device.default": "cpu"}, "extra_key": 7,
           "nodes": [{"uid": uid(0), "class_name": "ConstNode", "params": {"value": 2}, "title": "two",
                      "layout": {"x": 1}}],
           "edges": [], "event_edges": [], "views": [{"name": "v"}]}
    parsed = parse(raw)
    out = parsed.as_dict()
    check(out["extra_key"] == 7 and out["nodes"][0]["title"] == "two", "round trip keeps extras and titles")
    check(parse(out).as_dict() == out, "parse(as_dict()) is stable")
    check(parsed.for_run().nodes[0].layout == {} and parsed.for_run().views == (), "for_run strips layout and views")
    raises(GraphFormatError, lambda: parse({"format": 2, "nodes": []}), "other formats are refused", "format")
    raises(GraphFormatError, lambda: parse({"format": FORMAT, "nodes": [{"uid": 1, "class_name": "x"}]}), "bad uid type")
    raises(GraphFormatError, lambda: parse({"format": FORMAT, "edges": [{"from_node": "a"}]}), "partial edge")
    u = new_uid(); check(len(u) == 8 and u.startswith("n_"), f"new_uid shape {u}")
    check(new_uid({u}) != u, "new_uid avoids existing")
    raises(ValueError, lambda: topological_order(["a", "b"], [Edge("a", "p", "b", "q"), Edge("b", "p", "a", "q")]),
           "cycle raises", "cycle")


def validation():
    good = doc([("ConstNode", {"value": 1.0}), ("ConstNode", {}), ("AddNode", {})],
               edges=[(0, "value", 2, "terms"), (1, "value", 2, "terms")])
    v = runner().validate(good)
    check(v.ok and v.order == [uid(0), uid(1), uid(2)], f"valid graph passes ({[i.code for i in v.issues]})")

    bad_class = runner().validate(doc([("Nope", {})]))
    check("unknown_class" in codes(bad_class), "unknown class")
    check("missing_required_input" in codes(runner().validate(doc([("AddNode", {}), ("SaverNode", {})]))),
          "required wire missing")
    check("invalid_choice" in codes(runner().validate(doc([("ChoiceNode", {"mode": "z"})]))), "invalid choice")
    check("type_mismatch" in codes(runner().validate(doc([("ConstNode", {"value": "x"})]))), "type mismatch")
    check("unknown_param" in codes(runner().validate(doc([("ConstNode", {"bogus": 1})]))), "unknown param")
    check("out_of_range" in codes(runner().validate(doc([("ChoiceNode", {"n": 11})]))), "above maximum")
    check("incompatible_types" in codes(runner().validate(
        doc([("ConstNode", {}), ("SaverNode", {})], edges=[(0, "value", 1, "model")]))), "incompatible wire")
    check("unknown_output_port" in codes(runner().validate(
        doc([("ConstNode", {}), ("AddNode", {})], edges=[(0, "nope", 1, "terms")]))), "unknown output port")
    check("unknown_input_port" in codes(runner().validate(
        doc([("ConstNode", {}), ("AddNode", {})], edges=[(0, "value", 1, "nope")]))), "unknown input port")
    check("multiple_wires_to_input" in codes(runner().validate(
        doc([("ConstNode", {}), ("ConstNode", {}), ("ConstNode", {})],
            edges=[(0, "value", 2, "value"), (1, "value", 2, "value")]))), "two wires into a single port")
    check("edge_unknown_node" in codes(runner().validate(
        GraphDoc((NodeSpec(uid(0), "ConstNode"),), (Edge(uid(0), "value", "n_zzzzzz", "x"),)))), "edge to a missing node")
    check("cycle" in codes(runner().validate(
        doc([("PassNode", {}), ("PassNode", {})], edges=[(0, "model", 1, "model"), (1, "model", 0, "model")]))), "cycle")
    check("invalid_uid" in codes(runner().validate(GraphDoc((NodeSpec("bad", "ConstNode"),)))), "uid pattern")
    check("duplicate_uid" in codes(runner().validate(
        GraphDoc((NodeSpec(uid(0), "ConstNode"), NodeSpec(uid(0), "ConstNode"))))), "duplicate uid")
    check("invalid_setting" in codes(runner().validate(doc([("ConstNode", {})], settings={"memory.vram_max_mb": 1}))),
          "setting below minimum")
    sw = [i for i in runner().validate(doc([("ConstNode", {})], settings={"mystery.key": 1})).issues]
    check(sw and sw[0].severity == "warning" and sw[0].code == "unknown_setting", "unknown setting is a warning")
    check("invalid_setting" in codes(runner().validate(doc([("ConstNode", {})], settings={
        "memory.vram_min_mb": 4096, "memory.vram_max_mb": 2048}))), "floor above budget")
    # events
    check("event_incompatible" in codes(runner().validate(doc(
        [("EpochLoopNode", {}), ("ListenerNode", {}), ("TableNode", {})],
        edges=[], event_edges=[(0, "epoch_ending", 1, "prompts_changed")]))), "event payload mismatch")
    check("unknown_event_in" in codes(runner().validate(doc(
        [("EpochLoopNode", {}), ("ListenerNode", {})], event_edges=[(0, "epoch_ending", 1, "nope")]))), "unknown event in")
    check("unknown_event_out" in codes(runner().validate(doc(
        [("EpochLoopNode", {}), ("ListenerNode", {})], event_edges=[(0, "nope", 1, "prompts_changed")]))), "unknown event out")
    # dynamic shape
    dyn = lambda f: doc([("DynamicNode", {"file": f}), ("SaverNode", {})], edges=[(0, "denoiser", 1, "model")])  # noqa: E731
    check("unknown_output_port" in codes(runner().validate(dyn("small"))), "wire to a vanished port is reported")
    check("needs_choice" in codes(runner().validate(doc([("DynamicNode", {"file": "ambiguous"})]))), "needs_choice is an error")
    check("resolve_error" in codes(runner().validate(doc([("DynamicNode", {"file": "broken"})]))), "resolve error")
    check("shape_resolution_failed" in codes(runner().validate(doc([("DynamicNode", {"file": "raises"})]))),
          "a raising resolver becomes an issue")
    ok = runner().validate(doc([("DynamicNode", {"file": "full"}), ("ConstNode", {})]))
    check(ok.ok and set(ok.resolutions[uid(0)].outputs) == {"vae", "denoiser"}, "dynamic outputs resolve")


def running():
    r = runner()
    out = r.run(doc([("ConstNode", {"value": 2.0}), ("ConstNode", {"value": 3.5}), ("AddNode", {})],
                    edges=[(0, "value", 2, "terms"), (1, "value", 2, "terms")]))
    sums = [x for x in out.results if x.uid == uid(2)][0].outputs
    check(out.ok and sums["sum"] == 5.5 and sums["order"] == [2.0, 3.5], "multi port arrives in wire order")
    flipped = r.run(doc([("ConstNode", {"value": 2.0}), ("ConstNode", {"value": 3.5}), ("AddNode", {})],
                        edges=[(1, "value", 2, "terms"), (0, "value", 2, "terms")]))
    check([x for x in flipped.results if x.uid == uid(2)][0].outputs["order"] == [3.5, 2.0], "wire order, not uid order")
    check(r.run(doc([("AddNode", {})])).ok, "an empty multi port arrives as an empty list")
    check(r.run(doc([("ChoiceNode", {})])).results[0].outputs["out"] == "a", "defaults are filled for absent ports")
    bad = r.run(doc([("ConstNode", {"value": "x"})]))
    check(not bad.ok and bad.error.startswith("graph invalid"), "an invalid graph does not run")
    fail = r.run(doc([("FailingNode", {}), ("ConstNode", {})]))
    check(not fail.ok and "boom" in fail.error and fail.results[0].state == "failed", "a failing node ends the run")
    check(len(fail.results) == 1, "nothing runs after a failure")
    frames = Frames()
    runner(reporter=frames).run(doc([("ConstNode", {})]))
    states = [f["state"] for s, f in frames.frames if f.get("kind") == "node_state"]
    runs = [f["state"] for s, f in frames.frames if f.get("kind") == "run"]
    check(states == ["running", "done"] and runs == ["run_started", "run_finished"], f"state frames {states} {runs}")
    check(r.run(doc([("ConstNode", {})]), cancel_event=threading.Event()).results[0].duration_ms >= 0, "durations recorded")
    # stop handling
    plain = doc([("TrainerLikeNode", {"steps": 5, "cancel_at": 2})])
    requested = r.run(plain)
    check(requested.ok and requested.stopped, "an emitted StopRequested stops the run cleanly, without wiring")
    event_stop = doc([("TrainerLikeNode", {"steps": 5, "cancel_at": 2}), ("PassNode", {}), ("SaverNode", {}),
                      ("ConstNode", {})],
                     edges=[(0, "model", 1, "model"), (0, "model", 2, "model")])
    ev = threading.Event()
    class Stopper:
        def report(self, stream, frame):
            if frame.get("kind") == "event" and frame.get("summary", {}).get("step") == 2:
                ev.set()
    stopped = runner(reporter=Stopper()).run(event_stop, cancel_event=ev)
    by = {x.uid: x for x in stopped.results}
    check(stopped.stopped and stopped.ok, "a stop is not a failure")
    check(by[uid(1)].state == "skipped" and by[uid(3)].state == "skipped", "ordinary nodes are skipped after a stop")
    check(by[uid(2)].state == "done" and by[uid(2)].outputs["saved"] == "partial",
          "a node that accepts unfinished input still runs, and sees it is partial")


def events():
    toy_nodes.SwapNode.log.clear()
    toy_nodes.ListenerNode.log.clear()
    nodes = [("TableNode", {}), ("SwapNode", {}), ("ListenerNode", {}), ("EpochLoopNode", {"epochs": 3})]
    edges = [(0, "table", 1, "table"), (1, "table", 3, "table")]
    wires = [(3, "epoch_ending", 1, "epoch_ending"), (1, "prompts_changed", 2, "prompts_changed")]
    d = doc(nodes, edges, wires)
    v = runner().validate(d)
    check(v.ok and v.order.index(uid(2)) < v.order.index(uid(1)), f"subscriber is built before its emitter: {v.order}")
    frames = Frames()
    out = runner(reporter=frames).run(d)
    observed = [x for x in out.results if x.uid == uid(3)][0].outputs["observed"]
    check(out.ok, f"event graph runs ({out.error})")
    check(observed == [["tag-1"], ["tag-2"], ["tag-3"]],
          f"the handler finished before the emitter continued (barrier): {observed}")
    check(toy_nodes.SwapNode.log == [("swap", 0), ("swap", 1), ("swap", 2)], "handlers ran once per event, in order")
    check(toy_nodes.ListenerNode.log == [1, 2, 3],
          f"a handler's own emit reached the next subscriber: {toy_nodes.ListenerNode.log}")
    kinds = [f for s, f in frames.frames if f.get("kind") == "event"]
    check(len(kinds) == 6 and kinds[0]["type"] == "EpochEnding" and kinds[0]["summary"] == {"epoch": 0, "samples": 10},
          f"events are mirrored with summaries ({len(kinds)} frames)")
    check(all(f["seq"] == i + 1 for i, f in enumerate(kinds)), "mirrored frames are numbered")

    # direct hub rules
    class Boom(Node):
        INPUTS = {}; OUTPUTS = {"x": Port("x", int)}
        EVENTS_IN = {"e": EventIn(EpochEnding, handler="h")}
        def build(self, **i): return {"x": 1}  # noqa: E704
        def h(self, ev, ctx): raise ValueError("handler broke")
    hub = EventHub()
    hub.connect(Wire("a", "e", "b", "e"), "h")
    hub.attach("b", Boom())
    raises(EventHandlerError, lambda: hub.emit("a", "e", EpochEnding(0, 1)), "a handler failure is attributed", "b.e")
    hub2 = EventHub()
    hub2.connect(Wire("a", "e", "b", "e"), "h")
    raises(EventDeliveryError, lambda: hub2.emit("a", "e", EpochEnding(0, 1)), "emit to an unbuilt subscriber", "before")

    class Echo(Node):
        INPUTS = {}; OUTPUTS = {"x": Port("x", int)}
        EVENTS_IN = {"e": EventIn(EpochEnding, handler="h")}
        EVENTS_OUT = {"o": EventOut(EpochEnding)}
        def build(self, **i): return {"x": 1}  # noqa: E704
        def h(self, ev, ctx): ctx.emit("o", ev)
    hub3 = EventHub()
    hub3.connect(Wire("a", "e", "b", "e"), "h")
    hub3.connect(Wire("b", "o", "b", "e"), "h")
    hub3.attach("b", Echo())
    raises(EventLoopError, lambda: hub3.emit("a", "e", EpochEnding(0, 1)), "a re-entrant loop is stopped", "loop")

    hub4 = EventHub()
    box = []
    def worker():
        try:
            hub4.emit("a", "e", EpochEnding(0, 1))
        except EventThreadError as exc:
            box.append(exc)
        hub4.post("a", "e", EpochEnding(1, 1))
    t = threading.Thread(target=worker); t.start(); t.join()
    check(len(box) == 1, "emit from another thread is refused")
    got = []
    class Sink(Node):
        INPUTS = {}; OUTPUTS = {"x": Port("x", int)}
        EVENTS_IN = {"e": EventIn(EpochEnding, handler="h")}
        def build(self, **i): return {"x": 1}  # noqa: E704
        def h(self, ev, ctx): got.append(ev.epoch)
    hub4.connect(Wire("a", "e", "s", "e"), "h"); hub4.attach("s", Sink())
    check(hub4.drain() == 1 and got == [1], "posted events are delivered by drain on the run thread")
    hub4.mark_stopped(); hub4.emit("a", "e", EpochEnding(2, 1))
    check(got == [1], "dispatch stops after a stop")


def settings_checks():
    r = runner()
    out = r.run(doc([("SettingsReaderNode", {})], settings={"device.default": "cpu"}))
    check(out.ok and out.results[0].outputs["device"] == "cpu", "graph setting reaches the node")
    out = runner(app_defaults={"device.default": "xpu:1"}).run(doc([("SettingsReaderNode", {})]))
    check(out.results[0].outputs["device"] == "xpu:1", "app default is used when the graph has none")
    out = runner(app_defaults={"device.default": "xpu:1"}).run(
        doc([("SettingsReaderNode", {})], settings={"device.default": "cpu"}))
    check(out.results[0].outputs["device"] == "cpu", "graph value beats app default")
    out = r.run(doc([("SettingsReaderNode", {"undeclared": True})]))
    check(not out.ok and "READS_SETTINGS" in out.error, "reading an undeclared key fails the node")
    snap = SettingsRegistry().snapshot({"seed.base": 5})
    check(snap.get("seed.base") == 5 and snap.origin("seed.base") == "graph" and snap.origin("device.default") == "default",
          "origin is tracked")
    raises(SettingsError, lambda: snap.get("no.such"), "unknown key raises")
    from ..graph.settings import SettingSpec
    dup = SettingsRegistry([SettingSpec("device.default", "x", "str", "a", contributed_by="n")])
    check(dup.errors and "duplicate" in dup.errors[0], "duplicate contributions are reported")


def dynamic_and_catalog():
    res = resolve_node(REGISTRY, "DynamicNode", {"file": "full"})
    check(res["state"] == "ready" and [p["name"] for p in res["outputs"]] == ["vae", "denoiser"], "resolve returns ports")
    check(resolve_node(REGISTRY, "DynamicNode", {"file": "ambiguous"})["choices"] == {"file": ["a", "b"]}, "computed choices")
    check(resolve_node(REGISTRY, "DynamicNode", {"file": "raises"})["state"] == "error", "a raising resolver is an error state")
    raises(KeyError, lambda: resolve_node(REGISTRY, "Nope", {}), "unknown class")
    cat = build_catalog(REGISTRY)
    text = json.dumps(cat)
    check(cat["catalog_format"] == 1 and not cat["load_errors"], f"catalog builds ({cat['load_errors']})")
    info = {n["class_name"]: n for n in cat["nodes"]}
    check(info["DynamicNode"]["dynamic"] == {"depends_on": ["file"], "debounce_ms": 300}, "dynamic marker is served")
    check([g["id"] for g in info["GroupedNode"]["groups"]] == ["main", "more"], "groups are served")
    check(info["AddNode"]["inputs"][0]["multi"] is True, "multi flag is served")
    check(info["EpochLoopNode"]["events_out"][0]["payload"] == "EpochEnding", "event ports are served")
    check("OtherHandle" in cat["handle_types"] and "object" in cat["handle_types"]["OtherHandle"], "handle MRO is served")
    check(info["SettingsReaderNode"]["reads_settings"] == ["device.default"], "READS_SETTINGS is served")
    check(len(text) > 1000 and info["ConstNode"]["display_name"] == "Const", "display names are derived")


def named_tasks():
    from ..graph import ExecutionContext

    class Frames:
        def __init__(self):
            self.frames = []

        def report(self, stream, frame):
            self.frames.append((stream, frame))

    reporter = Frames()
    context = ExecutionContext(reporter=reporter).scoped("n_task", (), "test")
    with context.task("Encoding", total=3) as task:
        task.update(1)
        task.update(3)
    states = [f["state"] for s, f in reporter.frames if s == "task"]
    check(states[0] == "started" and states[-1] == "finished" and all(f["node"] == "n_task" and f["title"] == "Encoding" for _, f in reporter.frames),
          "a task reports that it started and finished, under its title and its node")
    check(reporter.frames[-1][1]["done"] == 3 and reporter.frames[-1][1]["ok"] is True and len(states) <= 4, "progress is throttled but the last count always arrives")
    try:
        with context.task("Failing"):
            raise ValueError("x")
    except ValueError:
        pass
    check(reporter.frames[-1][1]["ok"] is False, "a task that raised says so, and the error still propagates")
    ExecutionContext().task("quiet").__enter__()  # a context without a reporter ignores tasks


if __name__ == "__main__":
    run(declarations, document, validation, running, events, settings_checks, dynamic_and_catalog, named_tasks)
