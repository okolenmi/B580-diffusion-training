"""Self-checks for addons: declarations, the graph file, validation, building, slots filled by wire, reuse on a second
host class, event scopes (bubbling, stopping, wires), and the outside API with addon-set files.

    python -m next.tests.check_addons
"""

from __future__ import annotations

import json

from ..graph import (Addon, AddonError, AddonSlot, AddonSpec, Edge, GraphDoc, GraphRunner, Node, NodeRegistry, NodeSpec,
                     Port, apply_set, available, configure, disable, enable, parse, remove)
from ..graph.addon_edit import host_of
from . import toy_addons
from .support import check, raises, run

REGISTRY = NodeRegistry(packages=("next.tests.toy_addons",))
CALC, OTHER, FACTOR, SINK, STEPPER, PLAIN = (f"n_{c * 6}" for c in "abcdef")


def spec(uid, cls, params=None, addons=()):
    return NodeSpec(uid, cls, dict(params or {}), addons=tuple(addons))


def doc(nodes, edges=(), event_edges=(), addons=()):
    return GraphDoc(tuple(nodes), tuple(Edge(*e) for e in edges), tuple(Edge(*e) for e in event_edges), addons=tuple(addons))


def codes(d):
    return {i.code for i in GraphRunner(REGISTRY).validate(d).issues}


def result(d, uid=CALC, port="y"):
    outcome = GraphRunner(REGISTRY).run(d)
    assert outcome.ok, outcome.error
    return {r.uid: r.outputs for r in outcome.results}[uid][port]


def declarations():
    def define(**attrs):
        return type("Bad", (Addon,), attrs)
    raises(TypeError, lambda: define(), "an addon names its slot", "SLOT")
    raises(TypeError, lambda: define(SLOT="x", INPUTS={"a.b": Port("a.b", int)}), "a parameter name has no dot", "may not contain")
    raises(TypeError, lambda: type("BadHost", (Node,), {"OUTPUTS": {"o": Port("o", int)}, "build": lambda s, **i: {},
                                                        "ADDON_SLOTS": (AddonSlot("a", object), AddonSlot("a", object))}),
           "a host's slots are unique", "unique")
    classes, _ = REGISTRY.load()
    check(set(REGISTRY.addons()) >= {"Scale", "Shift", "Trigger"} and "Scale" not in classes, "addons are found beside nodes, not among them")


def document():
    plain = doc([spec(CALC, "Calc")])
    check("addons" not in plain.as_dict() and "addons" not in plain.as_dict()["nodes"][0], "no addons, no key in the file")
    d = doc([spec(CALC, "Calc", addons=[AddonSpec("a1", "Scale", True, {"by": 3.0})])], addons=[AddonSpec("g1", "Count")])
    back = parse(json.loads(json.dumps(d.as_dict())))
    check(back.as_dict() == d.as_dict() and back.node(CALC).addons[0].params == {"by": 3.0}, "addons round trip through the file")
    check(d.for_run().node(CALC).addons == d.node(CALC).addons and d.for_run().addons == d.addons, "a run keeps the addons")


def validation():
    scale = lambda i="a1", **p: AddonSpec(i, "Scale", True, p)  # noqa: E731
    on = lambda *addons, cls="Calc": doc([spec(CALC, cls, addons=addons)])  # noqa: E731
    check(codes(on(scale())) == set(), "a fitting addon validates")
    check("addon_unknown_kind" in codes(on(AddonSpec("a1", "Nope"))), "unknown kind")
    check("addon_slot_missing" in codes(doc([spec(PLAIN, "Plain", addons=[scale()])])), "slot the host lacks")
    check("addon_slot_taken" in codes(on(scale("a1"), scale("a2"))), "one-slot filled by two addons")
    check("addon_slot_taken" not in codes(on(AddonSpec("a1", "Shift"), AddonSpec("a2", "Shift"))), "a many slot takes several")
    check("addon_duplicate_id" in codes(on(AddonSpec("a1", "Shift"), AddonSpec("a1", "Shift"))), "ids are unique on a host")
    check("addon_bad_id" in codes(on(AddonSpec("A 1", "Shift"))), "id shape")
    check("addon_requires" in codes(on(AddonSpec("a1", "NeedsScale"))) and "addon_requires" not in codes(on(scale(), AddonSpec("a2", "NeedsScale"))),
          "REQUIRES")
    check("addon_conflict" in codes(on(AddonSpec("a1", "Shift"), AddonSpec("a2", "NoShift"))), "CONFLICTS")
    check("addon_not_offered" in codes(doc([spec(OTHER, "CalcTwo", addons=[AddonSpec("a1", "OnlyCalc")])])), "FITS limits the hosts")
    check("type_mismatch" in codes(on(scale(by="x"))) and "out_of_range" in codes(on(scale(by=-1.0))), "addon parameters are checked like node ports")
    check("unknown_param" in codes(on(scale(zzz=1))), "unknown addon parameter")
    check(codes(on(AddonSpec("a1", "Scale", False, {"by": "x"}))) == set(), "a disabled addon is not validated, and counts for no slot")
    check("addon_slot_taken" not in codes(on(scale("a1"), AddonSpec("a2", "Scale", False))), "a disabled addon leaves its slot free")
    check("addon_slot_missing" in codes(doc([], addons=[scale()])), "the graph is a host with only the events slot")
    check(codes(doc([], addons=[AddonSpec("g1", "Count")])) == set(), "a graph-scope event addon validates")
    wired = doc([spec(CALC, "Calc", addons=[scale()]), spec(FACTOR, "FactorNode")], [(FACTOR, "factor", CALC, "slot:factor")])
    check("addon_slot_taken" in codes(wired), "a slot filled by an addon and by a wire")


def building():
    d = lambda *addons, x=2.0: doc([spec(CALC, "Calc", {"x": x}, addons)])  # noqa: E731
    check(result(d()) == 2.0, "no addon: the host's default behaviour")
    check(result(d(AddonSpec("a1", "Scale", True, {"by": 3.0}))) == 6.0, "an addon's parameters reach it")
    check(result(d(AddonSpec("a1", "Scale"))) == 4.0, "an addon's default is used")
    check(result(d(AddonSpec("a1", "Shift", True, {"by": 1.0}), AddonSpec("a2", "Scale"), AddonSpec("a3", "Shift", True, {"by": 5.0}))) == 10.0,
          "saved order is the order a many slot applies")
    check(result(d(AddonSpec("a1", "Scale", False, {"by": 3.0}))) == result(d()), "a disabled addon leaves the run identical")
    const = doc([spec(CALC, "Calc", {"x": 2.0}, [AddonSpec("a1", "Scale")]), spec(FACTOR, "Number", {"value": 7.0})],
                [(FACTOR, "value", CALC, "a1.by")])
    check(result(const) == 14.0, "an addon parameter can be wired like a node's")
    wire = doc([spec(CALC, "Calc", {"x": 2.0}), spec(FACTOR, "FactorNode", {"by": 4.0})], [(FACTOR, "factor", CALC, "slot:factor")])
    check(result(wire) == 8.0, "a slot filled by a wire works like an addon")
    other = doc([spec(OTHER, "OtherCalc", {"x": 2.0}, [AddonSpec("a1", "Scale", True, {"by": 5.0})])])
    check(result(other, OTHER) == 10.0, "the same addon class serves a second host class")
    bad = GraphRunner(REGISTRY).run(doc([spec(CALC, "Calc", addons=[AddonSpec("a1", "Nope")])]))
    check(not bad.ok and "addon" in bad.error, "an invalid addon stops the run before it starts")


def events():
    toy_addons.LOG.clear()
    trigger = lambda every=2: AddonSpec("a1", "Trigger", True, {"every": every})  # noqa: E731
    result(doc([spec(SINK, "Sink"), spec(STEPPER, "Stepper", {"steps": 5}, [trigger()])]), STEPPER, "done")
    check([e for e in toy_addons.LOG if e[0] == "sink"] == [("sink", 2), ("sink", 4)],
          "a trigger on the host hears its steps; its event rises to the graph and a scoped consumer hears it")
    toy_addons.LOG.clear()
    result(doc([spec(SINK, "Sink"), spec(STEPPER, "Stepper", {"steps": 5})], addons=[trigger(3)]), STEPPER, "done")
    check([e for e in toy_addons.LOG if e[0] == "sink"] == [("sink", 3)], "a graph-scope trigger hears the steps of every node")
    toy_addons.LOG.clear()
    result(doc([spec(STEPPER, "Stepper", {"steps": 2}, [AddonSpec("a1", "Stop")])], addons=[AddonSpec("g1", "Count")]), STEPPER, "done")
    check(not [e for e in toy_addons.LOG if e[0] == "count"], "a handler can stop an event from rising to the graph")
    toy_addons.LOG.clear()
    result(doc([spec(STEPPER, "Stepper", {"steps": 2}, [AddonSpec("a1", "Count")])], addons=[AddonSpec("g1", "Count")]), STEPPER, "done")
    check(sorted(e[1] for e in toy_addons.LOG) == ["graph.g1"] * 2 + [f"{STEPPER}.a1"] * 2
          and [e[1] for e in toy_addons.LOG][:1] == [f"{STEPPER}.a1"], "without a stop everyone on the way hears it, the host's addon first")
    toy_addons.LOG.clear()
    wired = doc([spec(STEPPER, "Stepper", {"steps": 2}, [AddonSpec("a1", "Count")])], event_edges=[(STEPPER, "step", f"{STEPPER}.a1", "step")])
    check(codes(wired) == set(), "an event wire may end at an addon")
    result(wired, STEPPER, "done")
    check(len([e for e in toy_addons.LOG if e[0] == "count"]) == 2, "a wire and the scope together deliver once")
    ghost = doc([spec(STEPPER, "Stepper")], event_edges=[(STEPPER, "step", f"{STEPPER}.a9", "step")])
    check("edge_unknown_node" in codes(ghost), "an event wire to a missing addon is refused")
    wrong = doc([spec(STEPPER, "Stepper", addons=[AddonSpec("a1", "Count")])], event_edges=[(STEPPER, "step", f"{STEPPER}.a1", "zzz")])
    check("unknown_event_in" in codes(wrong), "an event wire names the addon's own event inputs")


def outside():
    d = doc([spec(CALC, "Calc", {"x": 2.0}), spec(OTHER, "OtherCalc")])
    kinds = {a["kind"] for a in available(d, REGISTRY, CALC)}
    check({"Scale", "Shift", "OnlyCalc", "Trigger"} <= kinds and "OnlyCalc" not in {a["kind"] for a in available(d, REGISTRY, OTHER)},
          "available() lists what a host's slots accept")
    check({a["kind"] for a in available(d, REGISTRY, "graph")} == {"Trigger", "Count", "Stop"}, "the graph offers the event addons")
    made = enable(d, REGISTRY, "Calc", "Scale", {"by": 3.0})
    check(d.node(CALC).addons == () and made.node(CALC).addons[0].id == "a1" and result(made) == 6.0, "enable() returns a new document")
    check(configure(made, REGISTRY, CALC, "a1", by=5.0).node(CALC).addons[0].params == {"by": 5.0}, "configure() merges parameters")
    off = disable(made, REGISTRY, CALC, "a1")
    check(not off.node(CALC).addons[0].enabled and off.node(CALC).addons[0].params == {"by": 3.0}, "disable() keeps the parameters")
    check(remove(made, REGISTRY, CALC, "a1").node(CALC).addons == (), "remove() drops it")
    raises(AddonError, lambda: enable(d, REGISTRY, CALC, "Nope"), "an unknown kind is refused", "Nope")
    raises(AddonError, lambda: enable(made, REGISTRY, CALC, "Scale"), "a taken slot is refused", "holds one")
    raises(AddonError, lambda: enable(d, REGISTRY, CALC, "Scale", {"by": -2.0}), "a bad parameter is refused", "below")
    raises(AddonError, lambda: configure(made, REGISTRY, CALC, "a9", by=1.0), "an unknown addon id is refused", "no addon")
    both = enable(enable(d, REGISTRY, CALC, "Scale"), REGISTRY, CALC, "NeedsScale")
    raises(AddonError, lambda: remove(both, REGISTRY, CALC, "a1"), "removing what another addon requires is refused", "needs")
    merged = apply_set(d, REGISTRY, {"Calc": [{"kind": "Scale", "params": {"by": 3.0}}, {"kind": "Shift", "id": "s"}],
                                     "graph": [{"kind": "Count"}]})
    check(result(merged) == 7.0 and merged.addons[0].kind == "Count", "an addon set merges several hosts in one go")
    raises(AddonError, lambda: apply_set(d, REGISTRY, {"Calc": [{"kind": "Scale"}, {"kind": "Scale"}]}), "a set that breaks a rule is refused whole", "holds one")
    raises(AddonError, lambda: apply_set(d, REGISTRY, {"Missing": []}), "an unknown node is refused", "Missing")
    two = doc([spec(CALC, "Calc"), spec(OTHER, "Calc")])
    raises(AddonError, lambda: host_of(two, "Calc"), "a class name used twice does not pick a node", "2 nodes")


if __name__ == "__main__":
    run(declarations, document, validation, building, events, outside)
