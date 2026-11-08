/* graph2 document model: wires, uids, round trip, addons, copy and paste. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { GraphDoc, GRAPH_FORMAT, endpointHost, newUid } from "../js/graph2/doc.js";
import { eventWireProblem, fieldValues, groupPorts, isVisible, portVisible, wireProblem } from "../js/graph2/ports.js";

const port = (name, type, extra = {}) => ({ name, label: name, type, type_mro: [type, "object"], ...extra });
const SHAPES = {
  Source: { inputs: [], outputs: [port("y", "int", { type_mro: ["int", "float", "object"] })], events_out: [{ name: "tick", payload: "StepFinished", payload_mro: ["StepFinished", "EventType"] }], events_in: [] },
  Sink: { inputs: [port("x", "float"), port("many", "float", { multi: true }), port("note", "str", { widget_only: true })], outputs: [port("z", "float")], events_in: [{ name: "tick", payload: "EventType" }], events_out: [] },
  Other: { inputs: [port("x", "str")], outputs: [], events_in: [], events_out: [] },
};
const makeDoc = () => {
  let n = 0;
  return new GraphDoc({ shapeOf: (node) => SHAPES[node.class_name] || null, random: () => ((n++ * 7919) % 1000) / 1000 });
};

test("uids are n_ plus six base32 characters and never repeat", () => {
  const taken = new Set();
  for (let i = 0; i < 50; i++) {
    const uid = newUid(taken, Math.random);
    assert.match(uid, /^n_[a-z2-7]{6}$/);
    assert.ok(!taken.has(uid));
    taken.add(uid);
  }
});

test("a single-input port keeps one wire, a multi port keeps them in order", () => {
  const d = makeDoc();
  const a = d.addNode("Source"), b = d.addNode("Source"), s = d.addNode("Sink");
  assert.ok(d.connect(a.uid, "y", s.uid, "x").ok);
  const second = d.connect(b.uid, "y", s.uid, "x");
  assert.equal(second.replaced.length, 1);
  assert.deepEqual(d.edgesInto(s.uid, "x").map((e) => e.from_node), [b.uid]);
  d.connect(a.uid, "y", s.uid, "many");
  d.connect(b.uid, "y", s.uid, "many");
  assert.deepEqual(d.edgesInto(s.uid, "many").map((e) => e.from_node), [a.uid, b.uid]);
  d.reorderInto(s.uid, "many", 1, 0);
  assert.deepEqual(d.edgesInto(s.uid, "many").map((e) => e.from_node), [b.uid, a.uid]);
});

test("wire checks: type, widget-only, self loop, cycle", () => {
  const d = makeDoc();
  const a = d.addNode("Source"), s = d.addNode("Sink"), o = d.addNode("Other");
  assert.match(d.connect(a.uid, "y", o.uid, "x").reason, /does not satisfy/);
  assert.match(d.connect(a.uid, "y", s.uid, "note").reason, /takes no wire/);
  assert.match(d.connect(s.uid, "z", s.uid, "x").reason, /itself/);
  const t = d.addNode("Sink");
  d.connect(a.uid, "y", s.uid, "x");
  d.connect(s.uid, "z", t.uid, "x");
  assert.match(d.connect(t.uid, "z", s.uid, "many").reason, /loop/);
});

test("a wire whose port vanished stays and is reported as dangling", () => {
  let shape = SHAPES.Sink;
  const d = new GraphDoc({ shapeOf: (n) => (n.class_name === "Sink" ? shape : SHAPES[n.class_name]) });
  const a = d.addNode("Source"), s = d.addNode("Sink");
  d.connect(a.uid, "y", s.uid, "x");
  shape = { ...SHAPES.Sink, inputs: [] };
  assert.equal(d.edges.length, 1);
  assert.equal(d.dangling().length, 1);
});

test("removing a node removes its wires and emits the changes", () => {
  const d = makeDoc();
  const seen = [];
  d.on("change", (e) => seen.push(e.type));
  const a = d.addNode("Source"), s = d.addNode("Sink");
  d.connect(a.uid, "y", s.uid, "x");
  d.removeNode(a.uid);
  assert.equal(d.edges.length, 0);
  assert.deepEqual(seen, ["node:add", "node:add", "edge:add", "edge:remove", "node:remove"]);
});

test("setParam stores, removes on undefined, and ignores an equal value", () => {
  const d = makeDoc();
  const n = d.addNode("Sink");
  let count = 0;
  d.on("node:params", () => count++);
  d.setParam(n.uid, "note", "hi");
  d.setParam(n.uid, "note", "hi");
  assert.equal(count, 1);
  d.setParam(n.uid, "note", undefined);
  assert.deepEqual(d.nodes.get(n.uid).params, {});
  assert.equal(count, 2);
});

test("round trip keeps unknown keys; a run payload has no layout or views; other formats are refused", () => {
  const d = makeDoc();
  const a = d.addNode("Source", { x: 5, y: 6, title: "first" });
  d.setLayout(a.uid, { collapsed: true });
  d.setSetting("memory.vram_max_mb", 9000);
  d.setViews([{ id: "v" }]);
  d.extras.future = { keep: 1 };
  const json = d.toJSON();
  assert.equal(json.format, GRAPH_FORMAT);
  const again = GraphDoc.fromJSON(JSON.parse(JSON.stringify(json)));
  assert.deepEqual(again.toJSON(), json);
  const run = d.runPayload();
  assert.ok(!("views" in run) && !("layout" in run.nodes[0]));
  assert.equal(run.future.keep, 1);
  assert.throws(() => GraphDoc.fromJSON({ format: 2, nodes: [] }), /format 2/);
});

test("addons get readable ids; removing one drops the wires that end at it", () => {
  const d = makeDoc();
  const n = d.addNode("Sink");
  const a = d.addAddon(n.uid, "RegionLadder", { start: 9 });
  const b = d.addAddon(n.uid, "RegionLadder");
  assert.deepEqual([a.id, b.id], ["region_ladder", "region_ladder2"]);
  const src = d.addNode("Source");
  d.connectEvent(`${src.uid}`, "tick", `${n.uid}.${a.id}`, "tick");
  d.removeAddon(n.uid, a.id);
  assert.equal(d.eventEdges.length, 0);
  assert.equal(endpointHost(`${n.uid}.x`), n.uid);
  d.addAddon("graph", "EveryNSteps");
  assert.equal(d.addons[0].id, "every_n_steps");
});

test("copy and paste give new uids, keep inner wires only, and offset the layout", () => {
  const d = makeDoc();
  const a = d.addNode("Source", { x: 10, y: 10 }), s = d.addNode("Sink", { x: 300, y: 10 }), t = d.addNode("Sink");
  d.connect(a.uid, "y", s.uid, "x");
  d.connect(a.uid, "y", t.uid, "x");
  const added = d.paste(d.copy([a.uid, s.uid]), 50, 60);
  assert.equal(added.length, 2);
  assert.ok(added.every((u) => ![a.uid, s.uid, t.uid].includes(u)));
  assert.equal(d.edges.filter((e) => added.includes(e.from_node)).length, 1);
  assert.equal(d.nodes.get(added[0]).layout.x, 60);
});

test("port helpers: wire problems, visible_when, grouping", () => {
  assert.equal(wireProblem(port("y", "int", { type_mro: ["int", "float"] }), port("x", "float")), null);
  assert.ok(wireProblem(port("y", "int"), port("x", "str")));
  assert.equal(eventWireProblem({ payload: "A", payload_mro: ["A", "B"] }, { payload: "B" }), null);
  assert.ok(eventWireProblem({ payload: "A", payload_mro: ["A"] }, { payload: "B" }));
  const gated = { name: "g", visible_when: ["mode", ["a", "b"]] };
  assert.ok(isVisible(gated, { mode: "b" }) && !isVisible(gated, { mode: "c" }));
  assert.ok(!portVisible({ name: "p", advanced: true }, {}, false) && portVisible({ name: "p", advanced: true }, {}, true));
  const groups = groupPorts([{ name: "a", group: "x" }, { name: "b" }, { name: "c", group: "gone" }], [{ id: "x", label: "X" }]);
  assert.deepEqual(groups.map((g) => [g.group.id, g.ports.map((p) => p.name)]), [["main", ["b", "c"]], ["x", ["a"]]]);
  assert.deepEqual(fieldValues([{ name: "k", default: 3 }, { name: "m", default: 1 }], { m: 2 }), { k: 3, m: 2 });
});
