import { test } from "node:test";
import assert from "node:assert/strict";
import { GraphDoc } from "../js/graph2/doc.js";
import { InstanceView } from "../js/graph2/instance.js";

const button = () => ({ handlers: {}, addEventListener(t, f) { this.handlers[t] = f; }, click() { this.handlers.click(); } });
function setup(confirm = () => true) {
  const doc = new GraphDoc();
  const notes = [];
  const log = { held: [], flushed: 0, left: 0, changes: [], cleared: 0 };
  const parts = {
    doc,
    canvas: { fit() {}, clearStatuses() { log.cleared++; } },
    saver: { hold: (on) => log.held.push(on), flush: () => log.flushed++ },
    bar: { hidden: true }, title: { textContent: "" }, copyBtn: button(), backBtn: button(),
    onNote: (m) => notes.push(m), onChange: (o) => log.changes.push(o), onLeave: () => log.left++, confirm,
  };
  return { doc, view: new InstanceView(parts), parts, notes, log };
}
const run = (uid) => ({ execution_id: 7, graph: { format: 3, settings: {}, nodes: [{ uid, class_name: "IntNode", title: "", params: { value: 3 }, layout: { x: 1, y: 2 } }], edges: [], event_edges: [], views: [] } });

test("a frozen run's graph replaces the canvas, cannot be edited, and the user's own graph comes back", () => {
  const { doc, view, parts, log } = setup();
  const mine = doc.addNode("IntNode", { params: { value: 1 } }).uid;
  assert.equal(view.enter(run("n_run")), true);
  assert.deepEqual([...doc.nodes.keys()], ["n_run"]);
  assert.equal(parts.bar.hidden, false);
  assert.equal(doc.setParam("n_run", "value", 99), null);
  assert.equal(doc.nodes.get("n_run").params.value, 3);
  assert.equal(doc.addNode("IntNode"), null);
  assert.equal(doc.nodes.size, 1);
  view.back();
  assert.deepEqual([...doc.nodes.keys()], [mine]);
  assert.equal(doc.nodes.get(mine).params.value, 1);
  assert.deepEqual(log.held, [true, false]);
  assert.equal(parts.bar.hidden, true);
  assert.equal(log.left, 1);
});

test("a refused edit redraws the true document, but a node drag does not", () => {
  const { doc, view } = setup();
  view.enter(run("n_a"));
  let resets = 0;
  doc.on("reset", () => resets++);
  doc.setParam("n_a", "value", 5);
  assert.equal(resets, 1);
  doc.moveNode("n_a", 10, 10);
  assert.equal(resets, 1);
});

test("opening a second run keeps the user's own graph, not the first run's", () => {
  const { doc, view } = setup();
  const mine = doc.addNode("IntNode").uid;
  view.enter(run("n_one"));
  view.enter({ ...run("n_two"), execution_id: 8 });
  view.back();
  assert.deepEqual([...doc.nodes.keys()], [mine]);
});

test("copy to editor keeps the run's graph, editable, and asks before dropping a graph with content", () => {
  const refuse = setup(() => false);
  refuse.doc.addNode("IntNode");
  refuse.view.enter(run("n_run"));
  refuse.view.copyToEditor();
  assert.equal(refuse.view.open, true);
  assert.equal(refuse.doc.readOnly, true);

  const { doc, view, log } = setup();
  view.enter(run("n_run"));           // the canvas was empty: nothing to ask
  view.copyToEditor();
  assert.equal(view.open, false);
  assert.equal(doc.readOnly, false);
  assert.deepEqual([...doc.nodes.keys()], ["n_run"]);
  doc.setParam("n_run", "value", 9);
  assert.equal(doc.nodes.get("n_run").params.value, 9);
  assert.equal(log.flushed, 1);
});

test("a run without a graph is reported and nothing changes", () => {
  const { doc, view, notes } = setup();
  doc.addNode("IntNode");
  assert.equal(view.enter({ execution_id: 3, graph: {} }), false);
  assert.equal(doc.nodes.size, 1);
  assert.equal(doc.readOnly, false);
  assert.match(notes[0], /no graph/);
});
