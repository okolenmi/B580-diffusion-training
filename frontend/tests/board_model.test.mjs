/* Unit tests for board/model.js -- the document rules. Pure; no DOM. */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  addBlock, autoFill, buildAvailable, defaultSize, docKey, emptyDoc, findTarget,
  getFieldValue, graphRef, layoutKey, newBlockId, nodeRef, refKey, removeBlock,
  setFieldValue, statusByBlock, titleFor, toSaveDoc, valuesPayload,
} from "../js/board/model.js";
import { collides } from "../js/board/layout.js";

const field = (name, extra = {}) => ({
  name, type: "int", required: false, default: null, wired: false, advanced: false,
  has_value: false, label: "", ...extra,
});
const modules = () => ({
  nodes: [
    {
      id: "n1", class_name: "Trainer", display_name: "Trainer", known: true, stream: "node:n1",
      fields: [
        field("steps", { has_value: true, value: 100, label: "Steps" }),
        field("lr", { type: "float", has_value: true }),
        field("seed", {}),                                   // unset, optional: not auto-exposed
        field("model", { type: "str", required: true }),     // required, no default: exposed
        field("rank_x", { advanced: true, has_value: true }), // advanced: not auto-exposed
        field("fed", { wired: true, has_value: true }),       // wired: never
        field("flag", { type: "bool", has_value: true }),
      ],
      displays: [
        { name: "loss", title: "Loss", kind: "chart", size: [8, 4] },
        { name: "progress", title: "Progress", kind: "progress", size: [12, 2] },
      ],
    },
    { id: "gone", class_name: "Missing", display_name: "Missing", known: false, fields: [], displays: [] },
  ],
  graph_fields: [
    { name: "memory.vram_max_mb", label: "VRAM budget", type: "float" },
    { name: "memory.strict", label: "Strict", type: "bool", advanced: true },
  ],
  graph_displays: [
    { name: "run", title: "Run status", size: [6, 3] },
    { name: "memory", title: "Device memory", size: [12, 5] },
  ],
});
const graph = () => ({
  nodes: [{ id: "n1", class_name: "Trainer", params: { steps: 100, lr: 0.0001 } }],
  edges: [], memory: { vram_max_mb: "auto" },
});

test("refKey distinguishes node, graph, field and display targets", () => {
  const keys = new Set([
    refKey(nodeRef("n1", "field", "x")), refKey(nodeRef("n1", "display", "x")),
    refKey(nodeRef("n2", "field", "x")), refKey(graphRef("field", "x")),
    refKey(graphRef("display", "x")),
  ]);
  assert.equal(keys.size, 5);
});

test("newBlockId never reuses an id, even after removals", () => {
  const blocks = [{ id: "b1" }, { id: "b3" }];
  assert.equal(newBlockId(blocks), "b3" === "b3" && !blocks.some((b) => b.id === "b3") ? "b3" : "b4");
  assert.ok(!blocks.some((b) => b.id === newBlockId(blocks)));
  assert.match(newBlockId([]), /^[A-Za-z0-9_-]{1,40}$/);
});

test("findTarget: present, missing node, missing module, graph scope", () => {
  const m = modules();
  assert.equal(findTarget(nodeRef("n1", "field", "steps"), m).descriptor.name, "steps");
  assert.equal(findTarget(nodeRef("zz", "field", "steps"), m).node, null);
  assert.equal(findTarget(nodeRef("n1", "field", "nope"), m).descriptor, null);
  assert.equal(findTarget(nodeRef("n1", "display", "loss"), m).descriptor.kind, "chart");
  assert.equal(findTarget(graphRef("display", "memory"), m).descriptor.title, "Device memory");
  assert.equal(findTarget(graphRef("field", "memory.nope"), m).descriptor, null);
  assert.deepEqual(findTarget(graphRef("field", "x"), null), { node: null, descriptor: null });
});

test("titleFor: custom title wins; orphan falls back to the saved name", () => {
  const m = modules();
  const b = { ref: nodeRef("n1", "field", "steps"), title: null };
  assert.equal(titleFor(b, m).title, "Steps");
  assert.equal(titleFor({ ...b, title: "Training length" }, m).title, "Training length");
  assert.equal(titleFor({ ref: nodeRef("zz", "field", "q"), title: null }, m).title, "q");
  assert.equal(titleFor({ ref: graphRef("field", "memory.vram_max_mb"), title: null }, m).subtitle, "Graph");
});

test("defaultSize: a field is two rows, json three, display from its module, narrow = full width", () => {
  const m = modules();
  assert.deepEqual(defaultSize(nodeRef("n1", "field", "flag"), m, 12), [4, 2]);
  assert.deepEqual(defaultSize({ ...nodeRef("n1", "field", "steps") }, m, 12), [4, 2]);
  assert.deepEqual(defaultSize(nodeRef("n1", "display", "loss"), m, 12), [8, 4]);
  assert.deepEqual(defaultSize(nodeRef("n1", "display", "loss"), m, 6), [6, 4]);
  assert.deepEqual(defaultSize(nodeRef("n1", "display", "loss"), m, 4), [4, 4]);
  assert.deepEqual(defaultSize(nodeRef("zz", "display", "q"), m, 12), [6, 4]);
});

test("tray: wired fields and unknown classes are never offered; advanced only on request", () => {
  const m = modules();
  const tray = buildAvailable(m, []);
  const all = tray.flatMap((g) => g.items.map((i) => i.key));
  assert.ok(!all.includes("node/n1/field/fed"), "wired field offered");
  assert.ok(!all.some((k) => k.startsWith("node/gone/")));
  assert.ok(!all.includes("node/n1/field/rank_x"), "advanced shown by default");
  assert.ok(!all.includes("graph/field/memory.strict"));
  const adv = buildAvailable(m, [], { showAdvanced: true }).flatMap((g) => g.items.map((i) => i.key));
  assert.ok(adv.includes("node/n1/field/rank_x") && adv.includes("graph/field/memory.strict"));
  assert.equal(tray[0].key, "graph", "graph group first");
});

test("tray: placed items are flagged, and an advanced placed item stays findable", () => {
  const m = modules();
  const blocks = [{ id: "b1", ref: nodeRef("n1", "field", "rank_x") }];
  const items = buildAvailable(m, blocks).find((g) => g.key === "node:n1").items;
  const it = items.find((i) => i.key === "node/n1/field/rank_x");
  assert.ok(it && it.placed);
  assert.ok(items.filter((i) => i.key !== it.key).every((i) => !i.placed));
});

test("addBlock: lands without overlap, twice is a no-op, remove closes the gap", () => {
  const m = modules();
  let doc = emptyDoc("template", graph());
  doc = addBlock(doc, nodeRef("n1", "field", "steps"), m);
  doc = addBlock(doc, nodeRef("n1", "display", "loss"), m);
  doc = addBlock(doc, nodeRef("n1", "field", "lr"), m);
  assert.equal(doc.blocks.length, 3);
  const again = addBlock(doc, nodeRef("n1", "field", "steps"), m);
  assert.equal(again, doc);
  for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++)
    assert.ok(!collides(doc.blocks[i], doc.blocks[j]));
  assert.equal(new Set(doc.blocks.map((b) => b.id)).size, 3);
  const removed = removeBlock(doc, doc.blocks[0].id);
  assert.equal(removed.blocks.length, 2);
});

test("autoFill exposes what the author set / what is required, never wired or advanced", () => {
  const doc = autoFill(modules(), 12);
  const keys = doc.map((b) => refKey(b.ref));
  for (const want of [
    "graph/field/memory.vram_max_mb", "node/n1/field/steps", "node/n1/field/lr",
    "node/n1/field/model", "node/n1/field/flag",
    "node/n1/display/loss", "node/n1/display/progress", "graph/display/run", "graph/display/memory",
  ]) assert.ok(keys.includes(want), `missing ${want}`);
  for (const bad of ["node/n1/field/seed", "node/n1/field/rank_x", "node/n1/field/fed"])
    assert.ok(!keys.includes(bad), `should not expose ${bad}`);
});

test("autoFill: no overlap, inside the grid, fields left / displays right, ids unique", () => {
  for (const columns of [12, 6, 4, 24]) {
    const blocks = autoFill(modules(), columns);
    for (const b of blocks) {
      assert.ok(b.x >= 0 && b.x + b.w <= columns, `${b.id} off grid at ${columns}`);
      assert.ok(Number.isInteger(b.y) && b.y >= 0);
    }
    for (let i = 0; i < blocks.length; i++) for (let j = i + 1; j < blocks.length; j++)
      assert.ok(!collides(blocks[i], blocks[j]), `${blocks[i].id}/${blocks[j].id} @${columns}`);
    assert.equal(new Set(blocks.map((b) => b.id)).size, blocks.length);
  }
  const wide = autoFill(modules(), 12);
  const leftmost = Math.min(...wide.filter((b) => b.ref.kind === "display").map((b) => b.x));
  const fieldRight = Math.max(...wide.filter((b) => b.ref.kind === "field").map((b) => b.x + b.w));
  assert.ok(leftmost >= fieldRight, "displays overlap the field column");
});

test("autoFill honours maxFields and copes with an empty graph", () => {
  const m = modules();
  m.nodes[0].fields = Array.from({ length: 40 }, (_, i) => field(`f${i}`, { has_value: true }));
  const nodeFields = autoFill(m, 12, { maxFields: 5 }).filter((b) => b.ref.scope === "node" && b.ref.kind === "field");
  assert.equal(nodeFields.length, 5);
  const empty = autoFill({ nodes: [], graph_fields: [], graph_displays: [] }, 12);
  assert.deepEqual(empty.map((b) => refKey(b.ref)), ["graph/display/run"]);
});

test("field values live in the draft graph; clearing deletes the key", () => {
  const doc = { ...emptyDoc("template", graph()) };
  const steps = nodeRef("n1", "field", "steps");
  assert.equal(getFieldValue(doc, steps), 100);
  assert.equal(setFieldValue(doc, steps, 250), true);
  assert.equal(doc.graph.nodes[0].params.steps, 250);
  assert.equal(setFieldValue(doc, steps, 250), false, "same value is not a change");
  assert.equal(setFieldValue(doc, steps, ""), true);
  assert.ok(!("steps" in doc.graph.nodes[0].params));
  assert.equal(getFieldValue(doc, steps), undefined);
  const vram = graphRef("field", "memory.vram_max_mb");
  assert.equal(getFieldValue(doc, vram), "auto");
  setFieldValue(doc, vram, 9000);
  assert.equal(doc.graph.memory.vram_max_mb, 9000);
  setFieldValue(doc, vram, null);
  assert.ok(!("vram_max_mb" in doc.graph.memory));
  assert.equal(setFieldValue(doc, nodeRef("nope", "field", "x"), 1), false);
});

test("valuesPayload: exposed fields only; cleared => null; unset memory omitted; refused statuses skipped", () => {
  const m = modules();
  let doc = emptyDoc("template", graph());
  doc = addBlock(doc, nodeRef("n1", "field", "steps"), m);
  doc = addBlock(doc, nodeRef("n1", "field", "seed"), m);   // exposed but unset
  doc = addBlock(doc, graphRef("field", "memory.vram_max_mb"), m);
  doc = addBlock(doc, graphRef("field", "memory.strict"), m); // unset memory key
  doc = addBlock(doc, nodeRef("n1", "display", "loss"), m);   // not a field
  assert.deepEqual(valuesPayload(doc), {
    nodes: { n1: { steps: 100, seed: null } }, memory: { vram_max_mb: "auto" },
  });
  const wiredBlock = doc.blocks.find((b) => b.ref.name === "seed");
  assert.deepEqual(valuesPayload(doc, { [wiredBlock.id]: "wired" }).nodes, { n1: { steps: 100 } });
  assert.deepEqual(valuesPayload(emptyDoc("monitor")), {});
});

test("dirty keys: layout changes layoutKey, a value changes only graphKey, live data neither", () => {
  const m = modules();
  let doc = addBlock(emptyDoc("template", graph()), nodeRef("n1", "field", "steps"), m);
  const l0 = layoutKey(doc), k0 = docKey(doc);
  setFieldValue(doc, nodeRef("n1", "field", "steps"), 7);
  assert.equal(layoutKey(doc), l0);
  assert.notEqual(docKey(doc), k0);
  const moved = { ...doc, blocks: doc.blocks.map((b) => ({ ...b, x: b.x + 1 })) };
  assert.notEqual(layoutKey(moved), l0);
  const renamed = { ...doc, blocks: doc.blocks.map((b) => ({ ...b, title: "T" })) };
  assert.notEqual(layoutKey(renamed), l0);
});

test("toSaveDoc stamps format and normalises optional block fields; statusByBlock maps ids", () => {
  const doc = toSaveDoc({ ...emptyDoc("monitor"), blocks: [{ id: "b1", ref: graphRef("display", "run"), x: 0, y: 0, w: 1, h: 1 }] });
  assert.equal(doc.format, 1);
  assert.deepEqual([doc.blocks[0].collapsed, doc.blocks[0].title, doc.blocks[0].options], [false, null, {}]);
  assert.deepEqual(statusByBlock({ blocks: [{ id: "b1", status: "wired" }] }), { b1: "wired" });
  assert.deepEqual(statusByBlock(null), {});
});
