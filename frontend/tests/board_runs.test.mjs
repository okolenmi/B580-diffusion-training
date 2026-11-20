/* Unit tests for what Phase 5 added to the board: the sentence and result displays' readers, the runs cards' pure helpers, and putting a
   past run's values back on the page ("Run again"). Pure; the cards themselves (DOM) are checked in the browser run. */
import { test } from "node:test";
import assert from "node:assert/strict";

import { StreamState, lineOf, resultOf } from "../js/board/series.js";
import { ActionCard } from "../js/board/actions.js";
import { describeValues, HistoryCard, QueueCard, runSeconds, whenText } from "../js/board/runs.js";
import { applyValues, autoFill, autoPage, collectIds, getFieldValue, graphRef, nodeRef } from "../js/board/model.js";

const fed = (...frames) => { const s = new StreamState(); frames.forEach((f, i) => s.applyFrame(f, i)); return s; };

test("a line shows the newest sentence a node reported, kept across frames that do not carry one", () => {
  const module = { value_key: "status" };
  assert.equal(lineOf(module, fed()), null);
  const s = fed({ step: 0, status: "Preparing" }, { step: 1, loss: 0.4 });
  assert.equal(lineOf(module, s), "Preparing");
  s.applyFrame({ step: 2, status: "Training · step 2 of 10" });
  assert.equal(lineOf(module, s), "Training · step 2 of 10");
});

test("a result lists only the keys that were reported, numbers formatted and text as it came", () => {
  const module = { series: [{ key: "step", label: "steps done" }, { key: "file", label: "file" }, { key: "missing", label: "never" }, { key: "loss", label: "last loss", unit: "" }] };
  const s = fed({ step: 250, file: "adapter_step000250.safetensors", loss: 0.123456 });
  assert.deepEqual(resultOf(module, s).map((r) => [r.label, r.text]), [
    ["steps done", "250"], ["file", "adapter_step000250.safetensors"], ["last loss", "0.123"],
  ]);
});

test("describeValues names the first few values and counts the rest; a cleared value reads as default", () => {
  const values = { nodes: { a: { steps: 30, cfg_scale: 5.5, seed: null }, b: { prompt: "x".repeat(60) } }, memory: { vram_max_mb: 8192 } };
  assert.equal(describeValues(values), `steps 30, cfg scale 5.5, seed default, prompt ${"x".repeat(27)}… +1`);
  assert.equal(describeValues({}), "");
  assert.equal(describeValues(null), "");
});

test("runSeconds counts to now for a live run, to its end for a finished one, and is null before it starts", () => {
  const now = Date.parse("2026-11-20T12:00:30Z");
  assert.equal(runSeconds({ started_at: null }, now), null);
  assert.equal(runSeconds({ started_at: "2026-11-20T12:00:00Z", finished_at: null }, now), 30);
  assert.equal(runSeconds({ started_at: "2026-11-20T12:00:00Z", finished_at: "2026-11-20T12:02:00Z" }, now), 120);
});

test("whenText gives the time for today and the date with it for another day", () => {
  const now = new Date(2026, 10, 20, 15, 0);
  const same = whenText(new Date(2026, 10, 20, 9, 5).toISOString(), now);
  const other = whenText(new Date(2026, 10, 19, 9, 5).toISOString(), now);
  assert.ok(!/Nov/.test(same) && /Nov/.test(other), `${same} / ${other}`);
  assert.equal(whenText("not a date", now), "");
});

function page() {
  const field = (id, ref) => ({ id, ref, x: 0, y: 0, w: 4, h: 2, collapsed: false, title: null, options: {} });
  const blocks = [
    field("b1", nodeRef("n1", "field", "steps")),
    field("b2", nodeRef("n1", "field", "seed")),
    field("b3", graphRef("field", "memory.vram_max_mb")),
    field("b4", nodeRef("n2", "field", "wired_port")),
  ];
  const graph = { nodes: [{ id: "n1", params: { steps: 10, seed: 3 } }, { id: "n2", params: { wired_port: 1 } }], memory: { vram_max_mb: 4096 } };
  return { columns: 12, blocks, graph };
}

test("applyValues puts a run's values back: set, clear on null, leave what the run did not record", () => {
  const doc = page();
  const changed = applyValues(doc, { nodes: { n1: { steps: 30, seed: null } }, memory: { vram_max_mb: 8192 } });
  assert.equal(changed, 3);
  assert.equal(getFieldValue(doc, nodeRef("n1", "field", "steps")), 30);
  assert.equal(getFieldValue(doc, nodeRef("n1", "field", "seed")), undefined);
  assert.equal(getFieldValue(doc, graphRef("field", "memory.vram_max_mb")), 8192);
  assert.equal(getFieldValue(doc, nodeRef("n2", "field", "wired_port")), 1, "not in the run: untouched");
  assert.equal(applyValues(doc, { nodes: { n1: { steps: 30 } } }), 0, "nothing new: nothing changed");
  assert.equal(applyValues(doc, {}), 0);
});

test("the draft builder puts the app's history after the live views when the graph offers it", () => {
  const modules = {
    graph_fields: [], graph_displays: [{ name: "run" }, { name: "history" }],
    nodes: [{ id: "n1", known: true, fields: [], actions: [], displays: [{ name: "loss", kind: "chart", size: [6, 4] }] }],
  };
  const names = autoFill(modules, 12).map((b) => b.ref.name);
  assert.deepEqual(names, ["run", "loss", "history"]);
});

test("autoPage sorts a training graph into Run, Charts and Runs tabs with ids unique across them; one kind of view stays one tab", () => {
  const node = (displays, actions = []) => ({
    id: "n1", known: true, fields: [{ name: "steps", has_value: true, wired: false, advanced: false, type: "int" }], actions, displays,
  });
  const modules = {
    graph_fields: [], graph_displays: [{ name: "run" }, { name: "memory", kind: "chart" }, { name: "history", kind: "history" }, { name: "queue", kind: "queue" }],
    nodes: [node(
      [{ name: "progress", kind: "progress", size: [8, 2] }, { name: "loss", kind: "chart", size: [8, 6] }, { name: "status", kind: "line", size: [8, 1] }, { name: "result", kind: "result", size: [4, 3] }],
      [{ name: "save_now", available: true }],
    )],
  };
  const tabs = autoPage(modules, 12);
  assert.deepEqual(tabs.map((t) => t.title), ["Run", "Charts", "Runs"]);
  const names = (tab) => tab.blocks.filter((b) => b.ref.kind !== "field").map((b) => b.ref.name);
  assert.deepEqual(names(tabs[0]), ["save_now", "run", "progress", "status", "result"]);
  assert.deepEqual(names(tabs[1]), ["loss", "memory"]);
  assert.deepEqual(names(tabs[2]), ["history", "queue"]);
  assert.equal(tabs[1].blocks[0].x, 0, "tabs without inputs use the whole width");
  assert.equal(collectIds(tabs.flatMap((t) => t.blocks)).size, tabs.reduce((n, t) => n + t.blocks.length, 0), "ids unique across tabs");

  const plain = autoPage({ graph_fields: [], graph_displays: [{ name: "run" }], nodes: [] }, 12);
  assert.deepEqual(plain.map((t) => t.title), ["Main"]);
});

test("every card class a board keeps as a display can be destroyed (leaving a tab drops them all)", () => {
  for (const Card of [ActionCard, HistoryCard, QueueCard]) assert.equal(typeof Card.prototype.destroy, "function", Card.name);
});
