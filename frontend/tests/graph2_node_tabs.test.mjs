/* graph2 node tabs: which tabs a node shows, which optional inputs are drawn, which addons fit. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { activeTab, addonChoices, optionalSockets, shownOptional, tabOfHidden, tabsOf } from "../js/graph2/node_tabs.js";

const p = (name, extra = {}) => ({ name, type: "Thing", required: false, field: false, widget_only: false, ...extra });
const shape = { inputs: [p("model", { required: true }), p("lr", { type: "float", field: true }), p("mask"), p("prompts")] };
const kinds = new Map([
  ["Ladder", { kind: "Ladder", slot: "ladder", fits: [] }],
  ["Warm", { kind: "Warm", slot: "lr_modifier", fits: [] }],
  ["OnlyHere", { kind: "OnlyHere", slot: "ladder", fits: ["Other"] }],
]);
const info = { class_name: "Noiser", addon_slots: [{ name: "ladder", many: false }, { name: "lr_modifier", many: true }, { name: "events", many: true }] };
const node = (extra = {}) => ({ uid: "n1", layout: {}, addons: [], ...extra });

test("optional sockets are the non-required handle inputs, not fields", () => {
  assert.deepEqual(optionalSockets(shape.inputs).map((x) => x.name), ["mask", "prompts"]);
});

test("an optional input is drawn when enabled or wired, never hidden under a wire", () => {
  const n = node({ layout: { optional: ["mask"] } });
  assert.deepEqual(shownOptional(n, shape.inputs, (name) => name === "prompts").map((x) => x.name), ["mask", "prompts"]);
  assert.deepEqual(shownOptional(node(), shape.inputs, () => false), []);
});

test("addon choices follow the slots, the FITS list and a single slot already filled", () => {
  assert.deepEqual(addonChoices(info, kinds, []).map((k) => k.kind), ["Ladder", "Warm"]);
  assert.deepEqual(addonChoices(info, kinds, [{ kind: "Ladder" }]).map((k) => k.kind), ["Warm"]);
  assert.deepEqual(addonChoices(info, kinds, [{ kind: "Warm" }]).map((k) => k.kind), ["Ladder", "Warm"], "a many slot keeps offering");
});

test("tabs appear only when they have something; the remembered tab falls back to Main", () => {
  const bare = { inputs: [p("model", { required: true })] };
  assert.deepEqual(tabsOf({ shape: bare, info: { class_name: "X", addon_slots: [] }, node: node(), kinds }).map((t) => t.id), ["main"]);
  const eventsOnly = new Map([...kinds, ["Tick", { kind: "Tick", slot: "events", fits: [] }]]);
  const plain = { class_name: "X", addon_slots: [{ name: "events", many: true }] };
  assert.deepEqual(tabsOf({ shape: bare, info: plain, node: node(), kinds: eventsOnly }).map((t) => t.id), ["main"], "event addons alone add no tab");
  const tabs = tabsOf({ shape, info, node: node({ addons: [{ kind: "Ladder" }] }), kinds });
  assert.deepEqual(tabs.map((t) => [t.id, t.count]), [["main", 0], ["optional", 0], ["addons", 1]]);
  assert.equal(activeTab(tabs, node({ layout: { tab: "addons" } })), "addons");
  assert.equal(activeTab(tabs.slice(0, 1), node({ layout: { tab: "addons" } })), "main");
});

test("the hub opens the tab that holds a hidden port", () => {
  assert.equal(tabOfHidden("ladder.start", shape, node()), "addons");
  assert.equal(tabOfHidden("slot:ladder", shape, node()), "addons");
  assert.equal(tabOfHidden("mask", shape, node()), "optional");
  assert.equal(tabOfHidden("lr", shape, node()), "main");
});
