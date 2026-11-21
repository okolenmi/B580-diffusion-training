/* Unit tests for sections and tabs in board/model.js and the undo history: nesting rules, ids across the document, moving cards
   between tabs, columns, what Run sends from nested cards. Pure; no DOM. */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  addContainerBlock, addNodeSection, addNoteBlock, addTab, collectIds, containerKey, depthOf, layoutKey, leafCards,
  locateCard, moveCard, moveTab, moveTargets, nodeRef, removeTab, renameTab, valuesPayload, walkCards, withColumns,
} from "../js/board/model.js";
import { History } from "../js/board/history.js";

const card = (id, name, extra = {}) => ({
  id, ref: nodeRef("n1", "field", name), x: 0, y: 0, w: 4, h: 2, collapsed: false, title: null, options: {}, ...extra,
});
const section = (id, tabs, extra = {}) => ({
  id, container: { columns: 6, tabs }, x: 0, y: 0, w: 6, h: 4, collapsed: false, title: null, options: {}, ...extra,
});
const tab = (id, title, blocks = []) => ({ id, title, blocks });

const page = () => ({
  columns: 12,
  tabs: [
    tab("main", "Main", [
      card("a", "steps"),
      section("s1", [tab("t1", "Basics", [card("b", "lr")]), tab("t2", "More", [section("s2", [tab("t1", "Deep", [card("c", "seed")])])])]),
    ]),
    tab("second", "Second", []),
  ],
});
const all = (root) => root.tabs.flatMap((t) => t.blocks);

test("walkCards visits every card at any depth, parents first; leafCards only the bound ones", () => {
  const blocks = all(page());
  assert.deepEqual([...walkCards(blocks)].map((b) => b.id), ["a", "s1", "b", "s2", "c"]);
  assert.deepEqual([...leafCards(blocks)].map((b) => b.id), ["a", "b", "c"]);
  assert.deepEqual([...collectIds(blocks)].sort(), ["a", "b", "c", "s1", "s2"]);
  assert.equal(depthOf(blocks), 2);
});

test("new cards get ids unused anywhere in the document, not just on their own tab", () => {
  const root = page();
  const doc = { columns: 12, blocks: root.tabs[1].blocks, graph: null };
  const used = collectIds(all(root));
  const next = addContainerBlock(doc, used);
  assert.ok(!used.has(next.blocks[0].id));
  const note = addNoteBlock(next, new Set([...used, next.blocks[0].id]));
  assert.ok(!used.has(note.blocks[1].id) && note.blocks[1].id !== next.blocks[0].id);
});

test("a heading is a full-width one-line note without a frame", () => {
  const doc = addNoteBlock({ columns: 12, blocks: [] }, null, "heading");
  const [h] = doc.blocks;
  assert.deepEqual([h.w, h.h, h.style, h.options.look], [12, 1, "bare", "heading"]);
});

test("a node section holds the node's free, non-advanced fields and skips ones already on the page", () => {
  const modules = {
    nodes: [{
      id: "n1", display_name: "Trainer", known: true, displays: [],
      fields: [
        { name: "steps", type: "int" }, { name: "lr", type: "float" },
        { name: "fed", type: "int", wired: true }, { name: "rank", type: "int", advanced: true },
      ],
    }],
  };
  const doc = { columns: 12, blocks: [card("b1", "lr")] };
  const next = addNodeSection(doc, "n1", modules);
  const made = next.blocks.find((b) => b.container);
  assert.equal(made.title, "Trainer");
  assert.deepEqual(made.container.tabs[0].blocks.map((b) => b.ref.name), ["steps"]);
  assert.equal(addNodeSection(next, "n1", modules), next, "nothing left to add leaves the doc alone");
});

test("tab operations keep at least one tab, cap at twelve and ignore empty titles", () => {
  let tabs = [tab("t1", "A")];
  tabs = addTab(tabs);
  assert.deepEqual(tabs.map((t) => t.id), ["t1", "t2"]);
  assert.equal(removeTab([tab("t1", "A")], "t1").length, 1);
  assert.deepEqual(moveTab(tabs, "t2", -1).map((t) => t.id), ["t2", "t1"]);
  assert.equal(moveTab(tabs, "t1", -1), tabs);
  assert.equal(renameTab(tabs, "t1", "  ").length, 2);
  assert.equal(renameTab(tabs, "t1", " New ")[0].title, "New");
  let many = [tab("t1", "A")];
  for (let i = 0; i < 20; i++) many = addTab(many);
  assert.equal(many.length, 12);
  assert.equal(new Set(many.map((t) => t.id)).size, 12);
});

test("moving a card: any tab but its own subtree, and never deeper than three containers", () => {
  const root = page();
  const forS1 = moveTargets(root, "s1").map((t) => `${t.owner}/${t.tabId}`);
  assert.ok(forS1.includes("/main") && forS1.includes("/second"));
  assert.ok(!forS1.some((k) => k.startsWith("s1/") || k.startsWith("s2/")), "not into itself");

  const moved = moveCard(root, "b", "", "second");
  assert.deepEqual(locateCard(moved, "b"), { owner: "", tabId: "second" });
  assert.deepEqual(locateCard(root, "b"), { owner: "s1", tabId: "t1" }, "the original is untouched");
  assert.deepEqual([...collectIds(all(moved))].sort(), [...collectIds(all(root))].sort());

  assert.equal(moveCard(root, "s1", "s2", "t1"), root, "a section cannot enter its own descendant");

  // s1 holds a section (height 2 with its own child chain); put it at depth 3 and nothing may go under it.
  const deep = { columns: 12, tabs: [tab("m", "M", [
    section("x1", [tab("t1", "1", [section("x2", [tab("t1", "2", [section("x3", [tab("t1", "3", [])])])])])]),
    section("y", [tab("t1", "Y", [card("z", "lr")])]),
  ])] };
  assert.equal(moveCard(deep, "y", "x3", "t1"), deep, "y would sit four containers deep");
  assert.notEqual(moveCard(deep, "z", "x3", "t1"), deep, "a plain card still fits at depth three");
});

test("a moved card narrows to its new grid and lands without collisions", () => {
  const root = { columns: 12, tabs: [tab("m", "M", [card("a", "steps", { w: 10 }), section("s", [tab("t1", "T", [card("b", "lr", { w: 6 })])], { y: 2 })])] };
  const moved = moveCard(root, "a", "s", "t1");
  const inner = moved.tabs[0].blocks[0].container.tabs[0].blocks;
  const a = inner.find((b) => b.id === "a");
  assert.equal(a.w, 6);
  const b = inner.find((b) => b.id === "b");
  assert.ok(a.y + a.h <= b.y || b.y + b.h <= a.y, "no overlap");
});

test("withColumns narrows every tab's cards and keeps them on the grid", () => {
  const c = withColumns({ columns: 12, tabs: [tab("t", "T", [card("a", "steps", { x: 8, w: 4 }), card("b", "lr", { x: 0, w: 12, y: 2 })])] }, 6);
  assert.equal(c.columns, 6);
  for (const b of c.tabs[0].blocks) assert.ok(b.x + b.w <= 6);
});

test("Run sends the fields of nested cards, and the layout key sees every nested change", () => {
  const root = page();
  const doc = { blocks: all(root), graph: { nodes: [{ id: "n1", params: { steps: 5, lr: 0.1 } }], memory: {} } };
  assert.deepEqual(valuesPayload(doc).nodes.n1, { steps: 5, lr: 0.1, seed: null });
  assert.deepEqual(valuesPayload(doc, { c: "missing_module" }).nodes.n1, { steps: 5, lr: 0.1 });

  const before = containerKey(root);
  moveCard(root, "b", "", "second");
  assert.equal(containerKey(root), before);
  const edited = structuredClone(root);
  edited.tabs[0].blocks[1].container.tabs[1].blocks[0].container.tabs[0].title = "Renamed";
  assert.notEqual(containerKey(edited), before);
  edited.tabs[0].blocks[1].container.tabs[1].blocks[0].container.tabs[0].title = "Deep";
  edited.tabs[0].blocks[1].style = "accent";
  assert.notEqual(containerKey(edited), before);
  assert.notEqual(layoutKey({ columns: 12, blocks: all(edited) }), layoutKey({ columns: 12, blocks: all(root) }));
});

test("History: undo/redo, dropping the redo tail, merging a typing run, capping", () => {
  const h = new History(3);
  h.reset("a");
  assert.equal(h.record("a"), false, "same snapshot is not a step");
  h.record("b");
  h.record("c", { merge: true });
  h.record("cc", { merge: true });
  assert.equal(h.undo(), "b", "the typing run was one step");
  assert.equal(h.redo(), "cc");
  assert.equal(h.redo(), null);
  h.undo();
  h.record("d");
  assert.equal(h.canRedo, false);
  h.record("e");
  h.record("f");
  assert.equal(h.steps.length, 3);
  assert.equal(h.undo(), "e");
  assert.equal(h.undo(), "d");
  assert.equal(h.undo(), null);
});

/* ---- grouping cards into one box ---- */

import { groupCards as group } from "../js/board/model.js";
import { collectIds as idsOf, depthOf as deep } from "../js/board/model.js";

const leaf = (id, x, y, w = 4, h = 2) => ({ id, ref: { scope: "node", kind: "field", name: id, node: "n" }, x, y, w, h, collapsed: false, title: null, options: {} });

test("grouping puts the cards in one new box, as they were arranged, and keeps every id", () => {
  const root = { columns: 12, tabs: [{ id: "p", title: "Run", blocks: [leaf("a", 2, 1), leaf("b", 6, 1), leaf("c", 0, 6)] }] };
  const out = group(root, ["a", "b"]);
  const [c, box] = [out.tabs[0].blocks.find((b) => b.id === "c"), out.tabs[0].blocks.find((b) => b.container)];
  assert.ok(c && box, "the other card stays, the box is new");
  assert.deepEqual([...idsOf(out.tabs[0].blocks)].sort(), [...idsOf(root.tabs[0].blocks), box.id].sort());
  const inner = box.container.tabs[0].blocks;
  assert.deepEqual(inner.map((b) => [b.id, b.x, b.y]), [["a", 0, 0], ["b", 4, 0]]);
  assert.equal(box.container.columns, 8);
  assert.equal(box.x, 2);
  assert.equal(box.container.tabs.length, 1);
  assert.equal(root.tabs[0].blocks.length, 3, "the input is not changed");
});

test("grouping inside a box works, refuses cards from different places and too deep a nest", () => {
  const root = { columns: 12, tabs: [{ id: "p", title: "Run", blocks: [leaf("a", 0, 0), leaf("b", 4, 0)] }] };
  const once = group(root, ["a"]);
  const box = once.tabs[0].blocks.find((b) => b.container);
  const twice = group(once, ["a", "b"].filter((id) => id !== "b"));
  assert.equal(deep(twice.tabs[0].blocks), 2);
  assert.equal(group(once, ["a", "b"]), once, "a card inside a box and one outside are not one group");
  assert.equal(group(root, ["zz"]), root, "an unknown card changes nothing");
  assert.equal(group(root, []), root);
  assert.ok(box.id);
  let deepest = root;
  for (let i = 0; i < 3; i++) deepest = group(deepest, [deepest.tabs[0].blocks.find((b) => b.ref || b.container).id]);
  assert.ok(deep(deepest.tabs[0].blocks) <= 3);
});
