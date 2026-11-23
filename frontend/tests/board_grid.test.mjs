/* Unit tests for board/grid.js and the grid-aware parts of the document rules: refining a coarse page, sizes on the fine grid,
   boxing and moving cards between containers on it. Pure. */

import assert from "node:assert/strict";
import test from "node:test";

import { addBlock, addNoteBlock, defaultSize, FINE, gridMetrics, groupCards, moveCard, refineContainer } from "../js/board/model.js";

const card = (id, x, y, w, h) => ({ id, ref: { scope: "note", kind: "note", name: id }, x, y, w, h, collapsed: false, title: null, options: {} });
const page = (blocks, extra = {}) => ({ columns: 12, tabs: [{ id: "main", title: "Main", blocks }], ...extra });

test("a coarse page is refined once: columns doubled, rows doubled, boxes inside too, nothing else moves", () => {
  const box = { id: "box", x: 0, y: 2, w: 6, h: 4, container: page([card("c", 1, 1, 3, 2)], { columns: 6 }) };
  const fine = refineContainer(page([card("a", 3, 1, 4, 2), box]));
  assert.equal(fine.columns, 24);
  assert.equal(fine.grid, FINE);
  assert.deepEqual(fine.tabs[0].blocks[0], { ...card("a", 6, 2, 8, 4) });
  const inner = fine.tabs[0].blocks[1];
  assert.deepEqual([inner.x, inner.y, inner.w, inner.h], [0, 4, 12, 8]);
  assert.equal(inner.container.columns, 12);
  assert.deepEqual(inner.container.tabs[0].blocks[0], card("c", 2, 2, 6, 4));
  assert.equal(refineContainer(fine), fine);
});

test("a page already wider than half the limit keeps its columns but its rows are still refined", () => {
  const fine = refineContainer(page([card("a", 2, 1, 10, 3)], { columns: 16 }));
  assert.equal(fine.columns, 16);
  assert.deepEqual([fine.tabs[0].blocks[0].x, fine.tabs[0].blocks[0].w, fine.tabs[0].blocks[0].y, fine.tabs[0].blocks[0].h], [2, 10, 2, 6]);
});

test("two fine rows and their gap are as tall as one coarse row", () => {
  const coarse = gridMetrics(1), fine = gridMetrics(FINE);
  assert.equal(2 * fine.rowPx + fine.gapPx, coarse.rowPx);
});

test("new cards on the fine grid come out in its units: sizes scaled, notes and boxes too", () => {
  assert.deepEqual(defaultSize({ scope: "graph", kind: "display", name: "x" }, null, 24, false, FINE), [12, 8]);
  assert.deepEqual(defaultSize({ scope: "graph", kind: "display", name: "x" }, null, 12, true, FINE), [12, 8]);
  const doc = { columns: 24, grid: FINE, blocks: [], graph: null };
  const note = addNoteBlock(doc).blocks[0];
  assert.deepEqual([note.w, note.h], [12, 4]);
  const head = addNoteBlock(doc, null, "heading").blocks[0];
  assert.deepEqual([head.w, head.h], [24, 2]);
});

test("a card dropped on a box goes where it was dropped, a box made on the fine grid is fine itself", () => {
  const root = refineContainer(page([card("a", 0, 0, 4, 2), { id: "box", x: 4, y: 0, w: 8, h: 4, container: page([], { columns: 12 }) }]));
  const moved = moveCard(root, "a", "box", "main", { x: 3, y: 5 });
  const inside = moved.tabs[0].blocks.find((b) => b.id === "box").container.tabs[0].blocks;
  assert.deepEqual(inside.map((b) => [b.id, b.x, b.y]), [["a", 3, 5]]);
  assert.equal(moved.tabs[0].blocks.some((b) => b.id === "a"), false);
  const grouped = groupCards(root, ["a"]);
  const box = grouped.tabs[0].blocks.find((b) => b.container && b.id !== "box");
  assert.equal(box.container.grid, FINE);
  assert.ok(box.container.columns >= 6);
});

test("dropping a card back where it already is, or into itself, changes nothing", () => {
  const root = refineContainer(page([card("a", 0, 0, 4, 2), { id: "box", x: 4, y: 0, w: 8, h: 4, container: page([], { columns: 12 }) }]));
  assert.equal(moveCard(root, "box", "box", "main", { x: 0, y: 0 }), root);
});

test("a field added to a box lands as a one-row field on the fine grid", () => {
  const modules = { nodes: [{ id: "n", fields: [{ name: "steps", type: "int" }], displays: [], actions: [] }], graph_fields: [] };
  const doc = { columns: 12, grid: FINE, blocks: [], graph: null };
  const block = addBlock(doc, { scope: "node", node: "n", kind: "field", name: "steps" }, modules, null, true).blocks[0];
  assert.equal(block.h, 2);
});
