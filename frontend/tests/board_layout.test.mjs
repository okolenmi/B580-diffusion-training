/* Unit tests for board/layout.js -- the grid arithmetic. Pure, so every rule
   here is checked by number, not by looking at a page.
   Run: node --test frontend/tests */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  addAt, cellDelta, clamp, collides, columnWidth, effH, findFreeSlot, flowOrder,
  keyNudge, moveTo, resizeTo, resolve, setCollapsed, totalRows,
} from "../js/board/layout.js";

const C = 12;
const b = (id, x, y, w, h, extra = {}) => ({ id, x, y, w, h, ...extra });
const by = (blocks, id) => blocks.find((q) => q.id === id);
function assertNoOverlap(blocks) {
  for (let i = 0; i < blocks.length; i++)
    for (let j = i + 1; j < blocks.length; j++)
      assert.ok(!collides(blocks[i], blocks[j]), `${blocks[i].id} overlaps ${blocks[j].id}`);
}

test("touching edges are not a collision; overlap is", () => {
  assert.ok(!collides(b("a", 0, 0, 6, 2), b("b", 6, 0, 6, 2)));
  assert.ok(!collides(b("a", 0, 0, 6, 2), b("b", 0, 2, 6, 2)));
  assert.ok(collides(b("a", 0, 0, 6, 2), b("b", 5, 1, 6, 2)));
});

test("a collapsed block is one row but keeps its stored height", () => {
  const x = b("a", 0, 0, 6, 5, { collapsed: true });
  assert.equal(effH(x), 1);
  assert.equal(x.h, 5);
  assert.ok(!collides(x, b("c", 0, 1, 6, 2)), "the row it gave up is usable");
});

test("clamp: integer, inside the grid, never below 1 cell; garbage does not throw", () => {
  assert.deepEqual(clamp(b("a", 10.4, -3, 6, 0), C), b("a", 6, 0, 6, 1));
  assert.deepEqual(clamp(b("a", 0, 0, 99, 2), C), b("a", 0, 0, 12, 2));
  const weird = clamp({ id: "z", x: NaN, y: undefined, w: "x", h: null }, C);
  assert.ok([weird.x, weird.y, weird.w, weird.h].every(Number.isInteger));
});

test("resolve: settles upward -- holes close, order is kept", () => {
  const out = resolve([b("a", 0, 5, 12, 2), b("b", 0, 9, 12, 2)], C);
  assert.equal(by(out, "a").y, 0);
  assert.equal(by(out, "b").y, 2);
});

test("resolve: side-by-side blocks both stay on row 0", () => {
  const out = resolve([b("a", 0, 3, 6, 2), b("b", 6, 7, 6, 2)], C);
  assert.equal(by(out, "a").y, 0);
  assert.equal(by(out, "b").y, 0);
});

test("resolve: overlapping input is separated, nothing overlaps afterwards", () => {
  const out = resolve([b("a", 0, 0, 8, 3), b("b", 2, 1, 8, 3), b("c", 4, 2, 8, 3)], C);
  assertNoOverlap(out);
});

test("resolve is deterministic and idempotent", () => {
  const input = [b("c", 3, 0, 6, 2), b("a", 0, 0, 6, 2), b("b", 0, 0, 6, 2)];
  const once = resolve(input, C);
  assert.deepEqual(resolve(input, C), once);
  assert.deepEqual(resolve(once, C), once);
});

test("resolve keeps the input's order of blocks (callers index into it)", () => {
  const input = [b("z", 0, 4, 6, 1), b("a", 0, 0, 6, 1)];
  assert.deepEqual(resolve(input, C).map((q) => q.id), ["z", "a"]);
});

test("moveTo: the held block lands where it was dropped; others make way below", () => {
  const start = resolve([b("a", 0, 0, 6, 2), b("b", 6, 0, 6, 2), b("c", 0, 2, 12, 2)], C);
  const out = moveTo(start, "c", 0, 0, C);
  assert.equal(by(out, "c").y, 0);
  assert.equal(by(out, "a").y, 4 - 2 + 0, "a pushed under c (c is 2 rows)");
  assertNoOverlap(out);
});

test("moveTo clamps to the grid: cannot be dragged off the right edge or above the top", () => {
  const start = [b("a", 0, 0, 6, 2)];
  const out = moveTo(start, "a", 50, -4, C);
  assert.equal(by(out, "a").x, 6);
  assert.equal(by(out, "a").y, 0);
});

test("resizeTo: growing pushes the neighbour below; shrinking lets it come back up", () => {
  const start = resolve([b("a", 0, 0, 12, 2), b("b", 0, 2, 12, 2)], C);
  const grown = resizeTo(start, "a", 12, 5, C);
  assert.equal(by(grown, "b").y, 5);
  const shrunk = resolve(resizeTo(grown, "a", 12, 2, C), C);
  assert.equal(by(shrunk, "b").y, 2);
});

test("resizeTo clamps width to the grid and height to at least one row", () => {
  const out = resizeTo([b("a", 4, 0, 4, 2)], "a", 40, 0, C);
  assert.equal(by(out, "a").w, 12);
  assert.equal(by(out, "a").x, 0, "a block that cannot fit where it is moves left");
  assert.equal(by(out, "a").h, 1);
});

test("collapsing frees the rows under the header, expanding takes them back", () => {
  const start = resolve([b("a", 0, 0, 12, 4), b("b", 0, 4, 12, 2)], C);
  const folded = setCollapsed(start, "a", true, C);
  assert.equal(by(folded, "b").y, 1);
  assert.equal(by(folded, "a").h, 4, "stored height survives");
  const open = setCollapsed(folded, "a", false, C);
  assert.equal(by(open, "b").y, 4);
});

test("findFreeSlot: fills a gap before using the bottom", () => {
  const blocks = [b("a", 0, 0, 6, 2), b("b", 0, 2, 12, 2)];
  assert.deepEqual(findFreeSlot(blocks, 6, 2, C), { x: 6, y: 0 });
  assert.deepEqual(findFreeSlot(blocks, 8, 2, C), { x: 0, y: 4 });
});

test("findFreeSlot on an empty page is the origin", () => {
  assert.deepEqual(findFreeSlot([], 6, 3, C), { x: 0, y: 0 });
});

test("addAt places, keeps existing blocks, never overlaps", () => {
  let blocks = [];
  for (let i = 0; i < 9; i++) blocks = addAt(blocks, b(`n${i}`, 0, 0, 5, 3), C);
  assert.equal(blocks.length, 9);
  assertNoOverlap(blocks);
  assert.ok(blocks.every((q) => q.x + q.w <= C));
});

test("flowOrder is reading order (y, x, id)", () => {
  const order = flowOrder([b("c", 6, 2, 6, 1), b("a", 0, 0, 6, 1), b("b", 6, 0, 6, 1)]).map((q) => q.id);
  assert.deepEqual(order, ["a", "b", "c"]);
});

test("totalRows counts collapsed blocks as one row", () => {
  assert.equal(totalRows([b("a", 0, 0, 12, 6, { collapsed: true }), b("b", 0, 1, 12, 2)]), 3);
});

test("cellDelta and columnWidth: pixels to cells", () => {
  const colW = columnWidth(1180, 12, 8); // (1180 - 88) / 12 = 91
  assert.equal(colW, 91);
  assert.deepEqual(cellDelta(99 * 2 + 10, -40, colW, 40, 8), { dCols: 2, dRows: -1 });
  assert.deepEqual(cellDelta(10, 10, colW, 40, 8), { dCols: 0, dRows: 0 });
});

test("keyNudge: arrows move one cell, shift+arrows resize, other keys do nothing", () => {
  const start = resolve([b("a", 2, 2, 4, 2)], C);
  assert.equal(by(keyNudge(start, "a", "ArrowRight", false, C), "a").x, 3);
  assert.equal(by(keyNudge(start, "a", "ArrowDown", true, C), "a").h, 3);
  assert.equal(by(keyNudge(start, "a", "ArrowLeft", true, C), "a").w, 3);
  assert.equal(keyNudge(start, "a", "x", false, C), start);
  assert.equal(keyNudge(start, "missing", "ArrowUp", false, C), start);
});

test("a randomized stress: any sequence of moves and resizes stays valid", () => {
  let seed = 7;
  const rnd = (n) => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % n; };
  let blocks = [];
  for (let i = 0; i < 12; i++) blocks = addAt(blocks, b(`n${i}`, 0, 0, 1 + rnd(8), 1 + rnd(4)), C);
  for (let step = 0; step < 300; step++) {
    const id = `n${rnd(12)}`;
    blocks = rnd(3) === 0
      ? resizeTo(blocks, id, 1 + rnd(14), 1 + rnd(6), C)
      : rnd(2) === 0
        ? moveTo(blocks, id, rnd(14) - 1, rnd(20) - 2, C)
        : setCollapsed(blocks, id, rnd(2) === 0, C);
    assertNoOverlap(blocks);
    for (const q of blocks) {
      assert.ok(q.x >= 0 && q.y >= 0 && q.w >= 1 && q.h >= 1 && q.x + q.w <= C, `bad block ${JSON.stringify(q)}`);
    }
  }
  assert.deepEqual(resolve(resolve(blocks, C), C), resolve(blocks, C), "settled layouts are stable");
});
