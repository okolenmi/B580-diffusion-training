/* graph2 layout: nodes without positions get columns by wire depth and no overlaps. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { arrange, isUnplaced } from "../js/graph2/layout.js";

test("a chain runs left to right and a fan-in lands after its last input", () => {
  const out = arrange(["a", "b", "c", "d"], [["a", "b"], ["b", "d"], ["c", "d"]], () => 100, (u) => (u === "b" ? 500 : 200));
  assert.deepEqual([out.get("a").x, out.get("b").x, out.get("c").x, out.get("d").x], [0, 260, 0, 820]);
});

test("nodes of one column never overlap", () => {
  const out = arrange(["a", "b", "c"], [], (u) => (u === "a" ? 300 : 100));
  const ys = ["a", "b", "c"].map((u) => out.get(u).y);
  assert.deepEqual(ys, [0, 340, 480]);
});

test("a loop of wires terminates and every node is placed", () => {
  const out = arrange(["a", "b"], [["a", "b"], ["b", "a"]]);
  assert.equal(out.size, 2);
});

test("only a document with nothing placed is arranged", () => {
  assert.equal(isUnplaced([{ layout: { x: 0, y: 0 } }, { layout: {} }]), true);
  assert.equal(isUnplaced([{ layout: { x: 0, y: 0 } }, { layout: { x: 5, y: 0 } }]), false);
  assert.equal(isUnplaced([{ layout: { x: 0, y: 0 } }]), false);
});
