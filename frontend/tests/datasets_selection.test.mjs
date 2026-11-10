import test from "node:test";
import assert from "node:assert/strict";
import { Selection } from "../js/datasets/selection.js";

const order = ["a", "b", "c", "d", "e"];

test("a plain click selects one image and makes it the anchor", () => {
  const s = new Selection();
  s.only("c");
  assert.deepEqual(s.list(), ["c"]);
  assert.equal(s.single, "c");
  s.only("d");
  assert.deepEqual(s.list(), ["d"]);
});

test("ctrl-click toggles one image without touching the others", () => {
  const s = new Selection();
  s.only("a");
  s.toggle("c");
  assert.deepEqual(s.list().sort(), ["a", "c"]);
  assert.equal(s.single, null);
  s.toggle("a");
  assert.deepEqual(s.list(), ["c"]);
});

test("shift-click selects the range from the anchor, in either direction", () => {
  const s = new Selection();
  s.only("b");
  s.range(order, "d");
  assert.deepEqual(s.list().sort(), ["b", "c", "d"]);
  const t = new Selection();
  t.only("d");
  t.range(order, "b");
  assert.deepEqual(t.list().sort(), ["b", "c", "d"]);
});

test("shift-click without an anchor, or on an unknown image, acts as a plain click", () => {
  const s = new Selection();
  s.range(order, "c");
  assert.deepEqual(s.list(), ["c"]);
  s.range(order, "zzz");
  assert.deepEqual(s.list(), ["c"]);
});

test("fill takes ids that are not loaded, clear empties and forgets the anchor", () => {
  const s = new Selection();
  s.fill(["x", "y", "z"]);
  assert.equal(s.size, 3);
  assert.ok(s.has("y"));
  s.clear();
  assert.equal(s.size, 0);
  s.range(order, "b");
  assert.deepEqual(s.list(), ["b"]);
});
