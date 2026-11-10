import test from "node:test";
import assert from "node:assert/strict";
import { justify } from "../js/datasets/layout.js";

const widthOf = (rows, ratios, gap) => rows.map((r) => {
  let w = gap * (r.count - 1);
  for (let i = r.start; i < r.start + r.count; i++) w += ratios[i] * r.height;
  return w;
});

test("every full row fills the width exactly and keeps each picture's proportions", () => {
  const ratios = [0.67, 1.5, 1, 0.56, 1.78, 1, 0.67, 1.5, 1, 1, 0.75, 1.33];
  const rows = justify(ratios, 1000, 180, 6);
  assert.ok(rows.length > 1);
  for (const w of widthOf(rows.slice(0, -1), ratios, 6)) assert.ok(Math.abs(w - 1000) < 1e-6, `row width ${w}`);
});

test("rows cover every picture once, in order", () => {
  const ratios = Array.from({ length: 37 }, (_, i) => 0.6 + (i % 5) * 0.4);
  const rows = justify(ratios, 900, 160, 6);
  let next = 0;
  for (const r of rows) { assert.equal(r.start, next); assert.ok(r.count >= 1); next += r.count; }
  assert.equal(next, ratios.length);
});

test("row heights stay near the target", () => {
  const ratios = Array.from({ length: 60 }, (_, i) => [0.67, 1.5, 1, 0.56, 1.78][i % 5]);
  for (const r of justify(ratios, 1100, 200, 6).slice(0, -1)) assert.ok(r.height > 120 && r.height < 280, `height ${r.height}`);
});

test("a short last row is not stretched past the target height", () => {
  const rows = justify([1, 1], 1000, 180, 6);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].height, 180);
});

test("a very wide picture gets a row of its own at a sensible height", () => {
  const rows = justify([3, 1, 1, 1], 800, 180, 6);
  assert.equal(rows[0].count >= 1, true);
  assert.ok(rows[0].height <= 270 + 1e-6);
});

test("nothing to show, nothing to lay out", () => {
  assert.deepEqual(justify([], 800, 180, 6), []);
});
