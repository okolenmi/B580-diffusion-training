/* The balanced layout (packing.js): pictures of different shapes get about the same area, nothing overlaps, and
   adding pictures never moves the ones already placed. */
import test from "node:test";
import assert from "node:assert/strict";
import { ColumnPacker, pack, readingOrder } from "../js/datasets/packing.js";
import { rowsLayout } from "../js/datasets/layout.js";

/** Pictures like a mixed dataset: tall, square and wide, with varied proportions. */
function sample(count, seed = 7) {
  let s = seed;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  return Array.from({ length: count }, () => {
    const u = rnd();
    return u < 0.45 ? 0.5 + rnd() * 0.4 : u < 0.55 ? 0.9 + rnd() * 0.2 : 1.2 + rnd() * 0.8;
  });
}
const areas = (boxes) => boxes.map((b) => b.w * b.h).sort((a, b) => a - b);
const quantile = (sorted, q) => sorted[Math.floor(q * (sorted.length - 1))];
const apart = (a, b) => a.x + a.w <= b.x + 1e-6 || b.x + b.w <= a.x + 1e-6 || a.y + a.h <= b.y + 1e-6 || b.y + b.h <= a.y + 1e-6;

test("every picture keeps its exact proportions (no gap)", () => {
  const ratios = sample(80);
  pack(ratios, 1200, 180, 0).boxes.forEach((b, n) => assert.ok(Math.abs(b.w / b.h - ratios[n]) < 1e-9, `picture ${n}`));
});

test("pictures stay inside the width and never overlap", () => {
  const { boxes, height } = pack(sample(120), 1200, 180, 6);
  for (const b of boxes) assert.ok(b.x >= -1e-6 && b.x + b.w <= 1200 + 1e-6 && b.y >= -1e-6 && b.y + b.h <= height + 1e-6);
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) assert.ok(apart(boxes[i], boxes[j]), `${i} and ${j} overlap`);
});

test("tall and wide pictures come out at similar sizes: the 5th-95th percentile spread stays under 1.8x, far below rows", () => {
  const ratios = sample(500);
  for (const width of [900, 1300]) {
    for (const size of [150, 180, 220]) {
      const packed = areas(pack(ratios, width, size, 6).boxes);
      const rows = areas(rowsLayout(ratios, width, size, 6).boxes);
      const spread = quantile(packed, 0.95) / quantile(packed, 0.05);
      assert.ok(spread < 1.8, `${width}/${size}: spread ${spread.toFixed(2)}`);
      assert.ok(spread < 0.7 * (quantile(rows, 0.95) / quantile(rows, 0.05)), `${width}/${size}: not better than rows`);
    }
  }
});

test("little is left empty: the packed height is close to what the picture areas need", () => {
  const ratios = sample(500);
  const { boxes, height } = pack(ratios, 1200, 180, 0);
  const used = boxes.reduce((sum, b) => sum + b.w * b.h, 0);
  assert.ok(used / (1200 * height) > 0.88, `filled ${(used / (1200 * height)).toFixed(2)}`);
});

test("pictures added later never move the earlier ones", () => {
  const ratios = sample(200);
  const packer = new ColumnPacker(1206, 180, ratios.slice(0, 60));
  const first = ratios.slice(0, 60).map((r) => packer.place(r));
  ratios.slice(60).forEach((r) => packer.place(r));
  const again = new ColumnPacker(1206, 180, ratios.slice(0, 60));
  assert.deepEqual(ratios.slice(0, 60).map((r) => again.place(r)), first);
});

test("wide pictures get wider columns than tall ones", () => {
  const ratios = sample(300);
  const { boxes } = pack(ratios, 1300, 180, 0);
  const mean = (list) => list.reduce((a, b) => a + b, 0) / list.length;
  const tall = mean(boxes.filter((_, n) => ratios[n] < 0.8).map((b) => b.w));
  const wide = mean(boxes.filter((_, n) => ratios[n] > 1.4).map((b) => b.w));
  assert.ok(wide > 1.3 * tall);
});

test("reading order is a permutation that goes top to bottom", () => {
  const ratios = sample(100);
  const { boxes, order } = pack(ratios, 1200, 180, 6);
  assert.deepEqual([...order].sort((a, b) => a - b), ratios.map((_, n) => n));
  assert.deepEqual(readingOrder(boxes, 180), order);
  const tops = order.map((n) => boxes[n].y);
  for (let k = 1; k < tops.length; k++) assert.ok(tops[k] >= tops[k - 1] - 90 - 1e-6);
});

test("deterministic, taller when narrower, and fine with one picture or none", () => {
  const ratios = sample(80);
  assert.deepEqual(pack(ratios, 1200, 180, 6), pack(ratios, 1200, 180, 6));
  assert.ok(pack(ratios, 600, 180, 6).height > pack(ratios, 1200, 180, 6).height);
  assert.equal(pack([1.5], 800, 180, 6).boxes.length, 1);
  assert.deepEqual(pack([], 800, 180, 6).boxes, []);
});

test("rows as boxes: every picture placed, in order, inside the width", () => {
  const ratios = sample(50);
  const { boxes, order, height } = rowsLayout(ratios, 1000, 180, 6);
  assert.deepEqual(order, ratios.map((_, n) => n));
  for (const b of boxes) assert.ok(b.x + b.w <= 1000 && b.y + b.h <= height + 1);
});
