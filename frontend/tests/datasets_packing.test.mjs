import test from "node:test";
import assert from "node:assert/strict";
import { pack } from "../js/datasets/packing.js";
import { rowsLayout } from "../js/datasets/layout.js";

/** Pictures like a character dataset: mostly tall, some wide, a few square. */
function sample(count, seed = 7) {
  let s = seed;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  return Array.from({ length: count }, () => {
    const u = rnd();
    return u < 0.7 ? 0.5 + rnd() * 0.3 : u < 0.93 ? 1.3 + rnd() * 0.5 : 0.95 + rnd() * 0.1;
  });
}
const spread = (areas) => {
  const logs = areas.map(Math.log);
  const mean = logs.reduce((a, b) => a + b) / logs.length;
  return Math.sqrt(logs.reduce((a, b) => a + (b - mean) ** 2, 0) / logs.length);
};

test("every picture keeps its exact proportions (no gap, no crop)", () => {
  const ratios = sample(60);
  const { boxes } = pack(ratios, 1200, 180, 0);
  boxes.forEach((b, n) => assert.ok(Math.abs(b.w / b.h - ratios[n]) < 1e-9, `picture ${n}`));
});

test("pictures fill the width and never overlap", () => {
  const ratios = sample(90);
  const { boxes, height } = pack(ratios, 1200, 180, 0);
  for (const b of boxes) assert.ok(b.x >= -1e-6 && b.x + b.w <= 1200 + 1e-6 && b.y >= -1e-6 && b.y + b.h <= height + 1e-6);
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i], b = boxes[j];
      const apart = a.x + a.w <= b.x + 1e-6 || b.x + b.w <= a.x + 1e-6 || a.y + a.h <= b.y + 1e-6 || b.y + b.h <= a.y + 1e-6;
      assert.ok(apart, `pictures ${i} and ${j} overlap`);
    }
  }
});

test("a gap leaves that much between neighbours and the outer edge stays flush", () => {
  const { boxes } = pack(sample(40), 1000, 180, 8);
  assert.ok(Math.min(...boxes.map((b) => b.x)) >= -1e-6);
  assert.ok(Math.max(...boxes.map((b) => b.x + b.w)) <= 1000 + 1e-6);
});

test("tall and wide pictures come out close to the same area, much closer than in rows", () => {
  const ratios = sample(400);
  for (const size of [150, 180, 260]) {
    const packed = pack(ratios, 1400, size, 6).boxes.map((b) => b.w * b.h);
    const rows = rowsLayout(ratios, 1400, size, 6).boxes.map((b) => b.w * b.h);
    assert.ok(spread(packed) < 0.6 * spread(rows), `size ${size}: packed ${spread(packed).toFixed(2)} vs rows ${spread(rows).toFixed(2)}`);
  }
});

test("order is a permutation, and pictures of one shape keep their relative order", () => {
  const ratios = sample(120);
  const { order } = pack(ratios, 1200, 180, 6);
  assert.deepEqual([...order].sort((a, b) => a - b), ratios.map((_, n) => n));
  for (const same of [(r) => r < 0.85, (r) => r >= 0.85 && r < 1.18, (r) => r >= 1.18]) {
    const seq = order.filter((n) => same(ratios[n]));
    assert.deepEqual(seq, [...seq].sort((a, b) => a - b));
  }
});

test("while more pictures are coming, an unfinished block waits instead of being stretched", () => {
  const ratios = sample(100);
  const waiting = pack(ratios, 1200, 180, 6, false);
  const done = pack(ratios, 1200, 180, 6, true);
  assert.ok(waiting.boxes.some((b) => b === null));
  assert.equal(waiting.order.length, waiting.boxes.filter(Boolean).length);
  assert.ok(done.boxes.every((b) => b !== null));
  assert.equal(done.order.length, ratios.length);
});

test("the same input gives the same layout, and a narrower space gives a taller one", () => {
  const ratios = sample(80);
  assert.deepEqual(pack(ratios, 1200, 180, 6), pack(ratios, 1200, 180, 6));
  assert.ok(pack(ratios, 600, 180, 6).height > pack(ratios, 1200, 180, 6).height);
});

test("one picture, and none, are fine", () => {
  assert.equal(pack([1.5], 800, 180, 6).boxes.length, 1);
  assert.deepEqual(pack([], 800, 180, 6).boxes, []);
});

test("rows as boxes: every picture placed, in order, inside the width", () => {
  const ratios = sample(50);
  const { boxes, order, height } = rowsLayout(ratios, 1000, 180, 6);
  assert.deepEqual(order, ratios.map((_, n) => n));
  for (const b of boxes) assert.ok(b.x + b.w <= 1000 && b.y + b.h <= height + 1);
});
