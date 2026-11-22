/* Unit tests for lib/chart_format.js -- how the charts label axes and average a window. Pure; no DOM. */

import assert from "node:assert/strict";
import test from "node:test";

import { averageOf, formatAxisLabel, formatLinearTick, formatMovement, formatNum, linearTicks, niceNum } from "../js/lib/chart_format.js";

test("numbers keep the precision a reader needs at their magnitude", () => {
  assert.equal(formatNum(9123.4568), "9123.5");
  assert.equal(formatNum(2.5), "2.5000");
  assert.equal(formatNum(0.05), "0.05000");
  assert.equal(formatNum(0.0004), "4.00e-4");
});

test("axis labels carry two significant digits and a sign", () => {
  assert.equal(formatAxisLabel(0), "0");
  assert.equal(formatAxisLabel(-0.123), "-0.12");
  assert.equal(formatAxisLabel(12345), "1.2e+4");
});

test("linear ticks sit on a nice step and print exactly", () => {
  const ticks = linearTicks(8000, 9000);
  assert.ok(ticks.length >= 3 && ticks.every((t) => t >= 8000 && t <= 9000.0001));
  assert.equal(formatLinearTick(8850), "8850");
  assert.deepEqual(linearTicks(5, 5), [5]);
  assert.equal(niceNum(0, true), 1);
  assert.equal(niceNum(130, true), 100);
});

test("movement is signed", () => {
  assert.equal(formatMovement(-0.123), "−0.12");
  assert.equal(formatMovement(0.5), "+0.50");
  assert.equal(formatMovement(null), "");
});

test("averaging a window: mean, median, ema", () => {
  assert.equal(averageOf([1, 2, 6], "mean", 3), 3);
  assert.equal(averageOf([1, 9, 2, 3], "median", 4), 2.5);
  const ema = averageOf([10, 20], "ema", 3); // alpha 0.5
  assert.equal(ema, 15);
});
