// LossChart's numbers: how the average is taken, what the tooltip leads with, and that the linear middle band follows the
// main series alone. Drawing is stubbed; only the maths is looked at.
import test from "node:test";
import assert from "node:assert/strict";

globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.requestAnimationFrame = () => 0;
globalThis.cancelAnimationFrame = () => {};
globalThis.window = { devicePixelRatio: 1, addEventListener() {}, removeEventListener() {} };
const { LossChart } = await import("../js/lib/loss_chart.js");

const canvas = () => ({
  getContext: () => new Proxy({}, { get: () => () => ({ width: 10 }), set: () => true }),
  addEventListener() {}, removeEventListener() {}, getBoundingClientRect: () => ({ left: 0, top: 0, width: 400, height: 200 }), style: {},
});
const SERIES = [{ key: "loss", label: "Loss", color: "#fff" }, { key: "high", label: "High t", color: "#f00" }];

function fill(chart, losses, highs) {
  losses.forEach((v, i) => chart.addPoint(i, { loss: v, high: highs ? highs[i] : null }));
}

test("the running mean waits for a full window, the median ignores a spike, the EMA leans on the latest", () => {
  const values = [1, 1, 1, 1, 10];
  const mean = new LossChart(canvas(), { smoothWindow: 5 }); fill(mean, values);
  assert.equal(mean.points[3].s.loss, null);
  assert.equal(mean.points[4].s.loss, 2.8);
  const median = new LossChart(canvas(), { smoothWindow: 5, averageMethod: "median" }); fill(median, values);
  assert.equal(median.points[4].s.loss, 1);
  const ema = new LossChart(canvas(), { smoothWindow: 5, averageMethod: "ema" }); fill(ema, values);
  assert.ok(ema.points[4].s.loss > 2.8 && ema.points[4].s.loss < 10);
});

test("changing the method recomputes every point from its raw values", () => {
  const chart = new LossChart(canvas(), { smoothWindow: 3 });
  fill(chart, [1, 2, 3, 4, 5, 6]);
  assert.equal(chart.points[5].s.loss, 5);
  chart.setAverage({ method: "median", window: 5 });
  assert.equal(chart.points[3].s.loss, null);
  assert.equal(chart.points[5].s.loss, 4);
  chart.setAverage({ method: "mean", window: 2 });
  assert.equal(chart.points[5].s.loss, 5.5);
});

test("the tooltip average exists before the window fills and counts only present values", () => {
  const chart = new LossChart(canvas(), { smoothWindow: 4 });
  fill(chart, [2, 4, 6], null);
  assert.equal(chart.points[2].s.loss, null);               // the drawn line waits
  assert.equal(chart._averageAt(chart.points[2], "loss"), 4); // the tooltip does not
  assert.equal(chart._averageAt(chart.points[1], "high"), null);
});

test("the linear middle band follows the main series alone; the other series only widen the log ends", () => {
  const chart = new LossChart(canvas(), { series: SERIES, smoothWindow: 2 });
  fill(chart, Array.from({ length: 10 }, (_, i) => 0.10 + i * 0.001), Array.from({ length: 10 }, (_, i) => 0.5 + i * 0.05));
  const r = chart._computeRangeCore();
  assert.ok(r.linMax < 0.12, `band ends near the main line (got ${r.linMax})`);
  assert.ok(r.fullMax >= 0.95, `the wide series still fits the top end (got ${r.fullMax})`);
  chart.series[0].hidden = true;                             // main series hidden: the next shown one leads
  assert.ok(chart._computeRangeCore().linMax > 0.5);
});
