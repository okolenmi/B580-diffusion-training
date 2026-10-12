/* The chart window (last N / freeze / slider) and the CSV export. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { ViewWindow, rowsForWindow } from "../js/board/view.js";
import { rowsToCsv } from "../js/board/series.js";

const xs = (n) => Array.from({ length: n }, (_, i) => i * 10); // 0,10,20...

test("All shows everything; a count follows the newest N records", () => {
  const v = new ViewWindow();
  let a = v.apply(xs(300));
  assert.equal(a.range, null);
  assert.ok(a.sliderDisabled);
  v.setCount(100, xs(300));
  a = v.apply(xs(300));
  assert.deepEqual(a.range, { min: 2000, max: 2990 });
  assert.equal(a.maxIdx, 200);
  a = v.apply(xs(310)); // newer records arrive: the window follows
  assert.deepEqual(a.range, { min: 2100, max: 3090 });
});

test("freeze holds the window while records keep arriving, resume follows again", () => {
  const v = new ViewWindow();
  v.setCount(50, xs(300));
  v.toggleFreeze(xs(300));
  const held = v.apply(xs(300)).range;
  assert.deepEqual(held, { min: 2500, max: 2990 });
  assert.deepEqual(v.apply(xs(400)).range, held, "a held view moved with incoming data");
  assert.ok(v.apply(xs(400)).frozen);
  v.toggleFreeze(xs(400));
  assert.deepEqual(v.apply(xs(400)).range, { min: 3500, max: 3990 });
});

test("widening a frozen window keeps its END anchored", () => {
  const v = new ViewWindow();
  v.setCount(50, xs(300));
  v.toggleFreeze(xs(300));
  v.setCount(100, xs(300));
  assert.deepEqual(v.apply(xs(300)).range, { min: 2000, max: 2990 });
});

test("the slider pans back; its far end returns to live", () => {
  const v = new ViewWindow();
  v.setCount(50, xs(300));
  v.pan(100, xs(300));
  let a = v.apply(xs(300));
  assert.ok(a.frozen);
  assert.deepEqual(a.range, { min: 1000, max: 1490 });
  v.pan(a.maxIdx, xs(300)); // dragged to the far end
  a = v.apply(xs(300));
  assert.ok(!a.frozen);
  assert.deepEqual(a.range, { min: 2500, max: 2990 });
});

test("a held slider keeps the far end it was grabbed with", () => {
  const v = new ViewWindow();
  v.setCount(50, xs(300));
  v.setDragging(true, 250);
  v.pan(250, xs(400)); // data grew while held: the old far end still means 'live'
  assert.ok(!v.apply(xs(400)).frozen);
  v.setDragging(false);
  assert.equal(v.dragging, false);
});

test("a new run restarts the follow but keeps the chosen count", () => {
  const v = new ViewWindow();
  v.setCount(100, xs(300));
  v.toggleFreeze(xs(300));
  v.restart();
  assert.equal(v.count, 100);
  assert.ok(!v.apply(xs(20)).frozen);
  assert.equal(v.apply([]).first, null);
});

test("rowsForWindow keeps a margin before the range for the running mean", () => {
  const rows = xs(100).map((x) => ({ x }));
  assert.equal(rowsForWindow(rows, null).length, 100);
  const part = rowsForWindow(rows, { min: 500, max: 700 }, 5);
  assert.equal(part[0].x, 450);
  assert.equal(part.at(-1).x, 700);
});

test("CSV: a column per key, a gap stays empty, text is quoted", () => {
  const csv = rowsToCsv([
    { frame: { step: 1, loss: 0.5, note: 'a,"b"' } },
    { frame: { step: 2, loss: null, extra: 7 } },
  ]);
  assert.equal(csv, 'step,loss,note,extra\n1,0.5,"a,""b""",\n2,,,7\n');
});

test("after a clear a stream has no series at all (what the chart must redraw as empty)", async () => {
  const { StreamState, seriesFor, chartRows } = await import("../js/board/series.js");
  const s = new StreamState();
  const mod = { series: [], pattern: "loss*", x: "step" };
  s.applyFrame({ step: 1, loss: 0.5 });
  assert.equal(seriesFor(mod, s).length, 1);
  s.applyFrame({ type: "clear" });
  assert.equal(seriesFor(mod, s).length, 0, "premise of the stale-chart fix");
  assert.equal(chartRows(s, mod, []).length, 0);
});
