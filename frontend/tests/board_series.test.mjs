/* Unit tests for board/series.js -- live data handling. Pure; the clock is passed in. */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  StreamState, barsOf, chartRows, decimate, formatDuration, formatValue, globToRegExp,
  progressOf, referenceLines, scalarOf, seriesFor, textOf, xOf,
} from "../js/board/series.js";

const step = (n, extra = {}) => ({ step: n, total_steps: 100, loss: 1 / n, ...extra });

test("frames: connected / clear / run_end / unknown type / not-telemetry / garbage", () => {
  const s = new StreamState();
  assert.equal(s.applyFrame({ type: "connected" }).kind, "connected");
  assert.equal(s.conn, "live");
  assert.equal(s.applyFrame(step(1)).kind, "point");
  assert.equal(s.applyFrame({ type: "mystery", step: 2 }).kind, "ignored");
  assert.equal(s.applyFrame({ hello: "world" }).kind, "ignored");
  assert.equal(s.rows.length, 1);
  assert.equal(s.applyRaw("{not json").kind, "bad");
  assert.equal(s.applyFrame([1, 2]).kind, "bad");
  assert.equal(s.applyFrame(null).kind, "bad");
  assert.equal(s.bad, 3);
  assert.equal(s.applyFrame({ type: "run_end", step: 100, cancelled: false }).kind, "run_end");
  assert.deepEqual(s.runEnd, { step: 100, cancelled: false });
  s.applyFrame(step(2));                     // new report after run_end = a new run
  assert.equal(s.runEnd, null);
  s.applyFrame({ type: "clear" });
  assert.equal(s.rows.length, 0);
  assert.equal(s.conn, "live", "clear keeps the connection");
  assert.equal(Object.keys(s.last).length, 0);
});

test("a reconnect's replay does not duplicate history", () => {
  const s = new StreamState();
  s.applyFrame({ type: "connected" });
  for (let i = 1; i <= 5; i++) s.applyFrame(step(i));
  s.setConn("disconnected");
  s.applyFrame({ type: "connected" });            // EventSource reconnected
  for (let i = 1; i <= 7; i++) s.applyFrame(step(i)); // server replays all it holds
  assert.equal(s.rows.length, 7);
  assert.equal(s.conn, "live");
});

test("a key missing from a frame is a gap, and non-finite numbers are gaps", () => {
  const s = new StreamState();
  s.applyFrame({ step: 1, a: 1, b: 2 });
  s.applyFrame({ step: 2, a: 3 });
  s.applyFrame({ step: 3, a: null, b: null });
  const rows = chartRows(s, { x: "step" }, ["a", "b"]);
  assert.deepEqual(rows.map((r) => r.values), [{ a: 1, b: 2 }, { a: 3, b: null }, { a: null, b: null }]);
});

test("x=t is seconds since the first timed frame after a clear", () => {
  const s = new StreamState();
  s.applyFrame({ t: 1000.0, reserved_mb: 10 });
  s.applyFrame({ t: 1002.5, reserved_mb: 20 });
  assert.deepEqual(chartRows(s, { x: "t" }, ["reserved_mb"]).map((r) => r.x), [0, 2.5]);
  assert.equal(xOf(s, { step: 4 }, "step"), 4);
  assert.equal(xOf(s, { step: 4 }, "t"), null);
  s.applyFrame({ type: "clear" });
  s.applyFrame({ t: 5000, reserved_mb: 1 });
  assert.equal(chartRows(s, { x: "t" }, ["reserved_mb"])[0].x, 0);
});

test("row cap drops the oldest and says how many", () => {
  const s = new StreamState({ maxRows: 10 });
  for (let i = 1; i <= 25; i++) s.applyFrame(step(i));
  assert.equal(s.rows.length, 10);
  assert.equal(s.dropped, 15);
  assert.equal(s.rows[0].frame.step, 16);
  assert.equal(s.last.loss.x.step, 25, "latest survives");
});

test("glob matches like fnmatch", () => {
  const re = globToRegExp("*_ms");
  assert.ok(re.test("fwd_ms") && re.test("_ms") && !re.test("ms_x") && !re.test("fwd_msx"));
  assert.ok(globToRegExp("a?c").test("abc") && !globToRegExp("a?c").test("ac"));
  assert.ok(globToRegExp("[ab]x").test("ax") && !globToRegExp("[ab]x").test("cx"));
  assert.ok(globToRegExp("[!a]x").test("bx") && !globToRegExp("[!a]x").test("ax"));
  assert.ok(globToRegExp("a.b").test("a.b") && !globToRegExp("a.b").test("aXb"), "dot is literal");
  assert.ok(globToRegExp("a[b").test("a[b"), "unclosed bracket is literal");
});

test("seriesFor: declared + discovered (pattern, minus exclusions), colours stable as keys arrive", () => {
  const mod = { series: [{ key: "step_total_ms", label: "total", color: "#111" }], pattern: "*_ms", exclude: ["skip_ms"] };
  const s = new StreamState();
  s.applyFrame({ step: 1, step_total_ms: 9, fwd_ms: 3, skip_ms: 1, loss: 0.5 });
  let list = seriesFor(mod, s);
  assert.deepEqual(list.map((x) => x.key), ["step_total_ms", "fwd_ms"]);
  assert.equal(list[0].color, "#111");
  assert.equal(list[1].label, "fwd");
  const fwdColor = list[1].color;
  s.applyFrame({ step: 2, step_total_ms: 9, fwd_ms: 3, bwd_ms: 4 });
  list = seriesFor(mod, s);
  assert.deepEqual(list.map((x) => x.key), ["step_total_ms", "fwd_ms", "bwd_ms"]);
  assert.equal(list[1].color, fwdColor, "an existing series must not change colour");
  assert.ok(list.every((x) => x.color));
  assert.deepEqual(seriesFor({ series: [{ key: "a" }] }, s).map((x) => x.label), ["a"]);
});

test("decimate keeps every series' extremes, first and last, in order, with original rows", () => {
  const rows = Array.from({ length: 10000 }, (_, i) => ({
    x: i, values: { a: Math.sin(i / 50), b: i % 997 === 0 ? 100 : 1 },
  }));
  rows[4321].values.a = 50;           // a one-step spike
  rows[7777].values.a = -50;          // and a one-step dip
  const out = decimate(rows, 400, ["a", "b"]);
  assert.ok(out.length <= 400, `got ${out.length}`);
  assert.ok(out.includes(rows[4321]) && out.includes(rows[7777]), "spike or dip lost");
  assert.equal(out[0], rows[0]);
  assert.equal(out[out.length - 1], rows[rows.length - 1]);
  for (let i = 1; i < out.length; i++) assert.ok(out[i].x > out[i - 1].x);
  assert.ok(out.every((r) => rows.includes(r)), "decimate must not invent rows");
  assert.equal(decimate(rows.slice(0, 50), 400, ["a"]).length, 50, "short input untouched");
});

test("decimate survives all-gap series and many series", () => {
  const rows = Array.from({ length: 5000 }, (_, i) => ({ x: i, values: { a: null, b: i } }));
  const out = decimate(rows, 100, ["a", "b"]);
  assert.ok(out.length <= 100 && out.length >= 2);
  const many = Array.from({ length: 5000 }, (_, i) => ({ x: i, values: Object.fromEntries(Array.from({ length: 12 }, (_, k) => [`k${k}`, (i * (k + 1)) % 97])) }));
  const keys = Array.from({ length: 12 }, (_, k) => `k${k}`);
  assert.ok(decimate(many, 200, keys).length <= 200);
});

test("scalar: latest value, stale when the newest frame lacks the key", () => {
  const mod = { series: [{ key: "lr" }] };
  const s = new StreamState();
  assert.equal(scalarOf(mod, s), null);
  s.applyFrame({ step: 1, lr: 0.001 });
  assert.equal(scalarOf(mod, s).stale, false);
  s.applyFrame({ step: 2 });
  const r = scalarOf(mod, s);
  assert.deepEqual([r.value, r.stale, r.x.step], [0.001, true, 1]);
});

test("bars: newest present value per series, absent series omitted", () => {
  const mod = { series: [{ key: "a" }, { key: "b" }] };
  const s = new StreamState();
  s.applyFrame({ step: 1, a: 4 });
  assert.deepEqual(barsOf(mod, s).map((b) => [b.key, b.value]), [["a", 4]]);
  s.applyFrame({ step: 2, a: 5, b: 6 });
  assert.deepEqual(barsOf(mod, s).map((b) => [b.key, b.value]), [["a", 5], ["b", 6]]);
});

test("text keeps strings; reference line is the latest numeric reference value", () => {
  const s = new StreamState();
  s.applyFrame({ step: 1, status: "warming up", budget_mb: 9000 });
  assert.equal(textOf({ series: [{ key: "status" }] }, s), "warming up");
  assert.equal(textOf({ series: [{ key: "nope" }] }, s), null);
  assert.equal(textOf({ series: [] }, s), null);
  assert.deepEqual(referenceLines({ reference_key: "budget_mb" }, s), [{ value: 9000, label: "budget", color: "#ff9800" }]);
  assert.deepEqual(referenceLines({ reference_key: "nope" }, s), []);
  assert.deepEqual(referenceLines({}, s), []);
});

test("progress: fraction, rate and ETA from arrival times; none from a single sample", () => {
  const mod = { value_key: "step", total_key: "total_steps" };
  const s = new StreamState();
  assert.equal(progressOf(mod, s), null);
  s.applyFrame(step(10), 0);
  let p = progressOf(mod, s);
  assert.deepEqual([p.value, p.total, p.fraction, p.rate, p.eta], [10, 100, 0.1, null, null]);
  s.applyFrame(step(20), 10_000);           // 10 steps / 10 s
  p = progressOf(mod, s);
  assert.equal(p.rate, 1);
  assert.equal(p.eta, 80);
  const burst = new StreamState();                 // a late page: history replayed at once
  for (let i = 1; i <= 40; i++) burst.applyFrame(step(i), 5000 + i);
  assert.deepEqual([progressOf(mod, burst).rate, progressOf(mod, burst).eta], [null, null]);
  s.applyFrame(step(20), 20_000);           // stalled: dv = 0 over the window? window still spans 10->20
  p = progressOf(mod, s);
  assert.ok(p.rate > 0 && p.rate < 1);
  s.applyFrame({ type: "run_end", step: 100 });
  p = progressOf(mod, s);
  assert.equal(p.done, true);
  assert.equal(p.eta, null);
  // no total: fraction unknown, never NaN
  const t = new StreamState();
  t.applyFrame({ step: 5 }, 0);
  assert.deepEqual([progressOf(mod, t).total, progressOf(mod, t).fraction], [null, null]);
});

test("formatting", () => {
  assert.equal(formatDuration(0), "0s");
  assert.equal(formatDuration(65), "1m 05s");
  assert.equal(formatDuration(3720), "1h 02m");
  assert.equal(formatDuration(null), "—");
  assert.equal(formatDuration(NaN), "—");
  assert.equal(formatValue(1234567), "1,234,567");
  assert.equal(formatValue(0.000123456), "0.000123");
  assert.equal(formatValue(2.5), "2.500");
  assert.equal(formatValue(123.456), "123.5");
  assert.equal(formatValue(null), "—");
  assert.equal(formatValue(Infinity), "—");
});
