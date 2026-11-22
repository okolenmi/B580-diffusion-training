/* Unit tests for lib/monitor_format.js -- how the monitor page words numbers and exports a session. Pure; no DOM. */

import assert from "node:assert/strict";
import test from "node:test";

import { fmt, fmtDuration, fmtMB, toCsv } from "../js/lib/monitor_format.js";

test("absent values read as a dash, never a zero", () => {
  assert.equal(fmt(undefined), "—");
  assert.equal(fmtMB(null), "—");
  assert.equal(fmtDuration(NaN), "—");
});

test("sizes and durations pick the unit a person reads", () => {
  assert.equal(fmtMB(512.4), "512 MB");
  assert.equal(fmtMB(2048), "2.0 GB");
  assert.equal(fmtDuration(59), "59s");
  assert.equal(fmtDuration(125), "2m 5s");
  assert.equal(fmtDuration(3700), "1h 1m");
  assert.equal(fmt(2), "2.0000");
  assert.equal(fmt(0.0001), "1.00e-4");
});

test("CSV columns are the union of keys in first-seen order; missing cells stay empty and quoting is applied", () => {
  const csv = toCsv([{ step: 1, loss: 0.5 }, { step: 2, note: 'a,"b"' }]);
  assert.equal(csv, 'step,loss,note\n1,0.5,\n2,,"a,""b"""');
});
