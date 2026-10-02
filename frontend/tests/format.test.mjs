/* Unit tests for lib/format.js -- pure functions, no browser, no DOM.
   Run: node --test frontend/tests   (wired into scripts/full_gate.sh) */

import { test } from "node:test";
import assert from "node:assert/strict";

import { fmtDuration, fmtNum, fmtRel, fmtTime } from "../js/lib/format.js";

test("fmtNum: fixed decimals when there is room, exponential when not", () => {
  assert.equal(fmtNum(null), "—");
  assert.equal(fmtNum(undefined), "—");
  // 0 is below both fixed-decimal thresholds, so it takes the
  // exponential branch -- pinned because it looks like a bug and is not.
  assert.equal(fmtNum(0), "0.00e+0");
  assert.equal(fmtNum(1), "1.0000");
  assert.equal(fmtNum(0.001), "0.00100");
  assert.equal(fmtNum(0.5), "0.50000");
  assert.equal(fmtNum(0.0005), "5.00e-4");
});

test("fmtDuration: the largest unit that still says something", () => {
  assert.equal(fmtDuration(0), "0s");
  assert.equal(fmtDuration(45_000), "45s");
  assert.equal(fmtDuration(90_000), "1m 30s");
  assert.equal(fmtDuration(3_600_000), "1h 00m");
  assert.equal(fmtDuration(90_000_000), "1d 1h");
});

test("fmtDuration: absent or nonsensical input is an em dash, never a guess", () => {
  // The two merged copies used to disagree here (run.js answered
  // "just now" for a non-finite value). A duration is not a timestamp,
  // and "just now" would be a statement about the present tense.
  assert.equal(fmtDuration(NaN), "—");
  assert.equal(fmtDuration(-1), "—");
  assert.equal(fmtDuration(Infinity), "—");
  assert.equal(fmtDuration(undefined), "—");
});

test("fmtRel: relative time, and an unparseable timestamp claims nothing", () => {
  const now = Date.now();
  assert.equal(fmtRel(now), "just now");
  assert.equal(fmtRel(now - 30_000), "30s ago");
  assert.equal(fmtRel(now - 300_000), "5m ago");
  assert.equal(fmtRel(now - 7_200_000), "2h ago");
  assert.equal(fmtRel(now - 172_800_000), "2d ago");
  assert.equal(fmtRel(NaN), "—");
  assert.equal(fmtRel(undefined), "—");
});

test("fmtTime: absent ISO string is an em dash", () => {
  assert.equal(fmtTime(null), "—");
  assert.equal(fmtTime(""), "—");
  assert.equal(fmtTime(undefined), "—");
  // A real one renders wall clock plus relative; only assert it is not
  // the em dash and carries the parenthetical, since the wall-clock
  // part is locale-dependent.
  const rendered = fmtTime("2026-10-02T10:00:00Z");
  assert.notEqual(rendered, "—");
  assert.match(rendered, /\(.+\)$/);
});
