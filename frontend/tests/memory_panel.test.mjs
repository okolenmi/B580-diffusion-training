import test from "node:test";
import assert from "node:assert/strict";

import {
  describeFit,
  describeHolders,
  describePeak,
  describeRefusal,
  fmtMb,
  fmtSetting,
  panelModel,
} from "../js/lib/memory_panel.js";

/* A real /graphs/memory-preview response, as the B580's numbers produce
   one: capacity 11,192 MB, a configuration measured at 7,666 MB. */
const MEASURED = {
  fingerprint_key: "fp-abc",
  peak_mb: 7666,
  demand_mb: 7816,
  demand_source: "observed",
  exploratory: false,
  device_demand_mb: 8416,
  verdict: "fits",
  reason: null,
  capacity_mb: 11192,
  free_mb: 11192,
  held_mb: 0,
  foreign_reserve_mb: 1024,
  holders: {},
};

test("fmtMb separates thousands and dashes an unknown", () => {
  assert.equal(fmtMb(7666), "7,666 MB");
  assert.equal(fmtMb(0), "0 MB");
  assert.equal(fmtMb(11192.4), "11,192 MB");
  // The load-bearing ones: a number that was never measured, and a
  // number that is garbage, are both unknowns -- never 0.
  assert.equal(fmtMb(null), "—");
  assert.equal(fmtMb(undefined), "—");
  assert.equal(fmtMb(NaN), "—");
  assert.equal(fmtMb(Infinity), "—");
  assert.equal(fmtMb("6000"), "—", "a string is not a number here");
});

test("fmtSetting keeps auto as a real answer", () => {
  assert.equal(fmtSetting("auto"), "auto");
  assert.equal(fmtSetting(8000), "8,000 MB");
  // An unrecognised string is a value the panel cannot describe, so it
  // is unknown rather than echoed back as if it meant something.
  assert.equal(fmtSetting("everything"), "—");
  assert.equal(fmtSetting(null), "—");
});

test("describePeak separates an unkeyed graph from an unmeasured one", () => {
  assert.match(describePeak(MEASURED), /7,666 MB/);
  // Known fingerprint, never run: "nothing measured", not "needs nothing".
  const unmeasured = { ...MEASURED, peak_mb: null };
  assert.match(describePeak(unmeasured), /Never measured/);
  // The fingerprint itself unknown is a different sentence again.
  const unkeyed = { ...MEASURED, fingerprint_key: null, peak_mb: null };
  assert.match(describePeak(unkeyed), /cannot be keyed/);
  assert.doesNotMatch(describePeak(unkeyed), /Never measured/);
  assert.match(describePeak(null), /No measurement yet/);
});

test("describeFit gives four distinct answers", () => {
  const fits = describeFit(MEASURED);
  assert.equal(fits.verdict, "fits");
  assert.equal(fits.level, "ok");
  assert.match(fits.detail, /8,416 MB/);
  assert.match(fits.detail, /11,192 MB/);

  const refuses = describeFit({ ...MEASURED, verdict: "does_not_fit",
    device_demand_mb: 9416, free_mb: 1768, reason: "needs 9416 MB but only 1768 MB is free" });
  assert.equal(refuses.level, "bad");
  assert.match(refuses.detail, /only 1,768 MB is free/);

  // Unknown demand is neither a yes nor a no.
  const exploratory = describeFit({ ...MEASURED, verdict: "exploratory" });
  assert.equal(exploratory.level, "warn");
  assert.match(exploratory.headline, /whole card/);

  // No ledger: nothing was checked, and the panel must not imply it was.
  const unknown = describeFit({ ...MEASURED, verdict: "unknown",
    device_demand_mb: null, free_mb: null });
  assert.equal(unknown.level, "warn");
  assert.match(unknown.detail, /has not been read/);

  // A verdict the panel has never heard of must not fall through to a
  // reassuring "fits"; it becomes the honest unknown.
  const nonsense = describeFit({ ...MEASURED, verdict: "definitely-fine" });
  assert.equal(nonsense.verdict, "unknown");
  assert.notEqual(nonsense.level, "ok");
  // And a missing response entirely.
  assert.equal(describeFit(null).verdict, "unknown");
});

test("describeHolders puts the biggest claimant first", () => {
  const rows = describeHolders({
    "task:small": { mb: 500 },
    "graph:1": { mb: 8416 },
    "task:big": { mb: 4000 },
  });
  assert.deepEqual(rows.map((r) => r.owner), ["graph:1", "task:big", "task:small"]);
  assert.equal(rows[0].mb, "8,416 MB");
  assert.deepEqual(describeHolders({}), []);
  assert.deepEqual(describeHolders(null), []);
});

test("describeRefusal renders the breakdown it was given", () => {
  const refusal = describeRefusal({
    requested_mb: 9192,
    capacity_mb: 11192,
    free_mb: 1768,
    foreign_reserve_mb: 1024,
    holders: { "task:ingest_lora": 9424 },
    reason: "a graph run cannot be admitted",
  });
  assert.match(refusal.headline, /cannot be admitted/);
  assert.deepEqual(refusal.rows, [
    ["Asked for", "9,192 MB"],
    ["Free", "1,768 MB"],
    ["Capacity", "11,192 MB"],
    ["Left to other users", "1,024 MB"],
  ]);
  assert.deepEqual(refusal.holders, [{ owner: "task:ingest_lora", mb: "9,424 MB" }]);
});

test("describeRefusal accepts both holder shapes, and degrades honestly", () => {
  // The ledger's snapshot uses {owner: {mb}}; its refusal breakdown
  // uses {owner: mb}. Either may arrive; both must render.
  const nested = describeRefusal({ holders: { a: { mb: 100 } } });
  assert.deepEqual(nested.holders, [{ owner: "a", mb: "100 MB" }]);

  // No breakdown at all: the reason still shows, and no invented rows.
  const bare = describeRefusal(null, "the run was refused");
  assert.equal(bare.headline, "the run was refused");
  assert.deepEqual(bare.rows, []);
  assert.deepEqual(bare.holders, []);

  // A non-numeric holder is dropped rather than rendered as NaN MB.
  const junk = describeRefusal({ holders: { a: "lots" } });
  assert.deepEqual(junk.holders, []);
});

test("panelModel assembles the whole panel as data", () => {
  const model = panelModel(MEASURED);
  assert.deepEqual(model.settings, [
    ["Min VRAM", "—"],
    ["Max VRAM", "—"],
    ["Demand", "7,816 MB (observed)"],
    ["Device total", "11,192 MB"],
    ["Free now", "11,192 MB"],
  ]);
  assert.match(model.peak, /7,666 MB/);
  assert.equal(model.fit.verdict, "fits");

  // An entirely unknown configuration reads as unknown everywhere, and
  // nowhere as a zero.
  const blank = panelModel(null);
  assert.match(blank.peak, /No measurement yet/);
  assert.equal(blank.fit.verdict, "unknown");
  const demand = blank.settings.find(([k]) => k === "Demand");
  assert.match(demand[1], /unknown/);
});

test("panelModel marks an unmeasured demand as unknown, not as 0 MB", () => {
  const exploratory = panelModel({
    ...MEASURED,
    peak_mb: null,
    demand_mb: null,
    demand_source: "unknown",
    exploratory: true,
    verdict: "exploratory",
  });
  const demand = exploratory.settings.find(([k]) => k === "Demand");
  assert.equal(demand[1], "unknown (never measured)");
  assert.doesNotMatch(demand[1], /0 MB/);
});