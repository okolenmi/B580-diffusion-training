/* Unit tests for lib/events.js, driven by a fake EventSource.
   The behaviours pinned here are the ones that used to differ between
   views (docs 08 N-06): resync on every open including the first,
   unreadable frames counted and reported rather than swallowed, and a
   valid frame still reaching the handler after a bad one.
   Run: node --test frontend/tests  */

import { test } from "node:test";
import assert from "node:assert/strict";

/* events.js imports { sse } from ../api.js, which does `new
   EventSource(...)` and reads BASE from a module-level constant. Rather
   than pulling the whole module in (and depending on the URL constant),
   the seam is provided by installing globals before the import. */
let lastSource = null;
globalThis.EventSource = class {
  constructor(url) {
    this.url = url;
    lastSource = this;
  }
  close() { this.closed = true; }
};
globalThis.location = { origin: "http://test" };

const { subscribeEvents, NOTICE_INTERVAL_MS } = await import("../js/lib/events.js");

const source = () => lastSource;
const frame = (obj) => ({ data: JSON.stringify(obj) });
const badFrame = () => ({ data: "{not json" });

test("resync runs on EVERY open, including the first", () => {
  const resync = [];
  subscribeEvents({ onEvent: () => {}, onResync: (r) => resync.push(r) });
  source().onopen();
  source().onopen();
  source().onopen();
  assert.equal(resync.length, 3, "three opens, three resyncs");
});

test("a valid frame reaches onEvent", () => {
  const seen = [];
  subscribeEvents({ onEvent: (e) => seen.push(e) });
  source().onmessage(frame({ type: "run_progressed", run_id: 1, step: 7 }));
  assert.equal(seen.length, 1);
  assert.equal(seen[0].step, 7);
});

test("an unreadable frame is reported, counted, and does not throw", () => {
  const seen = [];
  const notices = [];
  subscribeEvents({ onEvent: (e) => seen.push(e), onNotice: (m) => notices.push(m) });
  source().onmessage(badFrame());
  assert.equal(notices.length, 1, "said out loud, not swallowed");
  assert.match(notices[0], /Unreadable event frame dropped/);
  assert.equal(seen.length, 0, "and nothing bogus reached the handler");
});

test("a bad frame does not stop the stream: the next good one still arrives", () => {
  const seen = [];
  subscribeEvents({ onEvent: (e) => seen.push(e), onNotice: () => {} });
  source().onmessage(badFrame());
  source().onmessage(frame({ type: "run_completed", run_id: 3 }));
  assert.equal(seen.length, 1);
  assert.equal(seen[0].type, "run_completed");
});

test("notices are rate-limited, and the count stays exact", async () => {
  const notices = [];
  const sub = subscribeEvents({ onEvent: () => {}, onNotice: (m) => notices.push(m) });
  for (let i = 0; i < 50; i += 1) source().onmessage(badFrame());
  // All 50 counted...
  assert.equal(sub.unreadable(), 50);
  // ...but not 50 log lines: the first is immediate, the rest wait for
  // the interval.
  assert.equal(notices.length, 1);
  assert.match(notices[0], /\(1 so far\)/);
  await new Promise((r) => setTimeout(r, NOTICE_INTERVAL_MS + 20));
  source().onmessage(badFrame());
  assert.equal(notices.length, 2, "a later failure is reported again");
  assert.match(notices[1], /\(51 so far\)/);
  sub.close();
});

test("close() closes the underlying source", () => {
  const sub = subscribeEvents({ onEvent: () => {} });
  const s = source();
  sub.close();
  assert.equal(s.closed, true);
});

test("the transport's own error is surfaced, not swallowed", () => {
  const errors = [];
  subscribeEvents({ onEvent: () => {}, onError: (m) => errors.push(m) });
  source().onerror();
  assert.equal(errors.length, 1);
});

test("a caller that supplies no callbacks still gets a working stream", () => {
  const sub = subscribeEvents({});
  assert.doesNotThrow(() => source().onmessage(badFrame()));
  assert.doesNotThrow(() => source().onmessage(frame({ type: "x" })));
  sub.close();
});
