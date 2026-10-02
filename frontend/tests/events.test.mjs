/* Unit tests for lib/events.js, driven by a fake EventSource.
   The behaviours pinned here are the ones that used to differ between
   views (docs 08 N-06) and the one WP-21 changed: whether to refetch is
   the server's call (stream_opened.resync_required), not "the socket
   opened". Also: unreadable frames counted and reported rather than
   swallowed, and a valid frame still reaching the handler after a bad
   one.
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

const opened = (resyncRequired) =>
  frame(
    resyncRequired === undefined
      ? { type: "stream_opened" }
      : { type: "stream_opened", resync_required: resyncRequired },
  );

test("resync runs when the server says it could not replay", () => {
  const resync = [];
  subscribeEvents({ onEvent: () => {}, onResync: (r) => resync.push(r) });
  source().onmessage(opened(true));
  assert.equal(resync.length, 1, "one un-replayable connect, one resync");
  assert.match(resync[0], /could not replay/);
});

test("resync is SKIPPED when the server replayed the gap", () => {
  // This is the entire point of the replay ring. The old code refetched
  // on every open because it had to: it could not know whether it had a
  // hole. Refetching anyway would make the feature invisible -- correct,
  // but paying full price for it.
  const resync = [];
  subscribeEvents({ onEvent: () => {}, onResync: (r) => resync.push(r) });
  source().onmessage(opened(false));
  assert.equal(resync.length, 0, "a covered reconnect needs no refetch");
});

test("an ABSENT resync_required refetches -- absence is not a yes", () => {
  // An older server, or a proxy that eats the opening frame. Reading
  // "unknown" as "I am current" is exactly how a page ends up believing a
  // finished run is still running.
  const resync = [];
  subscribeEvents({ onEvent: () => {}, onResync: (r) => resync.push(r) });
  source().onmessage(opened(undefined));
  assert.equal(resync.length, 1, "unknown coverage means refetch");
});

test("resync happens BEFORE the frame reaches onEvent", () => {
  // Otherwise a view's handler runs against data the resync is about to
  // replace, and its patch is applied to state it has not seen yet.
  const order = [];
  subscribeEvents({
    onEvent: (e) => order.push(`event:${e.type}`),
    onResync: () => order.push("resync"),
  });
  source().onmessage(opened(true));
  assert.deepEqual(order, ["resync", "event:stream_opened"]);
});

test("stream_opened is still forwarded to onEvent", () => {
  const seen = [];
  subscribeEvents({ onEvent: (e) => seen.push(e) });
  source().onmessage(opened(false));
  assert.equal(seen.length, 1, "a view may want last_seq, or to log it");
  assert.equal(seen[0].type, "stream_opened");
});

test("the socket opening no longer implies a refetch", () => {
  // onOpen says the socket is up. It says nothing about whether this
  // client holds current data; conflating the two was the workaround
  // this feature replaces.
  const resync = [];
  subscribeEvents({ onEvent: () => {}, onResync: (r) => resync.push(r) });
  assert.equal(source().onopen, undefined, "no onOpen handler is installed");
  assert.equal(resync.length, 0);
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
