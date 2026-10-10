import { test } from "node:test";
import assert from "node:assert/strict";
import { LiveStreams } from "../js/board/live.js";

function fake() {
  const opened = [];
  const open = (id, handlers) => {
    const src = { id, handlers, closed: false, close() { this.closed = true; } };
    opened.push(src);
    return src;
  };
  return { opened, live: new LiveStreams({ open, now: () => 1234 }) };
}
const msg = (src, obj) => src.handlers.onMessage({ data: JSON.stringify(obj) });

test("one connection per stream however many blocks watch it; closes with the last", () => {
  const { opened, live } = fake();
  const a = live.acquire("node:t", () => {});
  const b = live.acquire("node:t", () => {});
  live.acquire("graph:run", () => {});
  assert.equal(opened.length, 2);
  assert.strictEqual(a.state, b.state);
  a.release(); a.release();                       // releasing twice is harmless
  assert.equal(opened[0].closed, false);
  b.release();
  assert.equal(opened[0].closed, true);
  assert.deepEqual(live.openIds, ["graph:run"]);
});

test("frames reach every listener; ignored frames do not; errors mark disconnected", () => {
  const { opened, live } = fake();
  let n1 = 0, n2 = 0;
  const a = live.acquire("s", () => n1++);
  live.acquire("s", () => n2++);
  const src = opened[0];
  msg(src, { type: "connected" });
  msg(src, { step: 1, loss: 2 });
  msg(src, { type: "mystery" });
  assert.deepEqual([n1, n2], [2, 2]);
  assert.equal(a.state.rows.length, 1);
  assert.equal(a.state.rows[0].at, 1234);
  src.handlers.onError();
  assert.equal(a.state.conn, "disconnected");
  assert.equal(n1, 3);
});

test("re-acquiring after the last release starts a fresh, empty stream", () => {
  const { opened, live } = fake();
  const a = live.acquire("s", () => {});
  msg(opened[0], { step: 1, loss: 2 });
  a.release();
  const b = live.acquire("s", () => {});
  assert.equal(opened.length, 2);
  assert.equal(b.state.rows.length, 0);
});

test("closeAll closes everything", () => {
  const { opened, live } = fake();
  live.acquire("a", () => {}); live.acquire("b", () => {});
  live.closeAll();
  assert.ok(opened.every((s) => s.closed));
  assert.deepEqual(live.openIds, []);
});
