/* Unit tests for the console's data path: the journal client (history, polling cursor, local fallback, attention) and the pure model of what is shown.
   Run: node --test frontend/tests */

import { test } from "node:test";
import assert from "node:assert/strict";

globalThis.document = { addEventListener() {}, hidden: false, getElementById: () => null };
globalThis.window = { dispatchEvent() {} };

const { JournalClient } = await import("../js/console/journal_client.js");
const { visible, stamp, problemCount, asText } = await import("../js/console/journal_model.js");

const row = (seq, level = "info", text = `line ${seq}`) => ({ seq, at: 1_700_000_000 + seq, level, source: "server", text });

function fakeServer(rows = []) {
  const calls = [];
  const down = { on: false };
  const call = async (path, opts = {}) => {
    calls.push([path, opts]);
    if (down.on) throw new Error("no route");
    if (opts.method === "POST") { const r = row(rows.length + 1, opts.body.level, opts.body.text); r.source = "page"; rows.push(r); return { entries: [r], last: r.seq }; }
    const after = Number(new URL(`http://x${path}`).searchParams.get("after"));
    return { entries: rows.filter((r) => r.seq > after), last: rows.at(-1)?.seq ?? 0 };
  };
  return { rows, calls, down, call };
}

function client(server) {
  const shouts = [];
  const c = new JournalClient({ call: server.call, shout: (name) => shouts.push(name), hidden: () => false });
  return { c, shouts };
}

test("the history comes first, then only what is new, asked from the last number", async () => {
  const server = fakeServer([row(1), row(2)]);
  const { c } = client(server);
  const heard = [];
  c.subscribe((batch, info) => heard.push([batch.map((e) => e.seq), info.initial]));
  await c.poll({ initial: true });
  server.rows.push(row(3));
  await c.poll();
  assert.deepEqual(heard, [[[1, 2], true], [[3], false]]);
  assert.match(server.calls.at(-1)[0], /after=2/, "the second look starts after the last entry the first one returned");
  clearTimeout(c.timer);
});

test("an error that is new to the page asks the console to open; the history does not", async () => {
  const server = fakeServer([row(1, "error")]);
  const { c, shouts } = client(server);
  await c.poll({ initial: true });
  assert.deepEqual(shouts, [], "an old failure is not news");
  server.rows.push(row(2, "warn"), row(3, "error"));
  await c.poll();
  assert.deepEqual(shouts, ["console:attention"]);
  clearTimeout(c.timer);
});

test("an error that is the very first entry ever is still news", async () => {
  const server = fakeServer([]);
  const { c, shouts } = client(server);
  await c.poll({ initial: true });
  server.rows.push(row(1, "error"));
  await c.poll();
  assert.deepEqual(shouts, ["console:attention"]);
  clearTimeout(c.timer);
});

test("a line written by the page goes to the server and comes back like any other", async () => {
  const server = fakeServer();
  const { c } = client(server);
  const heard = [];
  c.subscribe((batch) => heard.push(...batch));
  await c.poll({ initial: true });
  await c.write("success", "Saved \"x\".");
  await c.polling;
  assert.deepEqual(heard.map((e) => [e.seq, e.level, e.source, e.text]), [[1, "success", "page", "Saved \"x\"."]]);
  clearTimeout(c.timer);
});

test("when the server cannot be reached the line stays on this page, and the loss and the return are said once", async () => {
  const server = fakeServer();
  const { c } = client(server);
  const heard = [];
  c.subscribe((batch) => heard.push(...batch));
  await c.poll({ initial: true });
  server.down.on = true;
  await c.write("error", "Could not save.");
  await c.poll();
  await c.poll();
  assert.deepEqual(heard.map((e) => e.text), ["Could not save.", "The server does not answer; this page keeps trying."]);
  assert.ok(heard.every((e) => e.seq === null), "local entries have no number");
  server.down.on = false;
  await c.poll();
  assert.equal(heard.at(-1).text, "The server answers again.");
  clearTimeout(c.timer);
});

test("a page that subscribes late is given what the client already holds", async () => {
  const server = fakeServer([row(1), row(2)]);
  const { c } = client(server);
  await c.poll({ initial: true });
  const heard = [];
  c.subscribe((batch, info) => heard.push([batch.length, info.initial]));
  assert.deepEqual(heard, [[2, true]]);
  clearTimeout(c.timer);
});

test("the filter and the clear mark decide what is shown; local entries always show", () => {
  const all = [row(1, "info"), row(2, "warn"), row(3, "error"), { ...row(0, "info"), seq: null }];
  assert.deepEqual(visible(all, "all", 0).map((e) => e.seq), [1, 2, 3, null]);
  assert.deepEqual(visible(all, "problems", 0).map((e) => e.seq), [2, 3]);
  assert.deepEqual(visible(all, "errors", 0).map((e) => e.seq), [3]);
  assert.deepEqual(visible(all, "all", 2).map((e) => e.seq), [3, null]);
  assert.deepEqual(visible(all, "nonsense", 0).length, 4, "an unknown filter shows everything");
  assert.equal(problemCount(all), 2);
});

test("a time is the clock time today and carries the date on another day", () => {
  const at = new Date(2026, 10, 27, 14, 2, 11).getTime() / 1000;
  assert.equal(stamp(at, new Date(2026, 10, 27, 20, 0, 0)), "14:02:11");
  assert.equal(stamp(at, new Date(2026, 10, 28, 9, 0, 0)), "11-27 14:02:11");
  assert.match(asText({ at, level: "error", source: "run #7", text: "failed: x" }, new Date(2026, 10, 27, 20)), /14:02:11 {2}ERROR\s+run #7\s+failed: x/);
});
