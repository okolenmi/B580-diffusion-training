// Paging of the open dataset: the layout's automatic top-up must not hammer a failing or empty endpoint, while a person's own
// movement may try again; a stale first page must not leave the loader switched off.
import test from "node:test";
import assert from "node:assert/strict";
import { datasets } from "../js/datasets/client.js";
import { LibraryState } from "../js/datasets/library_state.js";

const items = (n, from = 0) => Array.from({ length: n }, (_, i) => ({ id: `i${from + i}`, captions: {}, sets: [] }));

function state(respond) {
  const calls = [];
  datasets.images = async (_name, _view, { offset = 0 }) => { calls.push(offset); return respond(offset); };
  const s = new LibraryState("d", () => {});
  s.items = items(80);
  s.total = 500;
  return { s, calls };
}

test("a failed page is not requested again by the automatic top-up, but by moving on", async () => {
  let fail = true;
  const { s, calls } = state((offset) => { if (fail) throw new Error("down"); return { items: items(80, offset), total: 500 }; });
  await s.loadMore();
  await s.loadMore();
  await s.loadMore();
  assert.deepEqual(calls, [80]);
  fail = false;
  await s.loadMore(true);
  assert.equal(s.items.length, 160);
  assert.deepEqual(calls, [80, 80]);
});

test("an empty page with a larger total stops the loop too", async () => {
  const { s, calls } = state(() => ({ items: [], total: 500 }));
  await s.loadMore();
  await s.loadMore();
  assert.deepEqual(calls, [80]);
});

test("a new first page clears the stall", async () => {
  const { s, calls } = state((offset) => ({ items: offset ? [] : items(80), total: 160 }));
  await s.loadMore();
  assert.equal(s.stalled, true);
  await s.load();
  assert.equal(s.stalled, false);
  await s.loadMore();
  assert.deepEqual(calls, [80, 0, 80]);
});

test("a page that finishes after the view changed does not switch loading off for the newer request", async () => {
  let release;
  let releaseFresh;
  const gate = new Promise((r) => { release = r; });
  const freshGate = new Promise((r) => { releaseFresh = r; });
  const { s } = state(async (offset) => { await (offset === 80 ? gate : freshGate); return { items: items(80, offset), total: 500 }; });
  const more = s.loadMore();
  const fresh = s.load();           // newer first page: loading stays on until it ends
  release();
  await more;
  assert.equal(s.loading, true);
  releaseFresh();
  await fresh;
  assert.equal(s.loading, false);
});
