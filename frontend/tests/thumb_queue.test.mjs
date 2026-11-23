/* Unit tests for datasets/thumb_queue.js -- which thumbnail is fetched next, and what happens when one fails. Pure. */

import assert from "node:assert/strict";
import test from "node:test";

import { ThumbQueue } from "../js/datasets/thumb_queue.js";

const at = (id) => Number(id);

test("the nearest wanted picture goes first, and no more than the limit run at once", () => {
  const q = new ThumbQueue({ limit: 2 });
  for (const id of ["100", "900", "400", "450"]) q.want(id);
  assert.equal(q.next(at, 420), "400");
  assert.equal(q.next(at, 420), "450");
  assert.equal(q.next(at, 420), null);
  q.loaded("400");
  assert.equal(q.next(at, 420), "100");
});

test("a picture that left the neighbourhood before its turn is not fetched; a loaded one is not wanted again", () => {
  const q = new ThumbQueue();
  q.want("1"); q.want("2");
  q.unwant("1");
  assert.equal(q.next(at, 0), "2");
  q.loaded("2");
  q.want("2");
  assert.equal(q.next(at, 0), null);
});

test("a failure is retried with growing waits and then given up on", () => {
  const q = new ThumbQueue({ retries: 2, wait: 100 });
  q.want("7");
  assert.equal(q.next(at, 0), "7");
  assert.equal(q.failed("7"), 100);
  q.retry("7");
  assert.equal(q.next(at, 0), "7");
  assert.equal(q.failed("7"), 200);
  q.retry("7");
  assert.equal(q.next(at, 0), "7");
  assert.equal(q.failed("7"), null);
});
