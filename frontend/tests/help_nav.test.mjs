/* Unit tests for the pure parts of Help: lib/help_search.js (which words to mark) and lib/side_nav.js (which section is being read). */

import assert from "node:assert/strict";
import test from "node:test";

import { matches, queryWords, splitHits } from "../js/lib/help_search.js";
import { sectionAt } from "../js/lib/side_nav.js";

test("every word must be in the block, in any order and case", () => {
  assert.equal(matches("Lower the batch size to save Memory", "memory BATCH"), true);
  assert.equal(matches("Lower the batch size", "memory batch"), false);
  assert.equal(matches("anything", "  "), true);
  assert.deepEqual(queryWords("  Two  words "), ["two", "words"]);
});

test("a text is cut around every word, overlapping and touching hits joining into one run", () => {
  assert.deepEqual(splitHits("Run the run", ["run"]), [{ text: "Run", hit: true }, { text: " the ", hit: false }, { text: "run", hit: true }]);
  assert.deepEqual(splitHits("batchsize", ["batch", "size"]), [{ text: "batchsize", hit: true }]);
  assert.deepEqual(splitHits("abcd", ["abc", "bcd"]), [{ text: "abcd", hit: true }]);
  assert.deepEqual(splitHits("nothing", ["x"]), [{ text: "nothing", hit: false }]);
});

test("the section under the reading line is current, not the one barely showing at the top", () => {
  const boxes = [{ id: "a", top: -400, bottom: 30 }, { id: "b", top: 30, bottom: 700 }, { id: "c", top: 700, bottom: 1100 }];
  const none = { atTop: false, atBottom: false };
  assert.equal(sectionAt(boxes, 320, none), "b"); // a's last 30px are visible, b is what is read
  assert.equal(sectionAt([{ id: "a", top: 0, bottom: 100 }, { id: "b", top: 140, bottom: 900 }], 120, none), "a"); // in a gap: the nearer one
});

test("at either end of the scroll the first or last section is current, as the line may never reach them", () => {
  const boxes = [{ id: "a", top: 0, bottom: 200 }, { id: "b", top: 200, bottom: 900 }, { id: "z", top: 900, bottom: 960 }];
  assert.equal(sectionAt(boxes, 400, { atTop: true, atBottom: false }), "a");
  assert.equal(sectionAt(boxes, 400, { atTop: false, atBottom: true }), "z");
  assert.equal(sectionAt([], 0, { atTop: false, atBottom: false }), null);
});
