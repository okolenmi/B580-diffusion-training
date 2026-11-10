import test from "node:test";
import assert from "node:assert/strict";
import { applyEdit, EDITS } from "../js/datasets/caption_edit.js";

test("prepend and append add a trigger word once", () => {
  assert.equal(applyEdit("a cat", "prepend", "zq"), "zq, a cat");
  assert.equal(applyEdit("zq, a cat", "prepend", "zq"), "zq, a cat");
  assert.equal(applyEdit("a cat", "append", "zq"), "a cat, zq");
  assert.equal(applyEdit("a cat, zq", "append", "zq"), "a cat, zq");
});

test("an empty caption gets the text without a stray comma", () => {
  assert.equal(applyEdit("", "prepend", "zq"), "zq");
  assert.equal(applyEdit("", "append", "zq"), "zq");
});

test("replace changes every occurrence and ignores an empty find", () => {
  assert.equal(applyEdit("girl, girl hair", "replace", "boy", "girl"), "boy, boy hair");
  assert.equal(applyEdit("girl", "replace", "boy", ""), "girl");
});

test("set and clear overwrite", () => {
  assert.equal(applyEdit("old", "set", "new"), "new");
  assert.equal(applyEdit("old", "clear", "x"), "");
});

test("every offered edit is one applyEdit understands", () => {
  for (const [mode] of EDITS) assert.doesNotThrow(() => applyEdit("a", mode, "b", "a"));
});
