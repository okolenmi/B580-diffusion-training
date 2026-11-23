/* Unit tests for lib/adapter_rows.js -- the value and the pure operations of the adapter list control. */

import assert from "node:assert/strict";
import test from "node:test";

import { addRow, moveBy, offered, patchAt, removeAt, rowsOf, splitName } from "../js/lib/adapter_rows.js";

test("a stored list is read as rows; the older text form reads as the same rows", () => {
  assert.deepEqual(rowsOf([{ file: "a.safetensors", strength: "0.5" }, { nothing: 1 }]), [{ file: "a.safetensors", strength: 0.5, on: true }]);
  assert.deepEqual(rowsOf("a.safetensors\n# note\nb.safetensors=0.5\nc.safetensors=off\n"), [
    { file: "a.safetensors", strength: 1, on: true }, { file: "b.safetensors", strength: 0.5, on: true }, { file: "c.safetensors", strength: 1, on: false }]);
  assert.deepEqual(rowsOf(undefined), []);
});

test("add, remove, move and patch return new lists and keep the order honest", () => {
  const rows = addRow(addRow([], "a"), "b");
  assert.equal(addRow(rows, "a"), rows);
  assert.deepEqual(moveBy(rows, 1, -1).map((r) => r.file), ["b", "a"]);
  assert.equal(moveBy(rows, 0, -1), rows);
  assert.deepEqual(removeAt(rows, 0).map((r) => r.file), ["b"]);
  assert.equal(patchAt(rows, 1, { strength: 0.2 })[1].strength, 0.2);
  assert.equal(rows[1].strength, 1);
});

test("a deep file name shows its short name and its folder; the offer skips listed files and matches every word", () => {
  assert.deepEqual(splitName("styles/ink/line.safetensors"), { folder: "styles/ink/", base: "line" });
  assert.deepEqual(splitName("plain.safetensors"), { folder: "", base: "plain" });
  const files = ["styles/ink/line.safetensors", "styles/paint/oil.safetensors", "chars/Fox.safetensors"];
  assert.deepEqual(offered(files, "STYLES oil", []), ["styles/paint/oil.safetensors"]);
  assert.deepEqual(offered(files, "", [{ file: "chars/Fox.safetensors" }]).length, 2);
});
