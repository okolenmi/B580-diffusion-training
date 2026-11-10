import test from "node:test";
import assert from "node:assert/strict";
import { fmtBytes, plural } from "../js/datasets/client.js";

test("sizes read in the unit a person expects", () => {
  assert.equal(fmtBytes(0), "0 B");
  assert.equal(fmtBytes(950), "950 B");
  assert.equal(fmtBytes(4200), "4 kB");
  assert.equal(fmtBytes(2_500_000), "2.5 MB");
  assert.equal(fmtBytes(3_200_000_000), "3.2 GB");
});

test("counts are worded in singular and plural", () => {
  assert.equal(plural(1, "image"), "1 image");
  assert.equal(plural(0, "image"), "0 images");
  assert.equal(plural(1200, "image"), "1,200 images");
  assert.equal(plural(2, "dataset"), "2 datasets");
});
