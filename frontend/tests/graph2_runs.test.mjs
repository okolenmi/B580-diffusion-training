import { test } from "node:test";
import assert from "node:assert/strict";
import { stopTarget } from "../js/graph2/runs.js";

const run = (id, status) => ({ execution_id: id, status });

test("Stop prefers a running run over the planned row that was just started and looked at", () => {
  const items = [run(36, "planned"), run(35, "running")];
  assert.equal(stopTarget(items, 36), 35);
});

test("Stop acts on the looked-at run when it is the running one", () => {
  const items = [run(37, "running"), run(36, "running")];
  assert.equal(stopTarget(items, 36), 36);
});

test("with nothing running Stop cancels the looked-at planned run, else the newest planned", () => {
  const items = [run(40, "planned"), run(39, "planned"), run(38, "finished")];
  assert.equal(stopTarget(items, 39), 39);
  assert.equal(stopTarget(items, 38), 40);
  assert.equal(stopTarget([run(1, "finished")], null), null);
});
