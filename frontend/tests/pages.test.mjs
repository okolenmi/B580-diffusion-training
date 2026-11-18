/* Purpose: the logic behind the start page and Settings -- what is concluded from the server's answers (failures named once, an
   unreadable answer never shown as a good one) and when a settings form counts as changed. */
import test from "node:test";
import assert from "node:assert/strict";
import { checklist, memoryView, nodeHealth, remaining } from "../js/lib/home_model.js";

test("nodeHealth: one shared cause is named once; no answer is unknown, not ok", () => {
  assert.equal(nodeHealth(null).state, "unknown");
  assert.equal(nodeHealth({ load_errors: [] }).state, "ok");
  const same = nodeHealth({ load_errors: [{ module: "a", error: "No module named 'torch'" }, { module: "b", error: "No module named 'torch'" }] });
  assert.deepEqual([same.state, same.failed, same.reason], ["bad", 2, "No module named 'torch'"]);
  assert.match(nodeHealth({ load_errors: [{ module: "a", error: "x" }, { module: "b", error: "y" }] }).reason, /other errors/);
});

test("memoryView: an unmeasured card is not a card with zero memory", () => {
  assert.equal(memoryView({ total_mb: null, holders: {} }).known, false);
  assert.equal(memoryView(null).known, false);
  const view = memoryView({ total_mb: 12000, held_mb: 3000, free_mb: 8000, holders: { "run-3": { mb: 3000 } } });
  assert.deepEqual([view.known, view.freeMb, view.holders], [true, 8000, [{ owner: "run-3", mb: 3000 }]]);
});

test("checklist: items are open until the server said they are done", () => {
  const unknown = checklist({ environmentReady: null, checkpoints: null, datasets: null, promptFiles: null, runs: null });
  assert.equal(remaining(unknown).length, 5);
  const done = checklist({ environmentReady: true, checkpoints: 2, datasets: 1, promptFiles: [{ complete: false }, { complete: true }], runs: 4 });
  assert.equal(remaining(done).length, 0);
  const partial = checklist({ environmentReady: true, checkpoints: 0, datasets: 1, promptFiles: [], runs: 0 });
  assert.deepEqual(remaining(partial).map((item) => item.id), ["checkpoint", "prompts", "run"]);
});

import { venvFolder } from "../js/lib/environment_check.js";

test("venvFolder: the folder of a virtualenv's interpreter, nothing for a bare command", () => {
  assert.equal(venvFolder("/home/me/proj/venv/bin/python"), "/home/me/proj/venv");
  assert.equal(venvFolder("C:\\proj\\venv\\Scripts\\python.exe"), "C:\\proj\\venv");
  assert.equal(venvFolder("python"), null);
  assert.equal(venvFolder(""), null);
});

import { changedKeys } from "../js/lib/settings_form.js";

test("changedKeys: an empty field is a change only when an override is stored; whitespace is not a change", () => {
  assert.deepEqual(changedKeys({ models_dir: "/m" }, { models_dir: " /m ", loras_dir: "" }), []);
  assert.deepEqual(changedKeys({ models_dir: "/m" }, { models_dir: "", loras_dir: "" }), ["models_dir"]);
  assert.deepEqual(changedKeys({}, { loras_dir: "/l" }), ["loras_dir"]);
});

import { matches } from "../js/lib/help_search.js";

test("help search: every word must be present, in any order and case; nothing typed keeps everything", () => {
  const block = "Lower the batch size or the resolution when a run does not fit in memory";
  assert.ok(matches(block, ""));
  assert.ok(matches(block, "MEMORY batch"));
  assert.ok(!matches(block, "memory tokenizer"));
});
