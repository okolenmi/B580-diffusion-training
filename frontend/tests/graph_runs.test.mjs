import { test } from "node:test";
import assert from "node:assert/strict";
import { Runs, statusRun, stopTarget } from "../js/graph/runs.js";

const run = (id, status) => ({ execution_id: id, status });

// Just enough DOM for Runs.render (rows and labels, no layout).
class FakeEl {
  constructor(tag) { this.tag = tag; this.children = []; this.textContent = ""; this.className = ""; }
  replaceChildren(...c) { this.children = [...c]; }
  append(...c) { this.children.push(...c); }
  addEventListener() {}
}
globalThis.document = { createElement: (tag) => new FakeEl(tag) };

const fakeCanvas = () => ({
  statuses: new Map(),
  clears: 0,
  setNodeStatus(uid, state, text = "") { this.statuses.set(uid, { state, text }); },
  clearStatuses() { this.clears += 1; this.statuses.clear(); },
});

const runsWith = (canvas, request) => new Runs(new FakeEl("div"), canvas, {
  request, onNote: () => {}, onRunState: () => {},
});

test("statuses show the run being looked at, else the newest while following", () => {
  const items = [run(40, "running"), run(39, "finished")];
  assert.equal(statusRun(items, 39, true), 39);
  assert.equal(statusRun(items, null, true), 40);
  assert.equal(statusRun(items, null, false), null, "after leaving the runs the cards show none");
  assert.equal(statusRun([], null, true), null);
});

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

test("selecting a run clears before the fetch, so progress that arrives meanwhile survives", async () => {
  const canvas = fakeCanvas();
  let answer;
  const detail = new Promise((resolve) => { answer = resolve; });
  const runs = runsWith(canvas, () => detail);
  const picking = runs.select(7);
  runs.onEvent({ type: "graph_execution_progressed", execution_id: 7, node_id: "n_1", ok: true, duration_ms: 1200 });
  answer({ execution_id: 7, status: "running", results: [] });
  await picking;
  assert.equal(canvas.statuses.get("n_1")?.state, "done");
  assert.equal(canvas.clears, 1, "only the switch to the new run clears the cards");
});

test("with nothing selected the newest run's stored results are shown; deselect stops that", async () => {
  const canvas = fakeCanvas();
  const request = async (path) => path.startsWith("/graphs/executions?")
    ? { executions: [run(9, "finished")] }
    : { execution_id: 9, status: "finished", results: [{ node_id: "n_1", ok: true, duration_ms: 1500 }] };
  const runs = runsWith(canvas, request);
  await runs.refresh();
  assert.equal(canvas.statuses.get("n_1")?.state, "done");
  assert.equal(canvas.statuses.get("n_1")?.text, "1.5s");
  runs.deselect();
  assert.equal(canvas.statuses.size, 0);
  await runs.refresh();
  assert.equal(canvas.statuses.size, 0, "after leaving the runs a refresh does not put results back");
});

test("a stream reopen the server covered still re-reads the run, because progress is not replayed", async () => {
  const canvas = fakeCanvas();
  const paths = [];
  const request = async (path) => {
    paths.push(path);
    return path.startsWith("/graphs/executions?")
      ? { executions: [run(9, "finished")] }
      : { execution_id: 9, status: "finished", results: [{ node_id: "n_1", ok: true, duration_ms: 800 }] };
  };
  const runs = runsWith(canvas, request);
  runs.onEvent({ type: "stream_opened", resync_required: false });
  await new Promise((r) => setImmediate(r));
  assert.ok(paths.some((p) => p === "/graphs/executions/9"), "the run detail was read");
  assert.equal(canvas.statuses.get("n_1")?.state, "done");
});
