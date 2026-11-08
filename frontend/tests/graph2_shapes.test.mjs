/* graph2 shape store: static ports, debounced resolve, superseded answers, error keeps the last good shape. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { ShapeStore } from "../js/graph2/shapes.js";

const CATALOG = { nodes: [
  { class_name: "Static", inputs: [{ name: "a" }], outputs: [] },
  { class_name: "Dyn", inputs: [{ name: "file" }], outputs: [{ name: "old" }], dynamic: { depends_on: ["file"], debounce_ms: 50 } },
] };

const fakeTimers = () => {
  const jobs = new Map();
  let next = 1;
  return {
    set: (fn) => { jobs.set(next, fn); return next++; },
    clear: (id) => jobs.delete(id),
    fire: () => { for (const [id, fn] of [...jobs]) { jobs.delete(id); fn(); } },
    pending: () => jobs.size,
  };
};

const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };

test("a static class has its catalog ports; an unknown class has none", () => {
  const store = new ShapeStore({ resolve: async () => ({}) });
  store.setCatalog(CATALOG);
  assert.deepEqual(store.shapeOf({ uid: "n_1", class_name: "Static" }).inputs, [{ name: "a" }]);
  assert.equal(store.shapeOf({ uid: "n_1", class_name: "Nope" }), null);
  assert.ok(!store.dependsOn("Static", "a"));
  assert.ok(store.dependsOn("Dyn", "file") && !store.dependsOn("Dyn", "other"));
});

test("schedule waits for the debounce, restarts on a new change, and applies the answer", async () => {
  const timers = fakeTimers();
  const calls = [];
  const store = new ShapeStore({ timers, resolve: async (cls, body) => { calls.push(body.params.file); return { state: "ready", inputs: [{ name: "x" }], outputs: [{ name: "new" }], info: "ok" }; } });
  store.setCatalog(CATALOG);
  const node = { uid: "n_1", class_name: "Dyn", params: { file: "a" } };
  const seen = [];
  store.onShape((e) => seen.push(e.state));
  store.schedule(node);
  node.params = { file: "b" };
  store.schedule(node);
  assert.equal(timers.pending(), 1);
  timers.fire();
  await new Promise((r) => setImmediate(r));
  assert.deepEqual(calls, ["b"]);
  assert.deepEqual(store.shapeOf(node).outputs, [{ name: "new" }]);
  assert.equal(store.stateOf("n_1").info, "ok");
  assert.deepEqual(seen, ["resolving", "resolving", "resolving", "ready"]);
});

test("a superseded answer is ignored, whatever order the answers arrive in", async () => {
  const first = deferred(), second = deferred();
  const queue = [first, second];
  const store = new ShapeStore({ resolve: () => queue.shift().promise });
  store.setCatalog(CATALOG);
  const node = { uid: "n_1", class_name: "Dyn", params: {} };
  const a = store.resolveNow(node);
  const b = store.resolveNow(node);
  second.resolve({ state: "ready", inputs: [], outputs: [{ name: "second" }] });
  await b;
  first.resolve({ state: "ready", inputs: [], outputs: [{ name: "first" }] });
  await a;
  assert.equal(store.shapeOf(node).outputs[0].name, "second");
});

test("an error keeps the last good shape and carries the message; needs_choice keeps its focus", async () => {
  const answers = [{ state: "ready", inputs: [], outputs: [{ name: "good" }] }, { state: "error", message: "bad file" }, { state: "needs_choice", inputs: [], outputs: [], focus: "family", choices: { family: ["sdxl", "anima"] } }];
  const store = new ShapeStore({ resolve: async () => answers.shift() });
  store.setCatalog(CATALOG);
  const node = { uid: "n_1", class_name: "Dyn", params: {} };
  await store.resolveNow(node);
  await store.resolveNow(node);
  assert.equal(store.stateOf("n_1").state, "error");
  assert.equal(store.stateOf("n_1").message, "bad file");
  assert.equal(store.shapeOf(node).outputs[0].name, "good");
  await store.resolveNow(node);
  assert.equal(store.stateOf("n_1").focus, "family");
  assert.deepEqual(store.stateOf("n_1").choices.family, ["sdxl", "anima"]);
});

test("a failed request is an error state, not an exception", async () => {
  const store = new ShapeStore({ resolve: async () => { throw new Error("offline"); } });
  store.setCatalog(CATALOG);
  await store.resolveNow({ uid: "n_1", class_name: "Dyn", params: {} });
  assert.equal(store.stateOf("n_1").message, "offline");
});
