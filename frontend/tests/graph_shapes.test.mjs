/* graph shape store: static ports, debounced resolve, superseded answers, error keeps the last good shape. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { ShapeStore } from "../js/graph/shapes.js";

const CATALOG = { nodes: [
  { class_name: "Static", inputs: [{ name: "a" }], outputs: [] },
  { class_name: "Dyn", inputs: [{ name: "file" }], outputs: [{ name: "old" }], dynamic: { depends_on: ["file"], debounce_ms: 50 } },
  { class_name: "Dyn2", inputs: [], outputs: [{ name: "old2" }], dynamic: { depends_on: ["file"], debounce_ms: 50 } },
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

test("a uid reused for another class does not inherit the old resolved shape", async () => {
  const store = new ShapeStore({ resolve: async (cls) => ({ state: "ready", inputs: [], outputs: [{ name: cls }] }) });
  store.setCatalog(CATALOG);
  const dyn = { uid: "n_1", class_name: "Dyn", params: {} };
  await store.resolveNow(dyn);
  assert.equal(store.shapeOf(dyn).outputs[0].name, "Dyn");
  const reused = { uid: "n_1", class_name: "Static" };
  assert.deepEqual(store.shapeOf(reused).inputs, [{ name: "a" }]);
  assert.equal(store.stateOf("n_1", "Static").state, "idle");
  assert.equal(store.stateOf("n_1", "Dyn").state, "ready");
  // Another dynamic class replaces the entry instead of reusing it.
  const dyn2 = { uid: "n_1", class_name: "Dyn2", params: {} };
  await store.resolveNow(dyn2);
  assert.equal(store.shapeOf(dyn2).outputs[0].name, "Dyn2");
  assert.equal(store.stateOf("n_1", "Dyn").state, "idle");
});

test("forget drops the entry, and an answer still on its way lands nowhere", async () => {
  const answer = deferred();
  const store = new ShapeStore({ resolve: () => answer.promise });
  store.setCatalog(CATALOG);
  const node = { uid: "n_1", class_name: "Dyn", params: {} };
  const pending = store.resolveNow(node);
  store.forget("n_1");
  answer.resolve({ state: "ready", inputs: [], outputs: [{ name: "late" }] });
  await pending;
  assert.equal(store.stateOf("n_1").state, "idle");
  assert.deepEqual(store.shapeOf(node).outputs, [{ name: "old" }]);
});

test("resolveNow cancels a pending debounce, so the node is asked once", async () => {
  const timers = fakeTimers();
  let calls = 0;
  const store = new ShapeStore({ timers, resolve: async () => { calls += 1; return { state: "ready", inputs: [], outputs: [] }; } });
  store.setCatalog(CATALOG);
  const node = { uid: "n_1", class_name: "Dyn", params: {} };
  store.schedule(node);
  await store.resolveNow(node);
  assert.equal(timers.pending(), 0);
  timers.fire();
  await new Promise((r) => setImmediate(r));
  assert.equal(calls, 1);
});
