/* The canvas draft (reload survival) and half-typed field values. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { GraphDoc } from "../js/editor/state.js";
import { DRAFT_KEY, createDraftSaver, readDraft, writeDraft } from "../js/editor/draft.js";

const memoryStorage = () => {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => void m.set(k, String(v)),
    removeItem: (k) => void m.delete(k),
    _m: m,
  };
};

const CLASSES = {
  A: { class_name: "A", inputs: [
    { name: "x", type: "float" }, { name: "y", type: "str" }] },
};
const docWith = () => {
  const d = new GraphDoc();
  d.classByName = CLASSES;
  const a = d.addNode("A", 10, 20);
  const b = d.addNode("A", 300, 40);
  a.params.x = 0.5;
  b.params.y = "kept";
  d.addEdge(a.id, "x", b.id, "x"); // feeds b.x: the run payload would drop b.x
  b.params.x = 2; // redundant while wired, must survive in a draft
  d.setMemory("vram_max_mb", 9000);
  return d;
};

test("a draft round-trips the canvas: nodes, positions, wires, memory, raw params", () => {
  const s = memoryStorage();
  const d = docWith();
  assert.ok(writeDraft(d, s));
  const back = new GraphDoc();
  back.classByName = CLASSES;
  back.load(readDraft(s));
  assert.deepEqual(back.toDraft(), d.toDraft());
  assert.equal(back.nodes.get("n2").params.x, 2, "wire-fed param dropped from the draft");
  assert.equal(back.nodes.get("n1").x, 10);
  assert.equal(back.nextId, 3, "new nodes must not reuse ids after a restore");
});

test("an empty canvas removes the draft, so Clear survives a reload", () => {
  const s = memoryStorage();
  const d = docWith();
  writeDraft(d, s);
  assert.ok(s._m.has(DRAFT_KEY));
  d.clear();
  writeDraft(d, s);
  assert.equal(readDraft(s), null);
  assert.ok(!s._m.has(DRAFT_KEY));
});

test("corrupt, foreign or empty storage reads as no draft, never as an error", () => {
  for (const bad of ["{not json", "null", "5", '{"nodes":"x"}', '{"nodes":[]}']) {
    const s = memoryStorage();
    s.setItem(DRAFT_KEY, bad);
    assert.equal(readDraft(s), null, bad);
  }
  assert.equal(readDraft(null), null);
  const denied = { getItem() { throw new Error("denied"); } };
  assert.equal(readDraft(denied), null);
});

test("a full or denied storage makes the write fail quietly", () => {
  const full = { setItem() { throw new Error("quota"); }, removeItem() {} };
  assert.equal(writeDraft(docWith(), full), false);
  assert.equal(writeDraft(docWith(), null), false);
});

test("the saver debounces changes and flush writes immediately", () => {
  const s = memoryStorage();
  const d = docWith();
  const pending = [];
  const timers = {
    setTimeout: (fn) => { pending.push(fn); return pending.length; },
    clearTimeout: (id) => { pending[id - 1] = null; },
  };
  const saver = createDraftSaver(d, { storage: s, timers });
  saver.notify(); saver.notify(); saver.notify();
  assert.equal(s._m.size, 0, "wrote before the debounce fired");
  pending.filter(Boolean).forEach((fn) => fn());
  assert.ok(s._m.has(DRAFT_KEY));
  s._m.clear();
  saver.notify();
  assert.ok(saver.flush());
  assert.ok(s._m.has(DRAFT_KEY));
});
