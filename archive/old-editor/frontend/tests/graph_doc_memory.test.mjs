/* GraphDoc carries the graph's memory settings through every payload form. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { GraphDoc } from "../js/editor/state.js";

const withNode = () => {
  const d = new GraphDoc();
  d.classByName = { A: { class_name: "A", inputs: [] } };
  d.addNode("A", 0, 0);
  return d;
};

test("an untouched graph submits exactly what it always did (no memory key)", () => {
  assert.ok(!("memory" in withNode().toRunPayload()));
});

test("settings travel in the run payload and the library payload", () => {
  const d = withNode();
  d.setMemory("vram_max_mb", 9000);
  d.setMemory("strict", false);
  assert.deepEqual(d.toRunPayload().memory, { vram_max_mb: 9000, strict: false });
  assert.deepEqual(d.toLibraryPayload().memory, { vram_max_mb: 9000, strict: false });
});

test("load restores settings, including keys the section does not draw", () => {
  const d = new GraphDoc();
  d.load({ nodes: [], edges: [], memory: { vram_max_mb: "auto", policy: "demand_driven", device: "xpu" } });
  assert.deepEqual(d.toRunPayload().memory, { vram_max_mb: "auto", policy: "demand_driven", device: "xpu" });
  d.setMemory("vram_min_mb", 512);
  assert.equal(d.toRunPayload().memory.policy, "demand_driven", "an undrawn key was dropped");
});

test("clearing a setting removes it; clear() and a memory-less load reset", () => {
  const d = withNode();
  d.setMemory("vram_max_mb", 9000);
  d.setMemory("vram_max_mb", "");
  assert.ok(!("memory" in d.toRunPayload()));
  d.setMemory("strict", false);
  d.clear();
  assert.deepEqual(d.memory, {});
  d.setMemory("strict", false);
  d.load({ nodes: [], edges: [] });
  assert.deepEqual(d.memory, {});
});

test("load ignores a malformed memory block instead of throwing", () => {
  const d = new GraphDoc();
  for (const bad of ["x", 5, [1, 2], null]) {
    d.load({ nodes: [], edges: [], memory: bad });
    assert.deepEqual(d.memory, {});
  }
});

test("a memory change notifies with reason 'memory'", () => {
  const d = new GraphDoc();
  const seen = [];
  d.onChange = (r, o) => seen.push([r, o]);
  d.setMemory("strict", false);
  assert.deepEqual(seen, [["memory", "settings"]]);
});

test("load/save round trip keeps the loaded copy independent of the source", () => {
  const src = { nodes: [], edges: [], memory: { vram_max_mb: 8000 } };
  const d = new GraphDoc();
  d.load(src);
  d.setMemory("vram_max_mb", 1);
  assert.equal(src.memory.vram_max_mb, 8000);
});
