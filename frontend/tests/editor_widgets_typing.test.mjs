/* Half-typed field text must survive a canvas re-render (no Enter pressed). */
import { test } from "node:test";
import assert from "node:assert/strict";

// Just enough DOM for buildWidget's text/number/JSON controls.
class FakeEl {
  constructor(tag) { this.tag = tag; this.value = ""; this.listeners = {}; this.children = []; this.style = {}; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  fire(type) { for (const fn of this.listeners[type] || []) fn(); }
  appendChild(c) { this.children.push(c); return c; }
  append(...c) { this.children.push(...c); }
}
globalThis.document = { createElement: (t) => new FakeEl(t), createTextNode: (t) => ({ text: t }) };

const { buildWidget } = await import("../js/editor/widgets.js");

function setup(port) {
  const node = { id: "n1", class_name: "A", params: {} };
  const changes = [];
  const doc = { classOf: () => ({ inputs: [port] }), changed: (...a) => changes.push(a) };
  const el = buildWidget({ doc, node, port, origin: "canvas", onNote: () => {}, onRebuild: null });
  return { node, changes, el };
}
const type = (el, text) => { el.value = text; el.fire("input"); };

test("typing in a text field is stored without a re-render", () => {
  const { node, changes, el } = setup({ name: "relative_path", type: "str" });
  type(el, "style/lora.safetensors");
  assert.equal(node.params.relative_path, "style/lora.safetensors");
  assert.equal(changes.length, 0, "an input event must not re-render (focus would be lost)");
  el.fire("change"); // Enter / blur still commits for real
  assert.equal(changes.length, 1);
});

test("typing a number is stored; text that is not yet a number is not", () => {
  const { node, el } = setup({ name: "lr", type: "float" });
  type(el, "0.0001");
  assert.equal(node.params.lr, 0.0001);
  type(el, "");
  assert.ok(!("lr" in node.params));
  type(el, "-");
  assert.ok(!("lr" in node.params), "a lone '-' must not become NaN");
});

test("JSON is kept only while it parses", () => {
  const { node, el } = setup({ name: "betas", type: "tuple" });
  type(el, "[0.9, 0.9");
  assert.ok(!("betas" in node.params));
  type(el, "[0.9, 0.99]");
  assert.deepEqual(node.params.betas, [0.9, 0.99]);
});
