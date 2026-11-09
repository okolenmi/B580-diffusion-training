/* graph2 palette: which classes fit a dropped wire, search, grouping. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { byCategory, compatibleClasses, matches } from "../js/graph2/palette.js";

const port = (name, type, mro = [type, "object"], extra = {}) => ({ name, label: name, type, type_mro: mro, ...extra });
const NODES = [
  { class_name: "Make", display_name: "Make Number", category: "Primitive", inputs: [], outputs: [port("y", "int", ["int", "float", "object"])] },
  { class_name: "Use", display_name: "Use Float", category: "Schedule", inputs: [port("x", "float"), port("note", "str", undefined, { widget_only: true })], outputs: [port("z", "float")] },
  { class_name: "Text", display_name: "Text In", category: "Primitive", inputs: [port("s", "str")], outputs: [] },
];

test("an output finds inputs it satisfies; widget-only inputs never match", () => {
  const found = compatibleClasses(NODES, NODES[0].outputs[0], true);
  assert.deepEqual(found.map((f) => [f.info.class_name, f.port.name]), [["Use", "x"]]);
});

test("an input finds outputs that satisfy it", () => {
  const found = compatibleClasses(NODES, NODES[1].inputs[0], false);
  assert.deepEqual(found.map((f) => f.info.class_name).sort(), ["Make", "Use"]);
});

test("a dynamic node without outputs is offered through its possible outputs, flagged", () => {
  const dynamic = { class_name: "Load", display_name: "Load", category: "Model", inputs: [], outputs: [], possible_outputs: [port("w", "float")] };
  const found = compatibleClasses([...NODES, dynamic], NODES[1].inputs[0], false);
  assert.deepEqual(found.filter((f) => f.possible).map((f) => f.info.class_name), ["Load"]);
  assert.equal(found.find((f) => f.info.class_name === "Make").possible, false);
  const wantsFloat = { ...dynamic, class_name: "Wants", possible_outputs: [], possible_inputs: [port("v", "float")] };
  const toward = compatibleClasses([wantsFloat], NODES[0].outputs[0], true);
  assert.deepEqual(toward.map((f) => [f.info.class_name, f.possible]), [["Wants", true]], "a possible input is offered to a dragged output");
});

test("search matches name, class and category; grouping sorts both levels", () => {
  assert.ok(matches(NODES[0], "number") && matches(NODES[0], "PRIMITIVE") && matches(NODES[0], "make") && !matches(NODES[0], "zzz"));
  const groups = byCategory(NODES);
  assert.deepEqual(groups.map(([c, l]) => [c, l.map((n) => n.class_name)]), [["Primitive", ["Make", "Text"]], ["Schedule", ["Use"]]]);
  assert.deepEqual(byCategory(NODES, "float").map(([c]) => c), ["Schedule"]);
});
