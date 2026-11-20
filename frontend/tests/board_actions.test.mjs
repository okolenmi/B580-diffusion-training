/* Unit tests for action cards: the node's answers kept by StreamState, how actions are found, offered in the tray and placed by the
   draft builder. Pure; the button itself (DOM) is checked in the browser run. */
import { test } from "node:test";
import assert from "node:assert/strict";

import { StreamState } from "../js/board/series.js";
import { autoFill, buildAvailable, defaultSize, findTarget, nodeRef, titleFor } from "../js/board/model.js";

const action = (extra = {}) => ({ name: "save_now", label: "Save now", doc: "", confirm: null, available: true, reason: "", ...extra });
const modules = (actions) => ({
  graph_fields: [], graph_displays: [],
  nodes: [{ id: "n_train", display_name: "Trainer", known: true, fields: [], displays: [], actions, stream: "node:n_train" }],
});

test("StreamState keeps the node's answers by command id, last status winning, and forgets them on clear", () => {
  const s = new StreamState();
  const frame = (status, detail = "") => ({ kind: "action", id: "c1", action: "save_now", status, detail });
  assert.equal(s.applyFrame(frame("queued"), 5).kind, "action");
  s.applyFrame(frame("delivered"), 9);
  assert.deepEqual(s.actions.get("c1"), { action: "save_now", status: "delivered", detail: "", at: 9 });
  assert.equal(s.rows.length, 0, "an answer is not a data point for charts");
  s.applyFrame({ type: "clear" });
  assert.equal(s.actions.size, 0);
});

test("an action is found on its node, titled by its label, and sized as a small button", () => {
  const m = modules([action()]);
  const ref = nodeRef("n_train", "action", "save_now");
  const { node, descriptor } = findTarget(ref, m);
  assert.equal(node.id, "n_train");
  assert.equal(descriptor.label, "Save now");
  assert.equal(titleFor({ id: "b1", ref, title: null }, m).title, "Save now");
  assert.deepEqual(defaultSize(ref, m, 12), [3, 2]);
  assert.equal(findTarget(nodeRef("n_train", "action", "nope"), m).descriptor, null);
});

test("the tray offers actions and marks the placed one", () => {
  const m = modules([action()]);
  const groups = buildAvailable(m, []);
  const item = groups[0].items.find((i) => i.kind === "action");
  assert.equal(item.label, "Save now");
  assert.equal(item.placed, false);
  const placed = buildAvailable(m, [{ id: "b1", ref: item.ref, x: 0, y: 0, w: 3, h: 2 }]);
  assert.equal(placed[0].items.find((i) => i.kind === "action").placed, true);
});

test("the draft builder puts a usable action on the page, and leaves out one nothing can act on", () => {
  const has = (blocks) => blocks.some((b) => b.ref.kind === "action");
  assert.equal(has(autoFill(modules([action()]))), true);
  assert.equal(has(autoFill(modules([action({ available: false, reason: "add a saver" })]))), false);
});
