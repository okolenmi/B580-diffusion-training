/* The editor's app session: the app's own graph on the canvas, what counts as unsaved, and what Save sends. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { GraphDoc } from "../js/graph/doc.js";
import { AppSession } from "../js/graph/app_session.js";

const button = () => ({ handlers: {}, textContent: "", classList: { toggle() {} }, addEventListener(t, f) { this.handlers[t] = f; } });
const appGraph = { format: 3, settings: {}, nodes: [{ uid: "n_aaaaaa", class_name: "IntNode", title: "", params: { value: 3 }, layout: { x: 5, y: 6 } }], edges: [], event_edges: [], views: [] };

function setup({ app = { version: 2, doc: { graph: appGraph } }, saved } = {}) {
  const calls = [];
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url, method: init.method || "GET", body: init.body ? JSON.parse(init.body) : null });
    if ((init.method || "GET") === "PUT") return new Response(JSON.stringify(saved || { app: { version: 3 }, orphaned_cards: [] }), { status: 200 });
    return app ? new Response(JSON.stringify(app), { status: 200 }) : new Response(JSON.stringify({ error: { code: "app_not_found", message: "no app" } }), { status: 404 });
  };
  const notes = [];
  const held = [];
  const timers = { set: (f) => { f(); return 1; }, clear() {} };
  const parts = {
    name: "my app", doc: new GraphDoc(), canvas: { fit() {} }, saver: { hold: (on) => held.push(on) },
    bar: { hidden: true }, title: { textContent: "" }, saveBtn: button(), backLink: {}, onNote: (m, k) => notes.push([m, k]), timers,
  };
  return { session: new AppSession(parts), parts, calls, notes, held };
}

test("opening puts the app's graph on the canvas, keeps the browser draft out of it and shows the bar", async () => {
  const { session, parts, held, calls } = setup();
  assert.equal(await session.open(), true);
  assert.deepEqual([...parts.doc.nodes.keys()], ["n_aaaaaa"]);
  assert.deepEqual(held, [true]);
  assert.equal(parts.bar.hidden, false);
  assert.match(parts.title.textContent, /my app.*version 2/);
  assert.equal(calls[0].url, "/api/v1/apps/my%20app");
  assert.equal(session.dirty, false);
});

test("an app that cannot be read leaves the free session alone", async () => {
  const { session, parts, held, notes } = setup({ app: null });
  assert.equal(await session.open(), false);
  assert.deepEqual(held, []);
  assert.equal(parts.bar.hidden, true);
  assert.equal(notes[0][1], "error");
});

test("an edit makes the session unsaved, and Save sends the whole document with its layout, then it is clean", async () => {
  const { session, parts, calls, notes } = setup();
  await session.open();
  parts.doc.setParam("n_aaaaaa", "value", 9);
  assert.equal(session.dirty, true);
  assert.match(parts.title.textContent, /unsaved/);
  assert.equal(await session.save(), true);
  const put = calls.at(-1);
  assert.equal(put.method, "PUT");
  assert.equal(put.url, "/api/v1/apps/my%20app/graph");
  assert.equal(put.body.graph.nodes[0].params.value, 9);
  assert.deepEqual(put.body.graph.nodes[0].layout, { x: 5, y: 6 });
  assert.equal(session.dirty, false);
  assert.match(notes.at(-1)[0], /version 3/);
});

test("a node drag counts as a change worth saving, and undoing it by hand makes the session clean again", async () => {
  const { session, parts } = setup();
  await session.open();
  parts.doc.moveNode("n_aaaaaa", 50, 60);
  assert.equal(session.dirty, true);
  parts.doc.moveNode("n_aaaaaa", 5, 6);
  assert.equal(session.dirty, false);
});

test("Save says how many cards lost their target", async () => {
  const { session, notes } = setup({ saved: { app: { version: 3 }, orphaned_cards: ["b1", "b2"] } });
  await session.open();
  await session.save();
  assert.match(notes.at(-1)[0], /2 cards.*lost their target/);
  assert.equal(notes.at(-1)[1], "warn");
});
