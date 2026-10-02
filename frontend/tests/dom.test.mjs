/* Unit tests for lib/errors.js and lib/dom.js -- no browser, no real DOM.
   Run: node --test frontend/tests

   errText is pure, so it is tested directly. showMessage needs an element
   but only touches two properties, so a plain object is a better fake than
   a DOM: it makes the *contract* (textContent plus hidden, empty hides)
   the thing under test rather than the browser's. */

import { test } from "node:test";
import assert from "node:assert/strict";

import { ApiError } from "../js/api.js";
import { errText } from "../js/lib/errors.js";

const { el, showMessage } = await import("../js/lib/dom.js");

/* ---- errText ----------------------------------------------------------- */

test("errText leads with the server's code, because that is what is documented", () => {
  const err = new ApiError("run_already_active", "run 4 is still running", 409);
  assert.equal(errText(err), "run_already_active: run 4 is still running");
});

test("errText on a plain Error is its message", () => {
  assert.equal(errText(new Error("boom")), "boom");
});

test("errText on a TypeError keeps the type's message, not 'undefined'", () => {
  assert.equal(errText(new TypeError("x is not a function")), "x is not a function");
});

test("errText on a non-Error still says something", () => {
  assert.equal(errText("just a string"), "just a string");
  assert.equal(errText(42), "42");
  assert.equal(errText(null), "null");
  assert.equal(errText(undefined), "undefined");
});

test("errText on an Error with an empty message is not empty", () => {
  // String(err) would give "Error", which is a type name pretending to be
  // a diagnosis. The empty message is the honest answer.
  assert.equal(errText(new Error("")), "");
});

/* ---- showMessage ------------------------------------------------------- */

const box = () => ({ textContent: "stale", hidden: true });

test("showMessage shows a message and clears the hidden flag", () => {
  const el_ = box();
  showMessage(el_, "something went wrong");
  assert.equal(el_.textContent, "something went wrong");
  assert.equal(el_.hidden, false);
});

test("showMessage with an empty message clears AND hides", () => {
  // The part that was being forgotten in four separate copies: clearing
  // the text but leaving the box visible shows a stale error after the
  // problem is fixed.
  const el_ = box();
  showMessage(el_, "gone");
  showMessage(el_, "");
  assert.equal(el_.textContent, "");
  assert.equal(el_.hidden, true);
});

test("showMessage accepts an element id as well as an element", () => {
  const nodes = { "items-error": box() };
  globalThis.document = { getElementById: (id) => nodes[id] ?? null };
  showMessage("items-error", "by id");
  assert.equal(nodes["items-error"].textContent, "by id");
  assert.equal(nodes["items-error"].hidden, false);
});

/* ---- el ---------------------------------------------------------------- */

test("el returns the node", () => {
  const node = { id: "run-state" };
  globalThis.document = { getElementById: () => node };
  assert.equal(el("run-state"), node);
});

test("el says which id was missing, instead of failing later at the use site", () => {
  globalThis.document = { getElementById: () => null };
  assert.throws(() => el("nope"), /no element with id "nope"/);
});