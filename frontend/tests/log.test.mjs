/* Unit tests for lib/log.js -- the console strip.
   Run: node --test frontend/tests

   Driven by a minimal fake element rather than a real DOM. That is a
   deliberate choice: what is worth testing here is the 60-line cap (the
   one number four copies each had to remember) and that a message is
   written as text, never as markup. Both are visible through a small
   fake, and a real DOM would only make the test slower to read. */

import { test } from "node:test";
import assert from "node:assert/strict";

/* A stand-in for the elements log() touches: appendChild, firstChild,
   removeChild, children.length, textContent, className, scrollTop. */
function fakeNode() {
  return {
    children: [],
    textContent: "",
    className: "",
    scrollTop: 0,
    scrollHeight: 100,
    appendChild(child) { this.children.push(child); return child; },
    removeChild(child) { this.children.splice(this.children.indexOf(child), 1); },
    get firstChild() { return this.children[0] ?? null; },
  };
}

const out = fakeNode();
globalThis.document = {
  getElementById: (id) => (id === "console-output" ? out : null),
  createElement: () => fakeNode(),
};

const { log, logError, MAX_CONSOLE_LINES } = await import("../js/lib/log.js");
// api.js reads `location` at module scope, so it is imported after the
// stub is in place -- like log.js itself, not inside a test body.
const { ApiError } = await import("../js/api.js");

function lines() { return out.children; }
function reset() { out.children = []; out.scrollTop = 0; }

test("a line carries the message and the kind as its class", () => {
  reset();
  log("run #4 started", "success");
  assert.equal(lines().length, 1);
  assert.equal(lines()[0].textContent, "run #4 started");
  assert.equal(lines()[0].className, "console-line success");
});

test("the kind defaults to info", () => {
  reset();
  log("plain");
  assert.equal(lines()[0].className, "console-line info");
});

test("the message is set as text, so markup in it is not markup", () => {
  reset();
  // A prompt, a file path or a server error string all reach here, and
  // none of them are trusted. textContent is the whole reason this is
  // safe; the test pins that it is what is used.
  log("<img src=x onerror=alert(1)>");
  assert.equal(lines()[0].textContent, "<img src=x onerror=alert(1)>");
  assert.equal(lines()[0].className.includes("img"), false);
});

test(`the strip is capped at ${MAX_CONSOLE_LINES} lines, oldest first`, () => {
  reset();
  const extra = MAX_CONSOLE_LINES + 25;
  for (let i = 0; i < extra; i += 1) log(`line ${i}`);
  assert.equal(lines().length, MAX_CONSOLE_LINES,
    "an unbounded strip grows the DOM forever");
  // What survives must be the *recent* lines: the interesting one is the
  // one that just happened.
  assert.equal(lines()[0].textContent, `line ${extra - MAX_CONSOLE_LINES}`);
  assert.equal(lines()[lines().length - 1].textContent, `line ${extra - 1}`);
});

test("the strip scrolls to the newest line", () => {
  reset();
  log("first");
  out.scrollTop = 0;
  log("second");
  assert.equal(out.scrollTop, out.scrollHeight);
});

test("logError logs one line, as an error, with the server's code", () => {
  reset();
  logError(new ApiError("dataset_not_found", "no dataset named 'x'", 404));
  assert.equal(lines().length, 1);
  assert.equal(lines()[0].className, "console-line error");
  assert.equal(lines()[0].textContent, "dataset_not_found: no dataset named 'x'");
});

test("logError on a plain Error uses its message", () => {
  reset();
  logError(new Error("kaboom"));
  assert.equal(lines()[0].textContent, "kaboom");
  assert.equal(lines()[0].className, "console-line error");
});