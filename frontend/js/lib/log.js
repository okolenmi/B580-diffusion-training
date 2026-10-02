// @ts-check
/* ---------------------------------------------------------------------------
   lib/log.js -- the console strip at the bottom of every page.

   This was four byte-identical copies in the four views, plus a fifth
   shape of the same idea in the editor. They disagreed only in
   whitespace, which is the best kind of duplication to have and no reason
   to keep: every one of them had to remember the 60-line cap, and a copy
   that forgets it grows the DOM without bound.

   One definition, so the cap is stated once. `logError` is built from
   `errText` rather than repeating its branch: the same sentence is used
   in a console line and in an error box, and two spellings of one
   sentence is how a bug report ends up quoting neither.
   --------------------------------------------------------------------------- */

import { el } from "./dom.js";
import { errText } from "./errors.js";

export const MAX_CONSOLE_LINES = 60;
/**
 * How many lines the strip keeps. Oldest go first: the interesting line
 * is the one that just happened, and a page that has been open for an
 * hour should not have to scroll to see that something failed now.
 */
const CONSOLE_ID = "console-output";

/**
 * @param {string} message
 * @param {"info"|"success"|"warn"|"error"} [kind]
 */
export function log(message, kind = "info") {
  const out = el(CONSOLE_ID);
  const line = document.createElement("div");
  line.className = `console-line ${kind}`;
  // textContent, never innerHTML: messages carry file paths, prompts and
  // server-supplied error text, all of which are untrusted.
  line.textContent = message;
  out.appendChild(line);
  while (out.children.length > MAX_CONSOLE_LINES) {
    out.removeChild(out.firstChild);
  }
  out.scrollTop = out.scrollHeight;
}

/**
 * Log anything that was thrown, using the same sentence the error boxes
 * use.
 *
 * @param {unknown} err
 */
export function logError(err) {
  log(errText(err), "error");
}