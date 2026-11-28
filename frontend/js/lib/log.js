// @ts-check
/* ---------------------------------------------------------------------------
   lib/log.js -- what a page says to the person: one line in the console.

   The console is the server's journal (console/journal_client.js), so a line written here is kept, is seen by every page and
   survives a reload. `logError` is built from `errText` rather than repeating its branch: the same sentence is used in a console
   line and in an error box, and two spellings of one sentence is how a bug report ends up quoting neither.
   --------------------------------------------------------------------------- */

import { journal } from "../console/journal_client.js";
import { errText } from "./errors.js";

/**
 * @param {string} message
 * @param {"info"|"success"|"warn"|"error"} [kind]
 */
export function log(message, kind = "info") {
  journal.write(kind, String(message));
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
