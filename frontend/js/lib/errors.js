// @ts-check
/* ---------------------------------------------------------------------------
   lib/errors.js -- one sentence for a failure.

   Four views spelled this out and two more (under the name `_errText`)
   spelled it out again with a slightly different falsy check. All of them
   exist to answer the same question: what do I show the user?

   The answer is the server's own `code: message` when there is one. That
   matters: a bare message drops the code, and the code is what the API
   reference documents and what a bug report needs. So the machine code
   comes first, not as decoration.

   Pure, no DOM: that is what makes it testable under plain `node --test`
   (frontend/tests/errors.test.mjs).
   --------------------------------------------------------------------------- */

import { ApiError } from "../api.js";

/**
 * The one sentence to show for anything that failed.
 *
 * @param {unknown} err
 * @returns {string}
 */
export function errText(err) {
  if (err instanceof ApiError) return `${err.code}: ${err.message}`;
  if (err instanceof Error) return err.message;
  return String(err);
}