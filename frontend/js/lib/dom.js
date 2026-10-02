// @ts-check
/* ---------------------------------------------------------------------------
   lib/dom.js -- the two element helpers every view had its own copy of.

   `el` is one line, in six files, and was six chances to spell it
   differently. The message-box helper was two lines copied four times
   under three names (`showError` twice, `showState` once), and one view
   forgetting `hidden = false` would leave a stale error on screen after
   the user fixed the problem.

   Both take elements rather than ids where they can, so the caller
   decides what "this error box" means.
   --------------------------------------------------------------------------- */

/**
 * @param {string} id
 * @returns {HTMLElement}
 */
export function el(id) {
  const node = document.getElementById(id);
  // A null here would fail later, at the point of use, with a message
  // about whatever property was being set. Saying so at the boundary is
  // both truer and cheaper to debug.
  if (node === null) throw new Error(`no element with id "${id}"`);
  return node;
}

/**
 * Show or clear a message box. An empty message hides the box, so the
 * call site does not have to remember to hide it -- which is the part
 * that was being forgotten, and a stale error left on screen after the
 * user fixed the problem reads as "still broken".
 *
 * Takes an id or an element. Both spellings were in use (config.js passed
 * an element, datasets.js an id, and `showState` in datasets.js was a
 * third copy of this body under another name), and consolidating meant
 * choosing. Accepting both keeps 35 call sites readable -- the ids are
 * literal strings next to the markup they name -- while leaving one
 * definition of the behaviour.
 *
 * @param {HTMLElement|string} target
 * @param {string} message  "" clears
 */
export function showMessage(target, message) {
  const box = typeof target === "string" ? el(target) : target;
  box.textContent = message || "";
  box.hidden = !message;
}