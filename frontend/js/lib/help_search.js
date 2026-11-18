// @ts-check
/* ---------------------------------------------------------------------------
   lib/help_search.js -- which help blocks a typed query keeps.

   Every word must appear somewhere in the block (any order, any case): "memory batch" finds the block that mentions both, which a
   phrase match would miss. An empty query keeps everything.
   --------------------------------------------------------------------------- */

/**
 * @param {string} text   the block's text
 * @param {string} query  what was typed
 */
export function matches(text, query) {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const haystack = text.toLowerCase();
  return words.every((word) => haystack.includes(word));
}
