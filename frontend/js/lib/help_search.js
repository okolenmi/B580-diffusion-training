// @ts-check
/* ---------------------------------------------------------------------------
   lib/help_search.js -- which help blocks a typed query matches, and where in a text the words are.

   Every word must appear somewhere in the block (any order, any case): "memory batch" finds the block that mentions both, which a
   phrase match would miss. An empty query matches everything.
   --------------------------------------------------------------------------- */

/** The words of a query, lower case. */
export function queryWords(/** @type {string} */ query) {
  return query.toLowerCase().split(/\s+/).filter(Boolean);
}

/**
 * @param {string} text   the block's text
 * @param {string} query  what was typed
 */
export function matches(text, query) {
  const haystack = text.toLowerCase();
  return queryWords(query).every((word) => haystack.includes(word));
}

/**
 * A text cut into runs, those that are one of the words marked as hits. Overlapping and touching hits join into one run.
 * @param {string} text
 * @param {string[]} words  lower case
 * @returns {{text: string, hit: boolean}[]}
 */
export function splitHits(text, words) {
  const lower = text.toLowerCase();
  const spans = [];
  for (const word of words) {
    for (let at = lower.indexOf(word); at >= 0; at = lower.indexOf(word, at + word.length)) spans.push([at, at + word.length]);
  }
  spans.sort((a, b) => a[0] - b[0]);
  const joined = [];
  for (const span of spans) {
    const last = joined[joined.length - 1];
    if (last && span[0] <= last[1]) last[1] = Math.max(last[1], span[1]);
    else joined.push([...span]);
  }
  const runs = [];
  let from = 0;
  for (const [start, end] of joined) {
    if (start > from) runs.push({ text: text.slice(from, start), hit: false });
    runs.push({ text: text.slice(start, end), hit: true });
    from = end;
  }
  if (from < text.length) runs.push({ text: text.slice(from), hit: false });
  return runs;
}
