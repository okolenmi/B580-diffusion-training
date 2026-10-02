/* ---------------------------------------------------------------------------
   lib/format.js -- one definition of each formatter.

   These were copy-pasted between views and then edited independently,
   which is how `fmtRel` ended up answering "what is this timestamp?"
   with "—" in dashboard.js and "just now" in run.js: two functions with
   one name, differing only in the branch nobody exercised. The
   divergence is resolved here rather than preserved.

   The merged behaviour is the dashboard's "—" for a non-finite
   timestamp. "just now" is a claim about *time*, and an unparseable
   timestamp supports no such claim; it is the kind of sentence that
   reads as reassurance and is not. (In practice the branch is
   unreachable from fmtTime, which already handles a missing ISO string,
   so this is about the definition being defensible rather than about a
   visible change.)

   Pure functions, no DOM: that is what makes them testable under plain
   `node --test` with no browser (see frontend/tests/format.test.mjs).
   --------------------------------------------------------------------------- */

/**
 * A number the way the UI shows numbers: fixed decimals when there is
 * room, exponential when there is not.
 * @param {number|null|undefined} v
 */
export function fmtNum(v) {
  if (v === undefined || v === null) return "—";
  return v >= 1 ? v.toFixed(4) : v >= 0.001 ? v.toFixed(5) : v.toExponential(2);
}

/**
 * An elapsed time, at the largest unit that still says something.
 * @param {number} ms
 */
export function fmtDuration(ms) {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ${String(m % 60).padStart(2, "0")}m`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
}

/**
 * An epoch-ms timestamp, as "how long ago".
 * @param {number} ts  epoch milliseconds; NaN/undefined -> "—"
 */
export function fmtRel(ts) {
  if (!Number.isFinite(ts)) return "—";
  const s = Math.round((Date.now() - ts) / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

/**
 * An ISO timestamp, as a wall-clock string plus how long ago it was.
 * @param {string|null|undefined} iso
 */
export function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return `${d.toLocaleString()} (${fmtRel(d.getTime())})`;
}