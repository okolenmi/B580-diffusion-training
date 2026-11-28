/* What the console shows from the journal: which entries pass the filter and the clear mark, and how one is written. Pure: no DOM, network or clock. */

/** Filter name -> the levels it lets through. */
export const FILTERS = {
  all: { label: "Everything", levels: ["info", "success", "warn", "error"] },
  problems: { label: "Warnings and errors", levels: ["warn", "error"] },
  errors: { label: "Errors only", levels: ["error"] },
};

export const DEFAULT_FILTER = "all";

/** The entries to draw: the filter's levels, and nothing at or before the number the person cleared up to. Local entries (not numbered) always show. */
export function visible(entries, filter, clearedTo) {
  const levels = (FILTERS[filter] || FILTERS[DEFAULT_FILTER]).levels;
  return entries.filter((e) => levels.includes(e.level) && (e.seq === null || e.seq > clearedTo));
}

const two = (n) => String(n).padStart(2, "0");

/** "14:02:11" in the browser's own clock; the date too when it is not today. */
export function stamp(at, now = new Date()) {
  const d = new Date(at * 1000);
  const time = `${two(d.getHours())}:${two(d.getMinutes())}:${two(d.getSeconds())}`;
  return d.toDateString() === now.toDateString() ? time : `${two(d.getMonth() + 1)}-${two(d.getDate())} ${time}`;
}

/** One entry as plain text, for Copy. */
export function asText(entry, now = new Date()) {
  return `${stamp(entry.at, now)}  ${entry.level.toUpperCase().padEnd(7)} ${entry.source.padEnd(8)} ${entry.text}`;
}

/** How many of `batch` are warnings or errors: what the docked button counts as unread. */
export function problemCount(batch) {
  return batch.filter((e) => e.level === "warn" || e.level === "error").length;
}
