/* ---------------------------------------------------------------------------
   lib/memory_panel.js -- what the graph's memory settings panel says.

   Pure functions, no DOM: the same rule lib/format.js and lib/value.js
   follow, so this is testable under plain `node --test` (see
   frontend/tests/memory_panel.test.mjs) and the panel itself stays a
   thin caller that only ever writes through textContent.

   The data comes from POST /graphs/memory-preview, which runs the same
   arithmetic admission runs. That is deliberate and load-bearing: a
   panel with its own idea of "will it fit" would eventually disagree
   with the run endpoint, and a panel that says "fits" and then refuses
   the run is worse than no panel.

   Three unknowns are load-bearing here and none of them is a zero:

     * no fingerprint   -> this graph's peak cannot be keyed, so there
                          is no remembered measurement to show;
     * no remembered peak-> the configuration has never been run, which
                          is a different fact from "it needs nothing";
     * no ledger        -> the device total was never read, so nothing
                          can be checked and nothing should be implied.

   Every one of them renders as its own word. None renders as 0 MB.
   --------------------------------------------------------------------------- */

/** The one definition of "MB, or nothing". */
const UNKNOWN = "—";

/**
 * An MB quantity as the panel shows it: thousands-separated, or an
 * em dash when it is genuinely unknown.
 *
 * Non-finite is treated as unknown rather than rendered. A NaN reaching
 * this panel means something upstream divided by zero or read garbage;
 * printing "NaN MB" in a fit indicator invites the reader to treat it
 * as a number, and printing "0" would be a claim nothing supports.
 * @param {number|null|undefined} v
 * @returns {string}
 */
export function fmtMb(v) {
  if (typeof v !== "number" || !Number.isFinite(v)) return UNKNOWN;
  return `${Math.round(v).toLocaleString("en-US")} MB`;
}

/**
 * One VRAM setting as "label: value".
 *
 * `vram_max_mb` is the field that can be the string "auto" rather than
 * a number, and "auto" is a real answer ("everything that is free"), so
 * it renders as itself. Anything else non-numeric is unknown -- an
 * unrecognised string is a value this panel cannot describe, and
 * echoing it back would read as if it meant something.
 * @param {number|string|null|undefined} v
 */
export function fmtSetting(v) {
  if (v === "auto") return "auto";
  return fmtMb(v);
}

/**
 * The remembered peak, in the words the panel leads with.
 *
 * An unknown fingerprint and a known-but-unmeasured one are different
 * facts and get different sentences: the first is "this graph's peak
 * cannot be keyed", the second is "nothing has run in this shape yet".
 * Collapsing them would tell a user their configuration has been
 * measured when it never has.
 * @param {{fingerprint_key?: string|null, peak_mb?: number|null}} preview
 * @returns {string}
 */
export function describePeak(preview) {
  if (!preview) return "No measurement yet.";
  if (!preview.fingerprint_key) {
    return "This graph's peak cannot be keyed, so nothing is remembered for it.";
  }
  if (typeof preview.peak_mb !== "number" || !Number.isFinite(preview.peak_mb)) {
    return "Never measured — nothing has run in this configuration yet.";
  }
  return `Last time: ${fmtMb(preview.peak_mb)} (peak of the device allocator).`;
}

/**
 * The "will it fit" verdict: a label, a severity for the CSS, and the
 * sentence under it.
 *
 * Four verdicts, not two, because "unknown" and "exploratory" are
 * neither a yes nor a no and reporting either as a yes would be the
 * wrong kind of reassuring:
 *
 *   fits          -- admission would admit it right now
 *   does_not_fit  -- it would be refused, with the numbers
 *   exploratory   -- demand is unknown: it would claim the whole card
 *                    and start only if nothing else holds it
 *   unknown       -- no ledger yet, so there is nothing to check against
 *
 * @param {object} preview  the /graphs/memory-preview response
 * @returns {{verdict: string, level: string, headline: string, detail: string}}
 */
export function describeFit(preview) {
  const verdict = (preview && preview.verdict) || "unknown";
  const need = fmtMb(preview ? preview.device_demand_mb : null);
  const free = fmtMb(preview ? preview.free_mb : null);

  if (verdict === "fits") {
    return {
      verdict,
      level: "ok",
      headline: "Fits",
      detail: `Needs ${need}; ${free} is free.`,
    };
  }
  if (verdict === "does_not_fit") {
    return {
      verdict,
      level: "bad",
      headline: "Will not fit",
      detail: `Needs ${need}; only ${free} is free.`,
    };
  }
  if (verdict === "exploratory") {
    return {
      verdict,
      level: "warn",
      headline: "Unknown — would take the whole card",
      detail: "This configuration has never been measured, so a run would claim "
        + "everything free and start only if nothing else is using the card.",
    };
  }
  return {
    verdict: "unknown",
    level: "warn",
    headline: "Cannot tell yet",
    detail: "The device's total memory has not been read, so nothing can be checked.",
  };
}

/**
 * Who holds the card, for the panel's holder list.
 *
 * Sorted largest-first, because the question a reader has is "what is in
 * my way", and the answer is nearly always the biggest claimant. An
 * empty ledger renders as a single honest line rather than an empty box.
 * @param {Record<string, {mb?: number}>} holders
 * @returns {Array<{owner: string, mb: string}>}
 */
export function describeHolders(holders) {
  const entries = Object.entries(holders || {});
  if (!entries.length) return [];
  return entries
    .map(([owner, held]) => [owner, (held && held.mb) || 0])
    .sort((a, b) => b[1] - a[1])
    .map(([owner, mb]) => ({ owner, mb: fmtMb(mb) }));
}

/**
 * A refused start, as the panel shows it.
 *
 * This is the breakdown the run endpoint already returns in its 409 --
 * the same numbers, rendered rather than re-derived. The point of
 * showing it in full is that a refusal naming only "could not start"
 * gives a user nothing to act on; this one names what was asked, what
 * was free, and who holds the rest.
 *
 * A missing or malformed breakdown renders as the reason alone, still
 * honest: the reason is the part that always arrives.
 * @param {object|null} breakdown  error details from the run endpoint
 * @param {string} [fallbackReason]
 * @returns {{headline: string, rows: Array<[string, string]>, holders: Array}}
 */
export function describeRefusal(breakdown, fallbackReason = "") {
  const detail = breakdown || {};
  const headline = detail.reason || fallbackReason || "The run was refused.";
  const rows = [];
  if (typeof detail.requested_mb === "number") {
    rows.push(["Asked for", fmtMb(detail.requested_mb)]);
  }
  if (typeof detail.free_mb === "number") {
    rows.push(["Free", fmtMb(detail.free_mb)]);
  }
  if (typeof detail.capacity_mb === "number") {
    rows.push(["Capacity", fmtMb(detail.capacity_mb)]);
  }
  if (typeof detail.foreign_reserve_mb === "number") {
    rows.push(["Left to other users", fmtMb(detail.foreign_reserve_mb)]);
  }
  // Holders arrive as {owner: mb} from the ledger's own breakdown,
  // unlike the snapshot's {owner: {mb}} -- both shapes are accepted so
  // a caller can pass either without this having to care which it got.
  const holders = Object.entries(detail.holders || {})
    .map(([owner, held]) => [owner, typeof held === "object" ? held.mb : held])
    .filter(([, mb]) => typeof mb === "number" && Number.isFinite(mb))
    .sort((a, b) => b[1] - a[1])
    .map(([owner, mb]) => ({ owner, mb: fmtMb(mb) }));

  return { headline, rows, holders };
}

/**
 * The whole panel as data: one object the DOM code walks.
 *
 * Returning a description rather than building nodes keeps every
 * judgment that can be wrong -- what an unknown means, what order
 * holders go in -- here, under test, instead of inside a click handler.
 * @param {object} preview
 * @returns {{settings: Array<[string,string]>, peak: string,
 *            fit: object, holders: Array}}
 */
export function panelModel(preview) {
  // The original is passed through to the two describers, not the `|| {}`
  // stand-in: with no response at all we do not know whether the
  // fingerprint is keyable, and "cannot be keyed" would assert a fact
  // nobody has established yet.
  const p = preview || {};
  return {
    settings: [
      ["Min VRAM", fmtSetting(p.vram_min_mb)],
      ["Max VRAM", fmtSetting(p.vram_max_mb)],
      ["Demand", p.demand_source === "unknown"
        ? "unknown (never measured)"
        : `${fmtMb(p.demand_mb)} (${p.demand_source || "unknown"})`],
      ["Device total", fmtMb(p.capacity_mb)],
      ["Free now", fmtMb(p.free_mb)],
    ],
    peak: describePeak(preview),
    fit: describeFit(preview),
    holders: describeHolders(p.holders),
  };
}