/* ---------------------------------------------------------------------------
   lib/events.js -- the one way to subscribe to the domain-event stream.

   `/api/v1/events` has no replay (docs 07 F-09): a frame published while
   the tab was asleep, while the socket was reconnecting, or before the
   page subscribed is simply never delivered. So the database stays the
   source of truth and every (re)connect refetches it. That is the whole
   reason this module exists rather than three copies of `new
   EventSource(...)`:

     * **resync on every open**, including the first -- not just on
       reconnect. run.js skipped this entirely, so a `run_completed` it
       missed left the page saying "running" until a manual reload;
     * **parse failures are counted and said**, never swallowed. A silent
       drop is indistinguishable from a run that stopped reporting, which
       is the most expensive possible bug to debug (docs 07 F-03, review
       rule 5);
     * **notices are rate-limited**, because a stream that is broken in a
       * loop would otherwise fill the log with one line per frame and
       bury everything else. The count is still exact and is repeated in
       every notice.

   No view concepts here: the caller supplies `onEvent`, `onResync`,
   `onNotice` and `onError`, so this is testable with a fake EventSource
   and no DOM (frontend/tests/events.test.mjs).
   --------------------------------------------------------------------------- */

import { sse } from "../api.js";

/** How long to wait before repeating a notice about the same failure. */
export const NOTICE_INTERVAL_MS = 5000;

/**
 * Subscribe to `/events` (or any SSE path).
 *
 * @param {object} opts
 * @param {(e: object) => void} opts.onEvent       one parsed frame
 * @param {(reason: string) => void} [opts.onResync] on EVERY open, first included
 * @param {(message: string) => void} [opts.onNotice] unreadable frames, rate-limited
 * @param {(reason: string) => void} [opts.onError] transport trouble
 * @param {string} [opts.path]                      defaults to "/events"
 * @returns {{close: () => void, unreadable: () => number}}
 */
export function subscribeEvents({
  onEvent,
  onResync,
  onNotice,
  onError,
  path = "/events",
} = {}) {
  let unreadable = 0;
  let lastNotice = 0;

  const note = (reason) => {
    if (!onNotice) return;
    unreadable += 1;
    const now = Date.now();
    // Always the first one; after that at most one per interval, and
    // each carries the running total so nothing is under-reported.
    if (now - lastNotice >= NOTICE_INTERVAL_MS) {
      lastNotice = now;
      onNotice(
        `Unreadable event frame dropped (${unreadable} so far)` +
          (reason ? `: ${reason}` : "."),
      );
    }
  };

  const source = sse(path, {
    onMessage: (raw) => {
      let event;
      try {
        event = JSON.parse(raw.data);
      } catch (err) {
        note(err && err.message ? err.message : "not JSON");
        return;
      }
      if (onEvent) onEvent(event);
    },
    onOpen: () => {
      // Deliberately on every open, not only on reconnect: the first
      // open is the one that has to catch whatever was published while
      // this page was not listening.
      if (onResync) onResync("Event stream (re)connected — refetching.");
    },
    onError: () => {
      if (onError) onError("Event stream reconnecting…");
    },
  });

  return { close: () => source.close(), unreadable: () => unreadable };
}

/**
 * A slow refetch underneath a stream that never notices it is stale.
 *
 * Reconnects and resyncs handle most gaps; this covers the case where
 * nothing at all happens -- no error, no reconnect, just a frame that
 * never arrived.
 *
 * @param {() => void} fn
 * @param {number} ms
 * @returns {() => void}  stop()
 */
export function startSafetyPoll(fn, ms) {
  const timer = setInterval(fn, ms);
  return () => clearInterval(timer);
}