/* ---------------------------------------------------------------------------
   lib/events.js -- the one way to subscribe to the domain-event stream.

   `/api/v1/events` **replays** now: the server numbers every event and
   rings the lifecycle ones, so a client that reconnects is handed what it
   missed (`docs/design/backend/09-event-contract.md`). That is the whole
   reason this module exists rather than three copies of `new
   EventSource(...)`:

     * **the refetch is driven by the server's answer, not by `onOpen`.**
       A socket being open says nothing about whether the client holds
       current data -- a reconnect that replayed 4 events and closed the
       gap needs no refetch, and one that could not needs a full one.
       The server says which, in `stream_opened.resync_required`. Before
       that this module refetched on every open as a workaround for the
       missing replay, which cost a full fetch per reconnect and was
       still racy (an event between the refetch and the subscription was
       lost);
     * **an absent flag means refetch.** The check is `!== false`, not
       `=== true`, so a server that does not send the flag -- an older
       one, a proxy that eats the frame -- degrades to the old
       always-refetch behaviour instead of to never refetching;
     * **parse failures are counted and said**, never swallowed. A silent
       drop is indistinguishable from a run that stopped reporting, which
       is the most expensive possible bug to debug (docs 07 F-03, review
       rule 5);
     * **notices are rate-limited**, because a stream that is broken in a
       * loop would otherwise fill the log with one line per frame and
       bury everything else. The count is still exact and is repeated in
       every notice.

   `stream_opened` is *also* forwarded to `onEvent`, so a view that wants
   the watermark can read it; the resync decision is made here so the
   three call sites cannot disagree about it.

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
 * @param {(reason: string) => void} [opts.onResync]
 *        when the server could not replay the gap -- first connect
 *        included, and any reconnect it cannot cover. NOT on every open.
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
    // No onOpen: the socket being open is not the same fact as this
    // client holding current data, and the old handler conflated them.
    onMessage: (raw) => {
      let event;
      try {
        event = JSON.parse(raw.data);
      } catch (err) {
        note(err && err.message ? err.message : "not JSON");
        return;
      }
      if (event.type === "stream_opened") {
        // Resync *before* forwarding, so a view's handler runs against
        // already-current data and cannot observe a half-applied frame.
        //
        // `!== false` and not `=== true`: an older server, or a proxy
        // that swallows this frame, must fall back to refetching.
        // Treating "I do not know" as "I am current" is the failure that
        // leaves a page saying a finished run is still running.
        if (event.resync_required !== false && onResync) {
          const reason = event.resync_required
            ? "Event stream could not replay missed events — refetching."
            : "Event stream (re)connected — refetching.";
          onResync(reason);
        }
      }
      if (onEvent) onEvent(event);
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