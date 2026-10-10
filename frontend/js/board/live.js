/* ---------------------------------------------------------------------------
   board/live.js -- one connection per stream, shared by every block on it.

   `LiveStreams.acquire(id, onChange)` opens `/api/v1/monitor/<id>/stream` the
   first time anyone wants it, hands back the shared `StreamState`, and closes
   the connection when the last block lets go. Blocks never hold sockets, so
   collapsing, removing or re-adding one costs nothing and a page with five
   views of one trainer opens one connection, not five.

   `onChange(id)` is called after every frame that changed state; the board
   coalesces those into at most one redraw per animation frame.

   The EventSource is injectable so the lifecycle is testable without a
   browser (frontend/tests/board_live.test.mjs).
   --------------------------------------------------------------------------- */

import { StreamState } from "./series.js";

const defaultOpen = (id, handlers) => {
  const source = new EventSource(`/api/v1/monitor/${encodeURIComponent(id)}/stream`);
  source.onmessage = handlers.onMessage;
  source.onerror = handlers.onError;
  return source;
};

export class LiveStreams {
  /** @param {{open?: Function, now?: () => number}} [opts] */
  constructor({ open = defaultOpen, now = () => Date.now() } = {}) {
    this._open = open;
    this._now = now;
    this._streams = new Map(); // id -> {state, source, listeners:Set}
  }

  /** Connect (or join) a stream. Returns `{state, release}`. */
  acquire(id, onChange) {
    let entry = this._streams.get(id);
    if (!entry) {
      const state = new StreamState();
      entry = { state, source: null, listeners: new Set() };
      this._streams.set(id, entry);
      const notify = () => entry.listeners.forEach((fn) => fn(id));
      entry.source = this._open(id, {
        onMessage: (ev) => {
          const r = state.applyRaw(ev.data, this._now());
          if (r.kind !== "ignored") notify();
        },
        onError: () => { state.setConn("disconnected"); notify(); },
      });
    }
    entry.listeners.add(onChange);
    let released = false;
    return {
      state: entry.state,
      release: () => {
        if (released) return;
        released = true;
        entry.listeners.delete(onChange);
        if (entry.listeners.size === 0) {
          try { entry.source.close(); } catch { /* already closed */ }
          this._streams.delete(id);
        }
      },
    };
  }

  /** Streams currently open (for tests and the page's connection line). */
  get openIds() { return [...this._streams.keys()]; }

  closeAll() {
    for (const entry of this._streams.values()) {
      try { entry.source.close(); } catch { /* already closed */ }
    }
    this._streams.clear();
  }
}
