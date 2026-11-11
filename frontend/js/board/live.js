/* ---------------------------------------------------------------------------
   board/live.js -- one connection per stream, shared by every block on it.

   `LiveStreams.acquire(id, onChange)` opens `/api/v1/monitor/<id>/stream` the
   first time anyone wants it, hands back the shared `StreamState`, and closes
   the connection when the last block lets go. Blocks never hold sockets, so
   collapsing, removing or re-adding one costs nothing and a page with five
   views of one trainer opens one connection, not five.

   `onChange(id)` is called after every frame that changed state; the board
   coalesces those into at most one redraw per animation frame.

   A scope (an execution id) points every stream at that run's own copy of it, `run:<id>:<stream>`: runs alive at once
   share node uids, so a page that looks at one run must not read the plain streams, which carry the latest start.
   Changing the scope reconnects the open streams in place; the shared states stay the same objects and refill from the
   server's replay.

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
    this._scope = null;
  }

  /** Which run the streams show (null: the plain streams, the latest start). */
  get scope() { return this._scope; }

  setScope(scope) {
    if (scope === this._scope) return;
    this._scope = scope;
    for (const [id, entry] of this._streams) {
      try { entry.source.close(); } catch { /* already closed */ }
      entry.state.reset();
      entry.state.setConn("connecting");
      entry.source = this._connect(id, entry);
      entry.listeners.forEach((fn) => fn(id));
    }
  }

  _connect(id, entry) {
    const busId = this._scope == null ? id : `run:${this._scope}:${id}`;
    const notify = () => entry.listeners.forEach((fn) => fn(id));
    return this._open(busId, {
      onMessage: (ev) => {
        const r = entry.state.applyRaw(ev.data, this._now());
        if (r.kind !== "ignored") notify();
      },
      onError: () => { entry.state.setConn("disconnected"); notify(); },
    });
  }

  /** Connect (or join) a stream. Returns `{state, release}`. */
  acquire(id, onChange) {
    let entry = this._streams.get(id);
    if (!entry) {
      entry = { state: new StreamState(), source: null, listeners: new Set() };
      this._streams.set(id, entry);
      entry.source = this._connect(id, entry);
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
