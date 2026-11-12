/* ---------------------------------------------------------------------------
   graph2/draft.js -- the canvas outlives a reload: the open document is kept
   in this browser (debounced) and restored on the next visit. Saved graphs
   live in the library; this is only "what was on the canvas". Storage can be
   missing or full (private mode): then nothing is kept and nothing breaks.
   --------------------------------------------------------------------------- */

import { GRAPH_FORMAT } from "./doc.js";

const KEY = "g2.draft.v1";

export function readDraft(storage = globalThis.localStorage) {
  try {
    const raw = storage?.getItem(KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed && parsed.format === GRAPH_FORMAT && Array.isArray(parsed.nodes) && parsed.nodes.length ? parsed : null;
  } catch {
    return null;
  }
}

export class DraftSaver {
  constructor(doc, { storage = globalThis.localStorage, wait = 800, timers = { set: (fn, ms) => setTimeout(fn, ms), clear: (id) => clearTimeout(id) } } = {}) {
    this.doc = doc;
    this.storage = storage;
    this.wait = wait;
    this.timers = timers;
    this._timer = null;
    this._held = false;
    doc.on("change", () => this.notify());
  }

  /** While held nothing is kept: the canvas shows something that is not the user's own work (a frozen run). */
  hold(on) {
    this._held = on;
    if (on && this._timer) { this.timers.clear(this._timer); this._timer = null; }
  }

  notify() {
    if (this._held) return;
    if (this._timer) this.timers.clear(this._timer);
    this._timer = this.timers.set(() => this.flush(), this.wait);
  }

  flush() {
    if (this._held) return;
    if (this._timer) this.timers.clear(this._timer);
    this._timer = null;
    try {
      if (!this.doc.nodes.size) this.storage.removeItem(KEY);
      else this.storage.setItem(KEY, JSON.stringify(this.doc.toJSON()));
    } catch {
      /* private mode or quota: the draft is a convenience */
    }
  }
}
