/* ---------------------------------------------------------------------------
   graph/node_preview.js -- the picture a node made, drawn on the node itself.

   A node whose class declares an `image` display (Preview Image, Sampling Preview) shows the newest frame of its own stream
   `node:<uid>` under its fields. The element is made once per node and kept, so a node that redraws itself (a param changed)
   puts the same element back and the picture does not flash; the stream is joined while the node exists and released when it goes.
   --------------------------------------------------------------------------- */

import { createDisplay } from "../board/displays.js";

export class NodePreviews {
  /** @param {{live: import("../board/live.js").LiveStreams, shapes: {info: Function}}} deps */
  constructor({ live, shapes }) {
    this.live = live;
    this.shapes = shapes;
    this.entries = new Map(); // uid -> {el, display, handle}
  }

  /** The module of a node class that draws pictures, or null. */
  moduleOf(className) {
    return (this.shapes.info(className)?.modules || []).find((m) => m.kind === "image") || null;
  }

  /** The element to show on `node`'s card, or null for a node that makes no pictures. */
  elementFor(node) {
    const module = this.moduleOf(node.class_name);
    const known = this.entries.get(node.uid);
    if (!module) {
      if (known) this.forget(node.uid);
      return null;
    }
    if (known) return known.el;
    const el = document.createElement("div");
    el.className = "g2-preview";
    const entry = { el, display: null, handle: null };
    entry.handle = this.live.acquire(`node:${node.uid}`, () => entry.display?.update());
    entry.display = createDisplay({ module, state: entry.handle.state, ctx: {} });
    el.append(entry.display.el);
    entry.display.update();
    this.entries.set(node.uid, entry);
    return el;
  }

  forget(uid) {
    const entry = this.entries.get(uid);
    if (!entry) return;
    entry.handle.release();
    entry.display.destroy?.();
    this.entries.delete(uid);
  }
}
