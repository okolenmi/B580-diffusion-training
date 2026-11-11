/* ---------------------------------------------------------------------------
   graph2/monitor.js -- the live displays of the nodes on the canvas.

   A node class lists display modules in the catalog (the trainer: progress, loss, learning rate, step time, memory...).
   For every node of the open graph that has some, this panel keeps one section with a chip per module and the module's
   display (board/displays.js, the renderers the templates page already uses). A section reads the node's stream
   (`node:<uid>`, board/live.js shares one connection per stream), so what it shows survives collapsing the panel and
   a page reload mid-run: the server replays the stream's history on connect. Which modules a person switched on is a
   per-viewer convenience, kept per node class in this browser.
   --------------------------------------------------------------------------- */

import { createDisplay } from "../board/displays.js";
import { LiveStreams } from "../board/live.js";
import { h } from "../lib/dom.js";

const STORE = "graph2.monitor.modules";
const ON_BY_DEFAULT = 5; // a class lists its modules by priority: the first few are on until the person chooses

const remembered = () => {
  try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch { return {}; }
};
const remember = (all) => {
  try { localStorage.setItem(STORE, JSON.stringify(all)); } catch { /* storage may be unavailable */ }
};

class Section {
  constructor(monitor, node, info) {
    this.monitor = monitor;
    this.uid = node.uid;
    this.className = node.class_name;
    this.modules = info.modules;
    this.displays = new Map(); // module name -> {cell, display}
    this.title = h("span", { class: "mon-title" });
    this.chips = h("div", { class: "mon-chips" });
    this.grid = h("div", { class: "mon-grid" });
    this.el = h("section", { class: "mon-section" }, h("header", { class: "mon-head" }, this.title, this.chips), this.grid);
    const stream = monitor.live.acquire(`node:${node.uid}`, () => monitor.schedule(this));
    this.state = stream.state;
    this.release = stream.release;
    this.rename(node, info);
    this.render();
  }

  rename(node, info) {
    this.title.textContent = node.title || info.display_name || node.class_name;
  }

  chosen() {
    const saved = remembered()[this.className] || {};
    return new Set(this.modules.filter((m, n) => saved[m.name] ?? n < ON_BY_DEFAULT).map((m) => m.name));
  }

  toggle(name) {
    const all = remembered();
    const now = this.chosen();
    now.has(name) ? now.delete(name) : now.add(name);
    all[this.className] = Object.fromEntries(this.modules.map((m) => [m.name, now.has(m.name)]));
    remember(all);
    this.render();
  }

  render() {
    const chosen = this.chosen();
    this.chips.replaceChildren(...this.modules.map((m) => h("button", {
      class: "mon-chip", type: "button", text: m.title, title: m.doc || m.title,
      "aria-pressed": String(chosen.has(m.name)), onclick: () => this.toggle(m.name),
    })));
    for (const [name, entry] of this.displays) {
      if (chosen.has(name)) continue;
      entry.display.destroy();
      entry.cell.remove();
      this.displays.delete(name);
    }
    for (const module of this.modules) {
      if (!chosen.has(module.name) || this.displays.has(module.name)) continue;
      const display = createDisplay({ module, state: this.state, ctx: { execution: () => null } });
      const cell = h("div", { class: "mon-cell", style: `--w:${module.size[0]};--h:${module.size[1]}` },
        h("div", { class: "mon-cell-title", text: module.title }), h("div", { class: "mon-cell-body" }, display.el));
      this.displays.set(module.name, { cell, display });
    }
    // keep the declared order, whatever order they were switched on in
    this.grid.replaceChildren(...this.modules.filter((m) => this.displays.has(m.name)).map((m) => this.displays.get(m.name).cell));
    this.redraw();
  }

  redraw() {
    for (const { display } of this.displays.values()) display.update();
  }

  destroy() {
    for (const { display } of this.displays.values()) display.destroy();
    this.release();
    this.el.remove();
  }
}

export class Monitor {
  /** @param root where the sections go  @param doc the graph document  @param shapes the catalog's node infos */
  constructor(root, doc, shapes, { live = new LiveStreams() } = {}) {
    this.root = root;
    this.doc = doc;
    this.shapes = shapes;
    this.live = live;
    this.sections = new Map(); // uid -> Section
    this.empty = h("div", { class: "console-line info", text: "No node of this graph has live displays yet (add a Diffusion Trainer)." });
    this._queued = new Set();
    this._frame = 0;
    for (const type of ["reset", "node:add", "node:remove", "node:title"]) doc.on(type, () => this.sync());
    this.sync();
  }

  /** Redraw at most once per animation frame, however fast the frames arrive. */
  schedule(section) {
    this._queued.add(section);
    if (this._frame) return;
    this._frame = requestAnimationFrame(() => {
      this._frame = 0;
      const due = [...this._queued];
      this._queued.clear();
      for (const s of due) s.redraw();
    });
  }

  sync() {
    const wanted = new Map();
    for (const node of this.doc.nodes.values()) {
      const info = this.shapes.info(node.class_name);
      if (info?.modules?.length) wanted.set(node.uid, { node, info });
    }
    for (const [uid, section] of this.sections) {
      const keep = wanted.get(uid);
      if (keep && keep.node.class_name === section.className) { section.rename(keep.node, keep.info); continue; }
      section.destroy();
      this.sections.delete(uid);
    }
    for (const [uid, { node, info }] of wanted) {
      if (!this.sections.has(uid)) this.sections.set(uid, new Section(this, node, info));
    }
    this.root.replaceChildren(...(this.sections.size ? [...this.sections.values()].map((s) => s.el) : [this.empty]));
  }

  /** The panel was opened: charts measured while hidden are drawn again. */
  shown() {
    for (const s of this.sections.values()) s.redraw();
  }
}
