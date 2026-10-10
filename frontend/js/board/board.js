/* ---------------------------------------------------------------------------
   board/board.js -- the grid of blocks, for both a template page and a
   monitor window (one engine, two hosts).

   What it owns: turning a template document + the backend's modules answer
   into DOM -- a CSS grid whose cells are the document's `x/y/w/h` -- and, in
   edit mode, drag, resize, collapse, rename and remove. What it does not
   own: the document (the page passes it in and is told when it changed),
   the network (the page fetches modules/resolutions and hands them over)
   and the arithmetic (board/layout.js, board/model.js, board/series.js --
   all pure and unit-tested).

   Two modes, one document:
     use   -- the blocks are the interface. Fields are editable, displays
              are live, nothing moves. (Collapsing a block is still allowed:
              it hides what you do not want to look at.)
     edit  -- the layout is editable too. Headers become drag handles, the
              corner resizes, blocks can be renamed or removed.

   Blocks are created once and then *re-placed* (a style change), never
   rebuilt, when the layout moves: a chart keeps its data, a half-typed
   field keeps its focus. A block's content is rebuilt only when what it
   points at actually changed (`contentKey`).

   On a narrow screen (< NARROW_PX) the grid collapses to one column in
   reading order and drag is disabled -- a phone is for *using* a page.
   --------------------------------------------------------------------------- */

import { createDisplay } from "./displays.js";
import { buildFieldControl } from "./fields.js";
import {
  cellDelta, columnWidth, effH, flowOrder, keyNudge, moveTo, resizeTo, resolve, setCollapsed,
} from "./layout.js";
import { addBlock, findTarget, patchBlock, removeBlock, titleFor } from "./model.js";

export const ROW_PX = 48;
export const GAP_PX = 10;
export const NARROW_PX = 640;

const STATUS_TEXT = {
  wired: "set by the graph",
  missing_node: "missing",
  unknown_class: "missing",
  missing_module: "missing",
};

const h = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

function button(label, title, cls = "") {
  const b = h("button", `bd-tool ${cls}`.trim(), label);
  b.type = "button";
  b.title = title;
  b.setAttribute("aria-label", title);
  return b;
}

export class Board {
  /**
   * @param {HTMLElement} container
   * @param {object} opts
   * @param {import("./live.js").LiveStreams} opts.live
   * @param {object} [opts.ctx]       passed to displays (`ctx.execution()`)
   * @param {(reason: string) => void} [opts.onChange]  "layout" | "blocks" | "value" | "title"
   * @param {(msg: string, kind?: string) => void} [opts.onNote]
   */
  constructor(container, { live, ctx = {}, onChange = () => {}, onNote = () => {} }) {
    this.container = container;
    this.live = live;
    this.ctx = ctx;
    this.onChange = onChange;
    this.onNote = onNote;
    this.doc = null;
    this.modules = null;
    this.statuses = {};
    this.mode = "use";
    this.entries = new Map(); // block id -> entry
    this.narrow = false;
    this._pending = new Set();
    this._raf = 0;

    this.root = h("div", "bd-grid");
    this.empty = h("div", "bd-grid-empty");
    container.append(this.root, this.empty);

    this._ro = new ResizeObserver(() => {
      const narrow = container.clientWidth < NARROW_PX;
      if (narrow !== this.narrow) { this.narrow = narrow; this.applyLayout(); }
    });
    this._ro.observe(container);
    this.narrow = container.clientWidth > 0 && container.clientWidth < NARROW_PX;

    // Elapsed-time readouts (run status) tick without any stream frame.
    this._tick = setInterval(() => {
      for (const e of this.entries.values()) if (e.tick) e.display.update();
    }, 1000);
  }

  destroy() {
    clearInterval(this._tick);
    cancelAnimationFrame(this._raf);
    this._ro.disconnect();
    for (const e of this.entries.values()) this._disposeContent(e);
    this.entries.clear();
    this.container.replaceChildren();
  }

  /* ---- inputs ---------------------------------------------------------- */

  setDoc(doc) {
    this.doc = doc;
    for (const e of this.entries.values()) { this._disposeContent(e); e.el.remove(); }
    this.entries.clear();
    this.sync();
  }

  setModules(modules) { this.modules = modules; this.sync(); }

  /** @param {Record<string, {status: string, detail: string}>} statuses */
  setStatuses(statuses) { this.statuses = statuses || {}; this.sync(); }

  setMode(mode) {
    this.mode = mode;
    this.root.classList.toggle("bd-edit", mode === "edit");
    this.applyLayout();
    this._updateEmpty();
  }

  /** Redraw every display now (e.g. after the run state changed). */
  refreshDisplays() {
    for (const e of this.entries.values()) if (e.display) e.display.update();
  }

  /** Re-evaluate field gating (a value elsewhere may have moved a gate). */
  refreshFields() {
    for (const e of this.entries.values()) if (e.field) e.field.refresh();
  }

  /* ---- document operations the page calls ------------------------------ */

  addRef(ref) {
    const before = new Set(this.doc.blocks.map((b) => b.id));
    const next = addBlock(this.doc, ref, this.modules);
    if (next === this.doc) return null;
    this.doc.blocks = next.blocks;
    this.sync();
    this.onChange("blocks");
    const added = this.doc.blocks.find((b) => !before.has(b.id)) || null;
    const entry = added && this.entries.get(added.id);
    if (entry) entry.el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    return added;
  }

  removeId(id) {
    this.doc.blocks = removeBlock(this.doc, id).blocks;
    this.sync();
    this.onChange("blocks");
  }

  /* ---- build / sync ---------------------------------------------------- */

  /** Make the DOM match the document: create new blocks, drop removed
      ones, rebuild content whose target changed, re-place everything. */
  sync() {
    if (!this.doc) return;
    const ids = new Set(this.doc.blocks.map((b) => b.id));
    for (const [id, e] of this.entries) {
      if (!ids.has(id)) { this._disposeContent(e); e.el.remove(); this.entries.delete(id); }
    }
    for (const block of this.doc.blocks) {
      let entry = this.entries.get(block.id);
      if (!entry) {
        entry = this._createEntry(block);
        this.entries.set(block.id, entry);
        this.root.append(entry.el);
      }
      entry.block = block;
      this._renderHeader(entry);
      const key = this._contentKey(block);
      if (entry.contentKey !== key) this._buildContent(entry, key);
    }
    this.applyLayout();
    this._updateEmpty();
  }

  _contentKey(block) {
    const st = this.statuses[block.id];
    const { node, descriptor } = findTarget(block.ref, this.modules);
    // value/has_value change with every keystroke; they must not rebuild a
    // control the person is typing into.
    const stable = descriptor
      ? JSON.stringify(descriptor, (k, v) => (k === "value" || k === "has_value" ? undefined : v))
      : "none";
    return `${st ? st.status : "ok"}|${this.modules ? "m" : "-"}|${node ? node.stream : ""}|${stable}`;
  }

  _createEntry(block) {
    const el = h("article", "bd-block");
    el.dataset.id = block.id;
    el.dataset.kind = block.ref.kind;
    const head = h("header", "bd-head");
    const grip = h("span", "bd-grip", "⋮⋮");
    grip.setAttribute("aria-hidden", "true");
    const titles = h("div", "bd-titles");
    const title = h("span", "bd-title");
    const sub = h("span", "bd-sub");
    titles.append(title, sub);
    const badge = h("span", "bd-badge");
    badge.hidden = true;
    const tools = h("div", "bd-tools");
    const collapse = button("▾", "Collapse / expand");
    const rename = button("✎", "Rename this block", "bd-edit-only");
    const remove = button("✕", "Remove this block from the page", "bd-edit-only");
    tools.append(collapse, rename, remove);
    head.append(grip, titles, badge, tools);
    const body = h("div", "bd-body");
    const resizer = h("div", "bd-resize");
    resizer.title = "Drag to resize";
    el.append(head, body, resizer);

    const entry = {
      el, head, title, sub, badge, body, block, contentKey: null,
      display: null, field: null, release: null, tick: false, lastUpdate: 0,
    };

    collapse.addEventListener("click", () => this._toggleCollapse(entry.block.id));
    remove.addEventListener("click", () => this.removeId(entry.block.id));
    rename.addEventListener("click", () => this._startRename(entry));
    title.addEventListener("dblclick", () => { if (this.mode === "edit") this._startRename(entry); });
    head.addEventListener("pointerdown", (ev) => this._startDrag(ev, entry, "move"));
    resizer.addEventListener("pointerdown", (ev) => this._startDrag(ev, entry, "resize"));
    head.tabIndex = 0;
    head.addEventListener("keydown", (ev) => this._onKey(ev, entry));
    return entry;
  }

  _renderHeader(entry) {
    const { block } = entry;
    const t = titleFor(block, this.modules);
    if (!entry.renaming) entry.title.textContent = t.title;
    entry.sub.textContent = t.subtitle;
    const st = this.statuses[block.id];
    const label = st && st.status !== "ok" ? STATUS_TEXT[st.status] || st.status : "";
    entry.badge.hidden = !label;
    entry.badge.textContent = label;
    entry.badge.title = st ? st.detail : "";
    entry.badge.dataset.status = st ? st.status : "";
    entry.el.dataset.status = st ? st.status : "ok";
    entry.el.classList.toggle("is-collapsed", !!block.collapsed);
    entry.body.hidden = !!block.collapsed;
  }

  _disposeContent(entry) {
    if (entry.display) { entry.display.destroy(); entry.display = null; }
    if (entry.release) { entry.release(); entry.release = null; }
    entry.field = null;
    entry.tick = false;
    entry.body.replaceChildren();
  }

  _buildContent(entry, key) {
    this._disposeContent(entry);
    entry.contentKey = key;
    const { block } = entry;
    const st = this.statuses[block.id];
    const status = st ? st.status : "ok";

    if (!this.modules) { entry.body.append(h("div", "bd-empty", "loading…")); return; }

    const { node, descriptor } = findTarget(block.ref, this.modules);
    if (status.startsWith("missing") || status === "unknown_class" || !descriptor) {
      const box = h("div", "bd-orphan");
      box.append(
        h("strong", "", "This block has lost its target."),
        h("span", "", st && st.detail ? st.detail : "The graph no longer has what this block shows."),
      );
      if (this.mode === "edit") box.append(h("span", "bd-orphan-hint", "Remove it with ✕, or restore the node in the graph."));
      entry.body.append(box);
      return;
    }

    if (block.ref.kind === "field") {
      if (status === "wired") {
        entry.body.append(h("div", "bd-wired",
          "This value comes from another node in the graph, so it is not set here."));
        return;
      }
      const siblings = node ? node.fields : this.modules.graph_fields;
      const field = buildFieldControl({
        block, descriptor, siblings, doc: this.doc,
        onChange: (origin) => { this.refreshFields(); this.onChange(origin); },
        onNote: this.onNote,
      });
      entry.field = field;
      entry.body.append(field.el);
      return;
    }

    // display
    const streamId = descriptor.stream || (node && node.stream) || null;
    let state = null;
    if (streamId) {
      const handle = this.live.acquire(streamId, () => this._schedule(entry));
      entry.release = handle.release;
      state = handle.state;
    }
    entry.display = createDisplay({
      module: descriptor, state: state || emptyState(), ctx: this.ctx,
    });
    entry.tick = descriptor.kind === "status";
    entry.body.append(entry.display.el);
    entry.display.update();
  }

  /* ---- throttled redraw ------------------------------------------------ */

  _schedule(entry) {
    this._pending.add(entry);
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = 0;
      const now = performance.now();
      for (const e of [...this._pending]) {
        if (!e.display || e.block.collapsed) { this._pending.delete(e); continue; }
        const min = e.block.ref.kind === "display" && e.display.el.querySelector("canvas") ? 250 : 80;
        if (now - e.lastUpdate >= min) {
          e.lastUpdate = now;
          this._pending.delete(e);
          e.display.update();
        }
      }
      if (this._pending.size) this._schedule([...this._pending][0]);
    });
  }

  /* ---- layout ---------------------------------------------------------- */

  applyLayout() {
    if (!this.doc) return;
    const cols = this.narrow ? 1 : this.doc.columns;
    this.root.style.setProperty("--bd-cols", String(cols));
    this.root.style.setProperty("--bd-row", `${ROW_PX}px`);
    this.root.style.setProperty("--bd-gap", `${GAP_PX}px`);
    this.root.classList.toggle("bd-narrow", this.narrow);
    const order = new Map(flowOrder(this.doc.blocks).map((b, i) => [b.id, i]));
    for (const b of this.doc.blocks) {
      const e = this.entries.get(b.id);
      if (!e) continue;
      e.block = b;
      if (this.narrow) {
        e.el.style.gridColumn = "1";
        e.el.style.gridRow = `auto / span ${effH(b)}`;
        e.el.style.order = String(order.get(b.id));
      } else {
        e.el.style.gridColumn = `${b.x + 1} / span ${b.w}`;
        e.el.style.gridRow = `${b.y + 1} / span ${effH(b)}`;
        e.el.style.order = "";
      }
      e.el.classList.toggle("is-collapsed", !!b.collapsed);
      e.body.hidden = !!b.collapsed;
    }
  }

  _updateEmpty() {
    const none = !this.doc || this.doc.blocks.length === 0;
    this.empty.hidden = !none;
    this.empty.textContent = this.mode === "edit"
      ? "Nothing here yet. Add blocks from the list on the right."
      : "This page is empty. Switch to Edit to add blocks.";
  }

  _toggleCollapse(id) {
    const b = this.doc.blocks.find((x) => x.id === id);
    if (!b) return;
    this.doc.blocks = setCollapsed(this.doc.blocks, id, !b.collapsed, this.doc.columns);
    this.applyLayout();
    this.onChange("layout");
  }

  /* ---- rename ---------------------------------------------------------- */

  _startRename(entry) {
    if (entry.renaming) return;
    entry.renaming = true;
    const { block } = entry;
    const input = h("input", "bd-rename");
    input.type = "text";
    input.maxLength = 120;
    input.value = entry.title.textContent;
    input.setAttribute("aria-label", "Block title");
    entry.title.replaceChildren(input);
    input.focus();
    input.select();
    const finish = (commit) => {
      if (!entry.renaming) return;
      entry.renaming = false;
      const text = input.value.trim();
      if (commit) {
        const t = titleFor({ ...block, title: null }, this.modules);
        const next = text && text !== t.defaultTitle ? text : null;
        if (next !== (block.title || null)) {
          this.doc.blocks = patchBlock(this.doc, block.id, { title: next }).blocks;
          this.onChange("title");
        }
      }
      this.sync();
    };
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); finish(true); }
      else if (ev.key === "Escape") { ev.preventDefault(); finish(false); }
      ev.stopPropagation();
    });
    input.addEventListener("blur", () => finish(true));
    input.addEventListener("pointerdown", (ev) => ev.stopPropagation());
  }

  /* ---- drag / resize / keyboard (edit mode) ----------------------------- */

  _startDrag(ev, entry, what) {
    if (this.mode !== "edit" || this.narrow || ev.button !== 0) return;
    if (ev.target.closest(".bd-tool, .bd-rename")) return;
    ev.preventDefault();
    const id = entry.block.id;
    const original = this.doc.blocks.map((b) => ({ ...b }));
    const start = original.find((b) => b.id === id);
    const width = this.root.clientWidth;
    const colW = columnWidth(width, this.doc.columns, GAP_PX);
    const x0 = ev.clientX, y0 = ev.clientY;
    const target = ev.currentTarget;
    target.setPointerCapture(ev.pointerId);
    entry.el.classList.add("is-dragging");

    const move = (e2) => {
      const d = cellDelta(e2.clientX - x0, e2.clientY - y0, colW, ROW_PX, GAP_PX);
      this.doc.blocks = what === "move"
        ? moveTo(original, id, start.x + d.dCols, start.y + d.dRows, this.doc.columns)
        : resizeTo(original, id, start.w + d.dCols, start.h + d.dRows, this.doc.columns);
      this.applyLayout();
    };
    const finish = (commit) => {
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", up);
      target.removeEventListener("pointercancel", cancel);
      document.removeEventListener("keydown", esc, true);
      entry.el.classList.remove("is-dragging");
      if (!commit) { this.doc.blocks = original; this.applyLayout(); return; }
      const moved = JSON.stringify(original) !== JSON.stringify(this.doc.blocks);
      if (moved) this.onChange("layout");
    };
    const up = () => finish(true);
    const cancel = () => finish(false);
    const esc = (e3) => { if (e3.key === "Escape") { e3.stopPropagation(); finish(false); } };
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerup", up);
    target.addEventListener("pointercancel", cancel);
    document.addEventListener("keydown", esc, true);
  }

  _onKey(ev, entry) {
    if (this.mode !== "edit" || ev.target !== entry.head) return;
    const next = keyNudge(this.doc.blocks, entry.block.id, ev.key, ev.shiftKey, this.doc.columns);
    if (next === this.doc.blocks) return;
    ev.preventDefault();
    this.doc.blocks = next;
    this.applyLayout();
    this.onChange("layout");
  }
}

/** A stream that never has data, for a display whose module names no
    stream (a status block, or a descriptor from an older backend). */
function emptyState() {
  return {
    rows: [], last: {}, keys: new Set(), t0: null, runEnd: null, latestFrame: null,
    conn: "connecting", dropped: 0, bad: 0,
  };
}

export { resolve as settleLayout };
