/* datasets/mosaic.js -- the image mosaic: rows keep every picture's own proportions (a LoRA set mixes portrait and
   landscape, and a square crop would hide exactly what bucketing does with them). Owns click/keyboard selection,
   loading more on scroll, and dragging the selection onto a set tab. */

import { h } from "../lib/dom.js";
import { CHANNELS, datasets } from "./client.js";
import { icon } from "./icons.js";
import { rowsLayout } from "./layout.js";
import { ColumnPacker, inset, readingOrder } from "./packing.js";

const THUMB = 384;
const GAP = 6;
const AHEAD = 800; // how far below the visible part pictures are already loaded
export const DRAG_TYPE = "application/x-dataset-images";

export class Mosaic {
  /** @param {import("./library_state.js").LibraryState} state
   *  @param {{lightbox: import("./dialogs.js").Lightbox, onRemove: () => void, onImport: () => void}} hooks */
  constructor(root, state, hooks) {
    this.state = state;
    this.hooks = hooks;
    this.tiles = new Map();
    this.boxes = new Map();
    this.shown = [];
    this.size = 180;
    this.mode = "balanced";
    this.packer = null; // the balanced layout keeps its columns while pictures are only appended
    this.cells = [];
    this.packKey = "";
    this.grid = h("div", { class: "mosaic", role: "listbox", "aria-multiselectable": "true" });
    this.sentinel = h("div", { class: "mosaic-end" });
    this.empty = h("div", { class: "mosaic-empty", hidden: true });
    this.scroller = h("div", { class: "mosaic-scroll", tabindex: "0", "aria-label": "Images" }, this.grid, this.sentinel, this.empty);
    root.append(this.scroller);

    this.grid.addEventListener("click", (e) => this.click(e));
    this.grid.addEventListener("dblclick", (e) => { const id = this.idAt(e); if (id) this.openViewer(id); });
    this.grid.addEventListener("dragstart", (e) => this.dragStart(e));
    this.scroller.addEventListener("keydown", (e) => this.key(e));
    new IntersectionObserver((entries) => { if (entries.some((x) => x.isIntersecting)) state.loadMore(); }, { root: this.scroller, rootMargin: `${AHEAD}px` }).observe(this.sentinel);

    let settle = null;
    new ResizeObserver(() => { clearTimeout(settle); settle = setTimeout(() => this.layout(), 120); }).observe(this.grid);
    state.addEventListener("items", () => this.render());
    state.addEventListener("selection", () => this.paintSelection());
    state.addEventListener("channel", () => this.items().forEach((item) => this.paint(item)));
    state.addEventListener("item", (e) => { const item = state.find(e.detail.id); if (item) this.paint(item); });
  }

  setSize(px) {
    this.size = px;
    this.layout();
  }

  setMode(mode) {
    this.mode = mode;
    this.layout();
  }

  /** Put every tile in its place; runs on new images, a tile-size or mode change, and (debounced) a resize. */
  layout() {
    const width = this.grid.clientWidth;
    const items = this.items();
    if (!width || !items.length) return;
    const ratios = items.map((i) => i.width / i.height);
    const result = this.mode === "rows" ? rowsLayout(ratios, width, this.size, GAP) : this.columns(ratios, width);
    this.boxes.clear();
    items.forEach((item, n) => {
      const tile = this.tiles.get(item.id);
      const box = result.boxes[n];
      if (!tile) return;
      tile.hidden = !box;
      if (!box) return;
      this.boxes.set(item.id, box);
      Object.assign(tile.style, { left: `${Math.round(box.x)}px`, top: `${Math.round(box.y)}px`, width: `${Math.round(box.w)}px`, height: `${Math.round(box.h)}px` });
    });
    this.grid.style.height = `${Math.ceil(result.height)}px`;
    this.state.display = result.order.map((n) => items[n].id);
    this.topUp();
  }

  /** The observer on the sentinel only speaks when it *enters* the loading zone, so a page that does not fill a tall window
   *  (or a layout that got shorter) would stop loading; every layout checks the zone itself. */
  topUp() {
    const s = this.scroller;
    if (s.scrollHeight - s.scrollTop - s.clientHeight < AHEAD) this.state.loadMore();
  }

  items() { return this.state.items; }

  /** Balanced layout: columns are shaped once and later pictures only join them, so scrolling in more never moves what is on screen. */
  columns(ratios, width) {
    const key = `${width}|${this.size}`;
    if (!this.packer || key !== this.packKey) {
      this.packer = new ColumnPacker(width + GAP, this.size, ratios);
      this.packKey = key;
      this.cells = [];
    }
    while (this.cells.length < ratios.length) this.cells.push(this.packer.place(ratios[this.cells.length]));
    const boxes = this.cells.map((c) => inset(c, GAP));
    return { boxes, order: readingOrder(boxes, this.size), height: Math.max(0, this.packer.height - GAP) };
  }

  /* ---- drawing ---- */

  render() {
    const items = this.items();
    const ids = items.map((i) => i.id);
    const continues = this.shown.length > 0 && this.shown.length <= ids.length && this.shown.every((id, n) => id === ids[n]);
    if (!continues) {
      this.grid.replaceChildren();
      this.tiles.clear();
      this.packer = null;
    }
    for (const item of items.slice(continues ? this.shown.length : 0)) {
      const tile = this.makeTile(item);
      this.tiles.set(item.id, tile);
      this.grid.append(tile);
    }
    this.shown = ids;
    this.layout();
    items.forEach((item) => this.paint(item));
    this.paintSelection();
    this.drawEmpty();
  }

  makeTile(item) {
    const dots = h("span", { class: "tile-dots" }, ...CHANNELS.map((c) => h("i", { "data-channel": c })));
    const tile = h("div", { class: "tile", role: "option", "data-id": item.id, draggable: true },
      h("img", { src: datasets.imageUrl(this.state.name, item.id, THUMB), alt: "", loading: "lazy", draggable: false }),
      h("button", { class: "tile-check", tabindex: "-1", title: "Select", "aria-label": "Select" }, icon("check", 14)),
      dots, h("div", { class: "tile-cap" }));
    return tile;
  }

  paint(item) {
    const tile = this.tiles.get(item.id);
    if (!tile) return;
    const text = item.captions[this.state.channel] || "";
    tile.classList.toggle("bare", !text.trim());
    tile.querySelector(".tile-cap").textContent = text;
    for (const dot of tile.querySelectorAll(".tile-dots i")) dot.classList.toggle("on", Boolean((item.captions[dot.dataset.channel] || "").trim()));
    tile.querySelector(".tile-dots").title = CHANNELS.map((c) => `${c}: ${(item.captions[c] || "").trim() ? "yes" : "none"}`).join(" · ");
  }

  paintSelection() {
    const { selection } = this.state;
    this.grid.classList.toggle("picking", selection.size > 0);
    for (const [id, tile] of this.tiles) {
      const on = selection.has(id);
      tile.classList.toggle("on", on);
      tile.setAttribute("aria-selected", String(on));
    }
  }

  drawEmpty() {
    const { total, loading, summary, view } = this.state;
    this.empty.hidden = total > 0 || loading;
    if (this.empty.hidden) return;
    const none = summary && summary.images === 0;
    this.empty.replaceChildren(icon("image", 40),
      h("h3", { text: none ? "This dataset is empty" : "No image matches" }),
      h("p", { class: "text-dim", text: none
        ? "Point it at a folder of pictures. They are read where they are, and a caption file next to a picture (name.txt, name.tags.txt, name.desc.txt) comes along."
        : "Nothing in this set fits the search and filter." }),
      none ? h("button", { class: "btn btn-start", onclick: this.hooks.onImport }, icon("import"), "Import images")
        : h("button", { class: "btn btn-secondary", text: "Clear search and filter", onclick: () => this.state.setView({ q: "", missing: "", set: view.set }) }));
  }

  /* ---- mouse ---- */

  idAt(e) { return e.target.closest(".tile")?.dataset.id ?? null; }

  click(e) {
    const id = this.idAt(e);
    if (!id) return;
    const how = e.target.closest(".tile-check") || e.ctrlKey || e.metaKey ? "toggle" : e.shiftKey ? "range" : "only";
    this.state.select(id, how);
  }

  openViewer(id) {
    const items = this.state.shown();
    this.hooks.lightbox.open(items, Math.max(0, items.findIndex((i) => i.id === id)));
  }

  dragStart(e) {
    const id = this.idAt(e);
    if (!id) return;
    if (!this.state.selection.has(id)) this.state.select(id, "only");
    const ids = this.state.selection.list();
    e.dataTransfer.setData(DRAG_TYPE, JSON.stringify(ids));
    e.dataTransfer.setData("text/plain", `${ids.length} images`);
    e.dataTransfer.effectAllowed = "copy";
  }

  /* ---- keyboard ---- */

  key(e) {
    const { state } = this;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === "a") { e.preventDefault(); state.selectAll(); return; }
    if (e.key === "Escape") { state.clearSelection(); return; }
    if (e.key === "Delete" && state.selection.size) { e.preventDefault(); this.hooks.onRemove(); return; }
    const order = state.order;
    const current = state.selection.single ?? state.selection.anchor;
    if ((e.key === " " || e.key === "Enter") && current) { e.preventDefault(); this.openViewer(current); return; }
    const target = this.neighbour(e.key, order, current);
    if (target === undefined) return;
    e.preventDefault();
    if (target === null) return;
    state.select(target, e.shiftKey ? "range" : "only");
    this.tiles.get(target)?.scrollIntoView({ block: "nearest" });
    if (order.length - order.indexOf(target) < 12) state.loadMore();
  }

  /** The image an arrow key moves to: undefined = not a movement key, null = nowhere to go. Rows are uneven and the
   *  balanced layout has no rows, so a move goes to the nearest tile that way; sideways, with nothing beside it, to
   *  the next in reading order. */
  neighbour(key, order, current) {
    if (!order.length || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(key)) return undefined;
    if (key === "Home") return order[0];
    if (key === "End") return order.at(-1);
    const here = this.boxes.get(current);
    if (!here) return order[0];
    const dir = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[key];
    const cx = here.x + here.w / 2;
    const cy = here.y + here.h / 2;
    let best = null;
    for (const [id, b] of this.boxes) {
      if (id === current) continue;
      const dx = b.x + b.w / 2 - cx;
      const dy = b.y + b.h / 2 - cy;
      const along = dx * dir[0] + dy * dir[1];
      const across = Math.abs(dx * dir[1]) + Math.abs(dy * dir[0]);
      if (along <= 1 || (dir[0] !== 0 ? across > here.h * 0.5 + b.h * 0.25 : across > here.w + b.w)) continue;
      const score = along + 2 * across;
      if (!best || score < best.score) best = { id, score };
    }
    if (best) return best.id;
    if (dir[0] === 0) return null;
    const at = order.indexOf(current);
    return order[at + dir[0]] ?? null;
  }
}
