/* datasets/mosaic.js -- the image mosaic: rows keep every picture's own proportions (a LoRA set mixes portrait and
   landscape, and a square crop would hide exactly what bucketing does with them). Owns click/keyboard selection,
   loading more on scroll, and dragging the selection onto a set tab. */

import { h } from "../lib/dom.js";
import { CHANNELS, datasets } from "./client.js";
import { icon } from "./icons.js";
import { justify } from "./layout.js";

const THUMB = 384;
const GAP = 6;
export const DRAG_TYPE = "application/x-dataset-images";

export class Mosaic {
  /** @param {import("./library_state.js").LibraryState} state
   *  @param {{lightbox: import("./dialogs.js").Lightbox, onRemove: () => void, onImport: () => void}} hooks */
  constructor(root, state, hooks) {
    this.state = state;
    this.hooks = hooks;
    this.tiles = new Map();
    this.shown = [];
    this.row = 180;
    this.grid = h("div", { class: "mosaic", role: "listbox", "aria-multiselectable": "true" });
    this.sentinel = h("div", { class: "mosaic-end" });
    this.empty = h("div", { class: "mosaic-empty", hidden: true });
    this.scroller = h("div", { class: "mosaic-scroll", tabindex: "0", "aria-label": "Images" }, this.grid, this.sentinel, this.empty);
    root.append(this.scroller);

    this.grid.addEventListener("click", (e) => this.click(e));
    this.grid.addEventListener("dblclick", (e) => { const id = this.idAt(e); if (id) this.openViewer(id); });
    this.grid.addEventListener("dragstart", (e) => this.dragStart(e));
    this.scroller.addEventListener("keydown", (e) => this.key(e));
    new IntersectionObserver((entries) => { if (entries.some((x) => x.isIntersecting)) state.loadMore(); }, { root: this.scroller, rootMargin: "800px" }).observe(this.sentinel);

    new ResizeObserver(() => this.layout()).observe(this.grid);
    state.addEventListener("items", () => this.render());
    state.addEventListener("selection", () => this.paintSelection());
    state.addEventListener("channel", () => this.items().forEach((item) => this.paint(item)));
    state.addEventListener("item", (e) => { const item = state.find(e.detail.id); if (item) this.paint(item); });
  }

  setRow(px) {
    this.row = px;
    this.layout();
  }

  /** Size every tile so the rows fill the width (layout.js); runs on new images, a resize and a tile-size change. */
  layout() {
    const width = this.grid.clientWidth;
    const items = this.items();
    if (!width || !items.length) return;
    for (const row of justify(items.map((i) => i.width / i.height), width, this.row, GAP)) {
      for (let n = row.start; n < row.start + row.count; n++) {
        const item = items[n];
        const tile = this.tiles.get(item.id);
        if (!tile) continue;
        tile.style.width = `${Math.floor((item.width / item.height) * row.height)}px`;
        tile.style.height = `${Math.round(row.height)}px`;
      }
    }
  }
  items() { return this.state.items; }

  /* ---- drawing ---- */

  render() {
    const items = this.items();
    const ids = items.map((i) => i.id);
    const continues = this.shown.length > 0 && this.shown.length <= ids.length && this.shown.every((id, n) => id === ids[n]);
    if (!continues) {
      this.grid.replaceChildren();
      this.tiles.clear();
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
    const items = this.items();
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

  /** The image an arrow key moves to: undefined = not a movement key, null = nowhere to go. */
  neighbour(key, order, current) {
    if (!order.length || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(key)) return undefined;
    if (key === "Home") return order[0];
    if (key === "End") return order.at(-1);
    const at = order.indexOf(current);
    if (at < 0) return order[0];
    if (key === "ArrowLeft") return order[at - 1] ?? null;
    if (key === "ArrowRight") return order[at + 1] ?? null;
    return this.rowNeighbour(order, at, key === "ArrowDown" ? 1 : -1);
  }

  /** Rows are uneven, so "the tile below" is the one in the next row whose centre is nearest this one's. */
  rowNeighbour(order, at, direction) {
    const rect = (id) => this.tiles.get(id)?.getBoundingClientRect();
    const here = rect(order[at]);
    if (!here) return null;
    const x = here.left + here.width / 2;
    let rowTop = null;
    let best = null;
    for (let i = at + direction; i >= 0 && i < order.length; i += direction) {
      const r = rect(order[i]);
      if (!r) break;
      if (Math.abs(r.top - here.top) < 4) continue;
      rowTop ??= r.top;
      if (Math.abs(r.top - rowTop) >= 4) break;
      const gap = Math.abs(r.left + r.width / 2 - x);
      if (!best || gap < best.gap) best = { id: order[i], gap };
    }
    return best ? best.id : null;
  }
}
