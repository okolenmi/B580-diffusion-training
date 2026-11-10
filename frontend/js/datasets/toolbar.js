/* datasets/toolbar.js -- the strip above the mosaic: search, caption filter, which caption to show, tile size,
   and what is selected. */

import { h } from "../lib/dom.js";
import { CHANNELS, plural } from "./client.js";
import { icon } from "./icons.js";

const SIZE_MIN = 110;
const SIZE_MAX = 320;
const MODES = [
  ["balanced", "Balanced", "Pictures of every shape get about the same area; the order on screen follows the import order loosely"],
  ["rows", "Rows", "Rows of one height, strictly in import order; wide pictures come out bigger than tall ones"],
];

/** A setting that survives a reload; a private window just forgets it. */
const remembered = (key, fallback) => ({
  get() { try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; } },
  set(value) { try { localStorage.setItem(key, String(value)); } catch { /* not remembered */ } },
});
const sizeStore = remembered("datasets.size", "180");
const modeStore = remembered("datasets.layout", "balanced");

export class Toolbar {
  /** @param {import("./library_state.js").LibraryState} state  @param {import("./mosaic.js").Mosaic} mosaic */
  constructor(root, state, mosaic) {
    this.state = state;
    this.search = h("input", { type: "search", class: "tb-search", placeholder: "Search captions or names", spellcheck: false, "aria-label": "Search" });
    let typing = null;
    this.search.addEventListener("input", () => { clearTimeout(typing); typing = setTimeout(() => state.setView({ q: this.search.value.trim() }), 280); });
    this.missing = h("select", { class: "tb-select", "aria-label": "Caption filter", onchange: () => state.setView({ missing: this.missing.value }) },
      h("option", { value: "", text: "All images" }),
      ...CHANNELS.map((c) => h("option", { value: c, text: `Without ${c} caption` })));
    this.channels = h("div", { class: "seg", role: "group", "aria-label": "Caption shown" },
      ...CHANNELS.map((c) => h("button", { class: "seg-btn", "data-channel": c, text: c, title: `Show and edit the ${c} caption`, onclick: () => state.setChannel(c) })));
    this.size = h("input", { type: "range", class: "tb-size", min: SIZE_MIN, max: SIZE_MAX, step: 10, "aria-label": "Tile size", title: "Tile size" });
    this.size.value = sizeStore.get();
    mosaic.size = Number(this.size.value);
    this.size.addEventListener("input", () => { mosaic.setSize(Number(this.size.value)); sizeStore.set(this.size.value); });
    this.mode = h("div", { class: "seg", role: "group", "aria-label": "Layout" },
      ...MODES.map(([id, label, hint]) => h("button", { class: "seg-btn", "data-mode": id, text: label, title: hint, onclick: () => this.setMode(id) })));
    this.mosaic = mosaic;
    this.setMode(MODES.some(([id]) => id === modeStore.get()) ? modeStore.get() : "balanced", false);
    this.count = h("span", { class: "tb-count" });
    this.picked = h("span", { class: "tb-picked", hidden: true });
    root.append(
      h("div", { class: "tb-search-wrap" }, icon("search", 15), this.search),
      this.missing, this.channels, this.mode,
      h("label", { class: "tb-size-wrap" }, icon("image", 13), this.size, icon("image", 18)),
      h("span", { class: "tb-gap" }), this.picked, this.count);
    for (const type of ["items", "selection", "channel", "summary"]) state.addEventListener(type, () => this.sync());
    this.sync();
  }

  setMode(id, apply = true) {
    for (const b of this.mode.children) b.classList.toggle("active", b.dataset.mode === id);
    modeStore.set(id);
    if (apply) this.mosaic.setMode(id);
    else this.mosaic.mode = id;
  }

  sync() {
    const { state } = this;
    for (const b of this.channels.children) b.classList.toggle("active", b.dataset.channel === state.channel);
    this.missing.value = state.view.missing;
    if (document.activeElement !== this.search) this.search.value = state.view.q;
    const filtered = state.view.q || state.view.missing;
    this.count.textContent = filtered ? `${state.total.toLocaleString("en")} match` : plural(state.total, "image");
    this.count.hidden = state.selection.size > 0;
    this.picked.replaceChildren();
    this.picked.hidden = state.selection.size === 0;
    if (state.selection.size) {
      this.picked.append(`${state.selection.size.toLocaleString("en")} of ${state.total.toLocaleString("en")} selected`,
        state.selection.size < state.total ? h("button", { class: "link", text: "Select all", onclick: () => state.selectAll() }) : "",
        h("button", { class: "link", text: "Clear", onclick: () => state.clearSelection() }));
    }
  }
}
