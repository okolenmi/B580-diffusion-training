/* datasets/toolbar.js -- the strip above the mosaic: search, caption filter, which caption to show, tile size,
   and what is selected. */

import { h } from "../lib/dom.js";
import { CHANNELS, plural } from "./client.js";
import { icon } from "./icons.js";

const ROW_KEY = "datasets.row";
const ROW_MIN = 110;
const ROW_MAX = 320;
const ROW_DEFAULT = 180;

const store = {
  get() { try { return Number(localStorage.getItem(ROW_KEY)) || ROW_DEFAULT; } catch { return ROW_DEFAULT; } },
  set(v) { try { localStorage.setItem(ROW_KEY, String(v)); } catch { /* a private window: the size just is not remembered */ } },
};

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
    this.size = h("input", { type: "range", class: "tb-size", min: ROW_MIN, max: ROW_MAX, step: 10, value: store.get(), "aria-label": "Tile size", title: "Tile size" });
    this.size.value = store.get();
    mosaic.setRow(Number(this.size.value));
    this.size.addEventListener("input", () => { mosaic.setRow(Number(this.size.value)); store.set(Number(this.size.value)); });
    this.count = h("span", { class: "tb-count" });
    this.picked = h("span", { class: "tb-picked", hidden: true });
    root.append(
      h("div", { class: "tb-search-wrap" }, icon("search", 15), this.search),
      this.missing, this.channels,
      h("label", { class: "tb-size-wrap" }, icon("image", 13), this.size, icon("image", 18)),
      h("span", { class: "tb-gap" }), this.picked, this.count);
    for (const type of ["items", "selection", "channel", "summary"]) state.addEventListener(type, () => this.sync());
    this.sync();
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
