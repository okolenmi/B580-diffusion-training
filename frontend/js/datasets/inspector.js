/* datasets/inspector.js -- the panel beside the mosaic. "Selection" follows what is selected (one image: its
   editor; several: bulk actions); "Library" is the dataset as a whole. It switches itself when a selection starts or
   ends, and stays where the user put it otherwise. */

import { h } from "../lib/dom.js";
import { BatchPanel } from "./batch_panel.js";
import { icon } from "./icons.js";
import { ImagePanel } from "./image_panel.js";
import { OverviewPanel } from "./overview_panel.js";

export class Inspector {
  /** @param {import("./library_state.js").LibraryState} state
   *  @param {{lightbox: import("./dialogs.js").Lightbox, onRemove: () => void}} hooks */
  constructor(root, state, hooks) {
    this.state = state;
    this.tab = "library";
    this.hadSelection = false;
    this.tabs = h("div", { class: "ins-tabs", role: "tablist" },
      this.tabButton("selection", "Selection", "image"), this.tabButton("library", "Library", "database"));
    this.hint = h("div", { class: "ins-hint" }, icon("image", 32), h("h3", { text: "Nothing selected" }),
      h("p", { class: "text-dim", text: "Click an image to caption it. Shift-click selects a range, Ctrl-click adds one, and the arrow keys move." }),
      h("p", { class: "text-dim", text: "Space opens it full size. Drag a selection onto a set tab to file it." }));
    const imageRoot = h("div", { class: "ins-pane" });
    const batchRoot = h("div", { class: "ins-pane" });
    this.overviewRoot = h("div", { class: "ins-pane" });
    this.image = new ImagePanel(imageRoot, state, hooks);
    this.batch = new BatchPanel(batchRoot, state, hooks);
    this.overview = new OverviewPanel(this.overviewRoot, state);
    this.imageRoot = imageRoot;
    this.batchRoot = batchRoot;
    this.body = h("div", { class: "ins-body" }, this.hint, imageRoot, batchRoot, this.overviewRoot);
    this.sets = h("datalist", { id: "ds-set-names" });
    root.append(this.tabs, this.body, this.sets);
    state.addEventListener("selection", () => this.onSelection());
    state.addEventListener("summary", () => this.sets.replaceChildren(...this.state.setNames().map((n) => h("option", { value: n }))));
    state.addEventListener("items", () => { if (this.state.selection.size === 1) this.image.sync(); });
    this.draw();
  }

  tabButton(id, label, glyph) {
    return h("button", { class: "ins-tab", role: "tab", "data-tab": id, onclick: () => { this.tab = id; this.draw(); } }, icon(glyph, 15), label);
  }

  onSelection() {
    const has = this.state.selection.size > 0;
    if (has !== this.hadSelection) this.tab = has ? "selection" : "library";
    this.hadSelection = has;
    this.draw();
  }

  showSelection() {
    const { selection } = this.state;
    const one = selection.single && this.state.find(selection.single);
    if (one) this.image.show(one);
    else if (selection.size > 1) this.batch.show();
  }

  draw() {
    const n = this.state.selection.size;
    for (const b of this.tabs.children) {
      b.classList.toggle("active", b.dataset.tab === this.tab);
      b.setAttribute("aria-selected", String(b.dataset.tab === this.tab));
    }
    const selectionTab = this.tab === "selection";
    this.hint.hidden = !(selectionTab && n === 0);
    this.imageRoot.hidden = !(selectionTab && n === 1);
    this.batchRoot.hidden = !(selectionTab && n > 1);
    this.overviewRoot.hidden = selectionTab;
    if (selectionTab) { this.overview.hide(); this.showSelection(); } else this.overview.show();
  }
}
