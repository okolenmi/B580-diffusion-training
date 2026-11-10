/* datasets/set_tabs.js -- the dataset's sets as tabs over the mosaic. A set is a named subset a run can train on;
   dropping selected images on a tab adds them to it. */

import { h } from "../lib/dom.js";
import { errText } from "../lib/errors.js";
import { DRAG_TYPE } from "./mosaic.js";

export class SetTabs {
  /** @param {import("./library_state.js").LibraryState} state */
  constructor(root, state) {
    this.state = state;
    this.root = root;
    for (const type of ["summary", "items"]) state.addEventListener(type, () => this.draw());
  }

  draw() {
    if (!this.state.summary) return;
    const { sets } = this.state.summary;
    const names = Object.keys(sets).sort((a, b) => (a === "all" ? -1 : b === "all" ? 1 : a.localeCompare(b)));
    this.root.replaceChildren(...names.map((n) => this.tab(n, sets[n])));
  }

  tab(name, count) {
    const active = this.state.view.set === name;
    const tab = h("button", { class: `settab${active ? " active" : ""}`, role: "tab", "aria-selected": String(active), onclick: () => this.state.setView({ set: name }) },
      h("span", { class: "settab-name", text: name === "all" ? "All images" : name }), h("span", { class: "settab-n", text: count.toLocaleString("en") }));
    if (name !== "all") this.dropTarget(tab, name);
    return tab;
  }

  dropTarget(tab, name) {
    tab.addEventListener("dragover", (e) => { if (e.dataTransfer.types.includes(DRAG_TYPE)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; tab.classList.add("drop"); } });
    tab.addEventListener("dragleave", () => tab.classList.remove("drop"));
    tab.addEventListener("drop", async (e) => {
      e.preventDefault();
      tab.classList.remove("drop");
      try { await this.state.addToSet(JSON.parse(e.dataTransfer.getData(DRAG_TYPE)), name); } catch (err) { this.state.note(errText(err), "error"); }
    });
  }
}
