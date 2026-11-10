/* datasets/batch_panel.js -- the inspector for SEVERAL selected images: edit one caption channel in bulk (with a
   preview of the result on the first image), file them into a set, or remove them. */

import { h } from "../lib/dom.js";
import { applyEdit, EDITS } from "./caption_edit.js";
import { datasets } from "./client.js";
import { icon } from "./icons.js";

const STACK = 5;

export class BatchPanel {
  /** @param {import("./library_state.js").LibraryState} state  @param {{onRemove: () => void}} hooks */
  constructor(root, state, hooks) {
    this.state = state;
    this.stack = h("div", { class: "bp-stack" });
    this.title = h("div", { class: "bp-title" });
    this.mode = h("select", { "aria-label": "Edit kind", onchange: () => this.sync() }, ...EDITS.map(([v, t]) => h("option", { value: v, text: t })));
    this.find = h("input", { type: "text", placeholder: "Find", spellcheck: false, "aria-label": "Find" });
    this.text = h("input", { type: "text", placeholder: "Text", spellcheck: false, "aria-label": "Text" });
    for (const input of [this.find, this.text]) {
      input.addEventListener("input", () => this.preview());
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") this.apply(); });
    }
    this.channelLabel = h("div", { class: "ip-label" });
    this.before = h("div", { class: "bp-diff dim" });
    this.after = h("div", { class: "bp-diff" });
    this.diff = h("div", { class: "bp-preview" }, this.before, this.after);
    this.applyBtn = h("button", { class: "btn btn-start btn-small", text: "Apply to all", onclick: () => this.apply() });
    this.setInput = h("input", { type: "text", list: "ds-set-names", placeholder: "Set name…", spellcheck: false, "aria-label": "Set name" });
    this.setInput.addEventListener("keydown", (e) => { if (e.key === "Enter") this.addToSet(); });
    this.addBtn = h("button", { class: "btn btn-secondary btn-small", text: "Add", onclick: () => this.addToSet() });
    this.leaveBtn = h("button", { class: "btn btn-secondary btn-small bp-wide", onclick: () => state.removeFromSet(state.selection.list(), state.view.set) });
    root.append(
      h("div", { class: "bp-head" }, this.stack, this.title),
      h("div", { class: "ip-block" }, this.channelLabel, this.mode, h("div", { class: "bp-pair" }, this.find, this.text), this.diff, this.applyBtn),
      h("div", { class: "ip-block" }, h("div", { class: "ip-label", text: "Set" }), h("div", { class: "bp-pair bp-set" }, this.setInput, this.addBtn), this.leaveBtn),
      h("button", { class: "btn btn-kill btn-small ip-remove", onclick: () => hooks.onRemove() }, icon("trash", 14), "Remove from dataset"));
    state.addEventListener("channel", () => this.sync());
    state.addEventListener("items", () => this.preview());
  }

  show() {
    const picked = this.state.selectedItems().slice(0, STACK);
    this.stack.replaceChildren(...picked.map((item, n) => {
      const img = h("img", { src: datasets.imageUrl(this.state.name, item.id, 128), alt: "" });
      img.style.setProperty("--n", n);
      return img;
    }));
    this.title.replaceChildren(h("strong", { text: `${this.state.selection.size.toLocaleString("en")} images` }), h("span", { class: "text-dim", text: "selected" }));
    this.leaveBtn.hidden = this.state.view.set === "all";
    this.leaveBtn.textContent = `Take out of “${this.state.view.set}”`;
    this.sync();
  }

  sync() {
    const mode = this.mode.value;
    this.find.hidden = mode !== "replace";
    this.text.hidden = mode === "clear";
    this.channelLabel.textContent = `${this.state.channel} caption`;
    this.preview();
  }

  preview() {
    const first = this.state.selectedItems()[0];
    const old = first?.captions[this.state.channel] || "";
    this.diff.hidden = !first;
    if (!first) return;
    this.before.textContent = old || "(empty)";
    this.after.textContent = applyEdit(old, this.mode.value, this.text.value, this.find.value) || "(empty)";
  }

  async apply() {
    const mode = this.mode.value;
    if (mode !== "clear" && mode !== "replace" && !this.text.value) return;
    if (mode === "replace" && !this.find.value) return;
    const ok = await this.state.editCaptions(this.state.selection.list(), { mode, text: this.text.value, find: this.find.value });
    if (ok) { this.text.value = ""; this.find.value = ""; }
  }

  async addToSet() {
    const name = this.setInput.value.trim();
    if (!name) return;
    this.setInput.value = "";
    await this.state.addToSet(this.state.selection.list(), name);
  }
}
