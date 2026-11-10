/* datasets/dialogs.js -- modal pieces the page shares: a confirmation and the full-size viewer. */

import { h } from "../lib/dom.js";
import { icon } from "./icons.js";

/** Ask before something that cannot be undone. Resolves true on confirm. */
export function confirmDialog({ title, text, ok = "Delete", danger = true }) {
  return new Promise((resolve) => {
    const dialog = h("dialog", { class: "dlg dlg-small" });
    const close = (answer) => { dialog.close(); dialog.remove(); resolve(answer); };
    dialog.append(
      h("div", { class: "dlg-body" }, h("h2", { class: "dlg-title", text: title }), h("p", { class: "dlg-text", text: text })),
      h("div", { class: "dlg-foot" }, h("span", { class: "dlg-gap" }),
        h("button", { class: "btn btn-secondary", text: "Cancel", onclick: () => close(false) }),
        h("button", { class: `btn ${danger ? "btn-danger" : "btn-start"}`, text: ok, onclick: () => close(true) })));
    dialog.addEventListener("cancel", (e) => { e.preventDefault(); close(false); });
    document.body.append(dialog);
    dialog.showModal();
    dialog.querySelector(".btn-secondary").focus();
  });
}

/** Full-size view of the images on screen, with ← → to move and Esc to close. */
export class Lightbox {
  /** @param {{name: string, url: (id: string, thumb?: number) => string, channel: () => string}} source */
  constructor(source) {
    this.source = source;
    this.items = [];
    this.at = 0;
    this.picture = h("img", { alt: "" });
    this.caption = h("div", { class: "lb-caption" });
    this.counter = h("span", { class: "lb-count" });
    this.dialog = h("dialog", { class: "lightbox" },
      h("button", { class: "lb-btn lb-close", title: "Close (Esc)", "aria-label": "Close", onclick: () => this.dialog.close() }, icon("x", 20)),
      h("button", { class: "lb-btn lb-prev", title: "Previous (←)", "aria-label": "Previous", onclick: () => this.step(-1) }, icon("left", 24)),
      h("button", { class: "lb-btn lb-next", title: "Next (→)", "aria-label": "Next", onclick: () => this.step(1) }, icon("right", 24)),
      this.picture, h("div", { class: "lb-foot" }, this.caption, this.counter));
    this.dialog.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") { e.preventDefault(); this.step(-1); }
      if (e.key === "ArrowRight") { e.preventDefault(); this.step(1); }
    });
    this.dialog.addEventListener("click", (e) => { if (e.target === this.dialog) this.dialog.close(); });
    document.body.append(this.dialog);
  }

  open(items, at) {
    this.items = items;
    this.at = at;
    this.show();
    if (!this.dialog.open) this.dialog.showModal();
  }

  step(delta) {
    const next = this.at + delta;
    if (next < 0 || next >= this.items.length) return;
    this.at = next;
    this.show();
  }

  show() {
    const item = this.items[this.at];
    this.picture.src = this.source.url(item.id);
    const text = item.captions[this.source.channel()] || "";
    this.caption.textContent = text || item.path.split("/").pop();
    this.caption.classList.toggle("dim", !text);
    this.counter.textContent = `${this.at + 1} / ${this.items.length}`;
  }
}
