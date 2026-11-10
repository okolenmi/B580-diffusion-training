/* datasets/image_panel.js -- the inspector for ONE selected image: preview, sets, and the caption editor, which saves
   by itself (idle, leaving the field, changing image or channel) so there is no Save button to forget. */

import { h } from "../lib/dom.js";
import { errText } from "../lib/errors.js";
import { CHANNELS, datasets, fmtBytes } from "./client.js";
import { icon } from "./icons.js";

const IDLE_MS = 900;
const gcd = (a, b) => (b ? gcd(b, a % b) : a);
const ratio = (w, h) => { const d = gcd(w, h); return w / d <= 32 ? `${w / d}:${h / d}` : (w / h).toFixed(2); };

export class ImagePanel {
  /** @param {import("./library_state.js").LibraryState} state
   *  @param {{lightbox: import("./dialogs.js").Lightbox, onRemove: () => void}} hooks */
  constructor(root, state, hooks) {
    this.state = state;
    this.hooks = hooks;
    this.item = null;
    this.dirty = false;
    this.timer = null;
    this.preview = h("img", { class: "ip-preview", alt: "", title: "Open full size", onclick: () => this.open() });
    this.name = h("div", { class: "ip-name" });
    this.meta = h("div", { class: "ip-meta text-dim" });
    this.sets = h("div", { class: "ip-sets" });
    this.setInput = h("input", { type: "text", list: "ds-set-names", placeholder: "Add to set…", spellcheck: false, "aria-label": "Add to set" });
    this.setInput.addEventListener("keydown", (e) => { if (e.key === "Enter") this.addToSet(); });
    this.tabs = h("div", { class: "seg seg-fill", role: "tablist" }, ...CHANNELS.map((c) => h("button", { class: "seg-btn", role: "tab", "data-channel": c, text: c, onclick: () => state.setChannel(c) })));
    this.field = h("textarea", { class: "ip-caption", rows: 7, spellcheck: false, "aria-label": "Caption" });
    this.field.addEventListener("input", () => this.edited());
    this.field.addEventListener("blur", () => this.flush());
    this.field.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); this.flush(); } });
    this.count = h("span", { class: "text-dim" });
    this.status = h("span", { class: "ip-status" });
    root.append(
      this.preview, this.name, this.meta,
      h("div", { class: "ip-block" }, this.tabs, this.field, h("div", { class: "ip-foot" }, this.count, this.status)),
      h("div", { class: "ip-block" }, h("div", { class: "ip-label", text: "Sets" }), this.sets, this.setInput),
      h("button", { class: "btn btn-kill btn-small ip-remove", onclick: () => hooks.onRemove() }, icon("trash", 14), "Remove from dataset"));
    state.addEventListener("channel", () => { this.flush(); this.drawCaption(); });
    state.addEventListener("items", () => this.sync());
  }

  /** Show this image; whatever was being typed for the previous one is saved first. */
  async show(item) {
    if (this.item && this.item.id !== item.id) await this.flush();
    this.item = item;
    this.preview.src = datasets.imageUrl(this.state.name, item.id, 768);
    this.name.textContent = item.path.split("/").pop();
    this.name.title = item.path;
    this.meta.textContent = `${item.width} × ${item.height} · ${ratio(item.width, item.height)} · ${fmtBytes(item.bytes)}`;
    this.status.textContent = "";
    this.drawSets();
    this.drawCaption();
  }

  /** The list was reloaded: take the fresh sets and, unless the user is mid-edit, the fresh caption. */
  sync() {
    if (!this.item) return;
    const fresh = this.state.find(this.item.id);
    if (!fresh) return;
    this.item = fresh;
    this.drawSets();
    if (!this.dirty) this.drawCaption();
  }

  open() {
    const items = this.state.shown();
    this.hooks.lightbox.open(items, Math.max(0, items.findIndex((i) => i.id === this.item.id)));
  }

  /* ---- sets ---- */

  drawSets() {
    const names = this.item.sets;
    this.sets.replaceChildren(...(names.length ? names.map((n) => h("span", { class: "chip" }, n,
      h("button", { class: "chip-x", title: `Take out of ${n}`, "aria-label": `Take out of ${n}`, onclick: () => this.state.removeFromSet([this.item.id], n).then(() => this.state.select(this.item.id, "only")) }, icon("x", 11))))
      : [h("span", { class: "text-dim ip-none", text: "Not in any set yet" })]));
  }

  async addToSet() {
    const name = this.setInput.value.trim();
    if (!name || !this.item) return;
    this.setInput.value = "";
    await this.state.addToSet([this.item.id], name);
  }

  /* ---- caption ---- */

  drawCaption() {
    if (!this.item) return;
    const channel = this.state.channel;
    for (const b of this.tabs.children) {
      b.classList.toggle("active", b.dataset.channel === channel);
      b.setAttribute("aria-selected", String(b.dataset.channel === channel));
      b.classList.toggle("filled", Boolean((this.item.captions[b.dataset.channel] || "").trim()));
    }
    this.field.value = this.item.captions[channel] || "";
    this.field.placeholder = channel === "tags" ? "comma, separated, tags" : channel === "desc" ? "A sentence describing the picture" : "The caption this picture trains with";
    this.dirty = false;
    this.countWords();
  }

  countWords() {
    const text = this.field.value.trim();
    const tags = this.state.channel === "tags";
    const n = !text ? 0 : tags ? text.split(/\s*,\s*/).filter(Boolean).length : text.split(/\s+/).length;
    this.count.textContent = tags ? `${n} tags` : `${n} words`;
  }

  edited() {
    this.dirty = true;
    this.status.className = "ip-status";
    this.status.textContent = "Editing…";
    this.countWords();
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.flush(), IDLE_MS);
  }

  async flush() {
    clearTimeout(this.timer);
    if (!this.dirty || !this.item) return;
    const { id } = this.item;
    const channel = this.state.channel;
    const text = this.field.value;
    this.dirty = false;
    this.status.className = "ip-status";
    this.status.textContent = "Saving…";
    try {
      await this.state.saveCaption(id, channel, text);
      if (this.item?.id === id) { this.status.className = "ip-status ok"; this.status.textContent = "Saved"; }
    } catch (err) {
      this.dirty = this.item?.id === id;
      this.status.className = "ip-status bad";
      this.status.textContent = `Not saved: ${errText(err)}`;
    }
  }
}
