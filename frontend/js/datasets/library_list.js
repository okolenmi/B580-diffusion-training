/* datasets/library_list.js -- the dataset cards: a mosaic of the newest pictures, how complete the captions are,
   and what is built on top (sets, latent caches). The first tile creates a new one. */

import { h } from "../lib/dom.js";
import { errText } from "../lib/errors.js";
import { CHANNELS, datasets, fmtBytes, plural } from "./client.js";
import { confirmDialog } from "./dialogs.js";
import { icon } from "./icons.js";

const pct = (n, total) => (total ? Math.round((100 * n) / total) : 0);

export class LibraryList {
  constructor(root, { note, sub }) {
    this.note = note;
    this.sub = sub;
    this.state = h("div", { class: "state-box", text: "Loading…" });
    this.grid = h("div", { class: "ds-cards" });
    root.append(this.state, this.grid);
  }

  async refresh() {
    try {
      const all = await datasets.list();
      this.state.hidden = true;
      this.grid.replaceChildren(...all.map((d) => this.card(d)), new NewCard(this).node);
      const images = all.reduce((n, d) => n + d.images, 0);
      this.sub.textContent = all.length ? `${plural(all.length, "dataset")} · ${plural(images, "image")}` : "Image libraries your runs train on";
    } catch (err) {
      this.state.hidden = false;
      this.state.textContent = errText(err);
    }
  }

  card(d) {
    const caps = d.captioned[CHANNELS[0]] || 0;
    const share = pct(caps, d.images);
    const covers = d.covers.map((id) => h("img", { src: datasets.imageUrl(d.name, id, 256), alt: "", loading: "lazy" }));
    const cover = h("div", { class: `card-cover n${covers.length}` }, ...(covers.length ? covers : [icon("image", 30), h("span", { text: "empty" })]));
    const fill = h("i", { class: "meter-fill" });
    fill.style.width = `${share}%`;
    const sets = Object.keys(d.sets).length - 1;
    const built = d.caches.reduce((n, c) => n + c.bytes, 0);
    const link = h("a", { class: "ds-card", href: `/datasets/${encodeURIComponent(d.name)}` }, cover,
      h("div", { class: "card-info" },
        h("div", { class: "card-name", text: d.name }),
        h("div", { class: "card-line" }, h("strong", { text: d.images.toLocaleString("en") }), " images", d.bytes ? h("span", { class: "text-dim", text: ` · ${fmtBytes(d.bytes)}` }) : ""),
        d.images ? h("div", { class: "card-caps", title: `${caps} of ${d.images} images have a text caption` },
          h("div", { class: `meter${share === 100 ? " full" : ""}` }, fill), h("span", { class: "text-dim", text: share === 100 ? "captioned" : `${share}% captioned` })) : "",
        h("div", { class: "card-tags text-dim", text: [sets ? plural(sets, "set") : "", d.caches.length ? `${plural(d.caches.length, "cache")} · ${fmtBytes(built)}` : ""].filter(Boolean).join(" · ") || "no sets or caches yet" })));
    const del = h("button", { class: "card-del icon-btn", title: "Delete this dataset", "aria-label": `Delete ${d.name}`, onclick: () => this.remove(d) }, icon("trash", 15));
    return h("div", { class: "ds-card-wrap" }, link, del);
  }

  async remove(d) {
    const ok = await confirmDialog({
      title: `Delete “${d.name}”?`,
      text: `Its ${plural(d.images, "image")}, captions, sets and latent caches are removed from the dataset folder. The picture files themselves are not touched.`,
    });
    if (!ok) return;
    try {
      await datasets.remove(d.name);
      this.note(`Deleted ${d.name}.`);
      this.refresh();
    } catch (err) { this.note(errText(err), "error"); }
  }
}

/** The dashed first tile: click, type a name, Enter. */
class NewCard {
  constructor(list) {
    this.list = list;
    this.input = h("input", { type: "text", placeholder: "Dataset name", autocomplete: "off", spellcheck: false, "aria-label": "Dataset name" });
    this.error = h("div", { class: "error-inline" });
    this.form = h("div", { class: "new-form", hidden: true }, this.input, this.error,
      h("div", { class: "new-actions" }, h("button", { class: "btn btn-secondary btn-small", text: "Cancel", onclick: () => this.close() }), h("button", { class: "btn btn-start btn-small", text: "Create", onclick: () => this.create() })));
    this.open = h("button", { class: "new-open", onclick: () => this.edit() }, icon("plus", 26), h("span", { text: "New dataset" }));
    this.node = h("div", { class: "ds-card ds-new" }, this.open, this.form);
    this.input.addEventListener("keydown", (e) => { if (e.key === "Enter") this.create(); if (e.key === "Escape") this.close(); });
  }

  edit() {
    this.open.hidden = true;
    this.form.hidden = false;
    this.input.focus();
  }

  close() {
    this.form.hidden = true;
    this.open.hidden = false;
    this.input.value = "";
    this.error.textContent = "";
  }

  async create() {
    const name = this.input.value.trim();
    if (!name) return;
    try {
      await datasets.create(name);
      location.href = `/datasets/${encodeURIComponent(name)}`;
    } catch (err) { this.error.textContent = errText(err); }
  }
}
