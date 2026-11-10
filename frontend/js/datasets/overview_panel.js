/* datasets/overview_panel.js -- the inspector's "Library" tab: how complete and how varied the dataset is, the
   problems worth fixing before a run (the report), its sets, and the latent caches runs built for it. */

import { h } from "../lib/dom.js";
import { errText } from "../lib/errors.js";
import { CHANNELS, datasets, fmtBytes, plural } from "./client.js";
import { confirmDialog } from "./dialogs.js";
import { icon } from "./icons.js";

const SHAPES = [["portrait", "Portrait"], ["square", "Square"], ["landscape", "Landscape"]];
const pct = (n, total) => (total ? Math.round((100 * n) / total) : 0);

export class OverviewPanel {
  /** @param {import("./library_state.js").LibraryState} state */
  constructor(root, state) {
    this.state = state;
    this.root = root;
    this.report = null;
    this.reportKey = "";
    this.visible = false;
    state.addEventListener("summary", () => this.visible && this.draw());
    state.addEventListener("channel", () => this.visible && this.loadReport());
  }

  show() {
    this.visible = true;
    this.draw();
    this.loadReport();
  }

  hide() { this.visible = false; }

  async loadReport() {
    const key = `${this.state.channel}:${this.state.summary?.images}:${JSON.stringify(this.state.summary?.captioned)}`;
    if (key === this.reportKey) return;
    this.reportKey = key;
    try {
      this.report = await datasets.report(this.state.name, this.state.channel);
    } catch (err) {
      this.report = { error: errText(err) };
    }
    if (this.visible) this.draw();
  }

  draw() {
    const me = this.state.summary;
    if (!me) return;
    this.root.replaceChildren(this.facts(me), this.shapes(me), this.coverage(me), this.problems(me), this.setList(me), this.cacheList(me));
  }

  /* ---- pieces ---- */

  section(title, ...children) {
    return h("section", { class: "ov-section" }, h("h3", { class: "ov-title", text: title }), ...children);
  }

  facts(me) {
    const cell = (value, label) => h("div", { class: "ov-fact" }, h("strong", { text: value }), h("span", { class: "text-dim", text: label }));
    const smallest = this.report?.smallest_side;
    return h("div", { class: "ov-facts" }, cell(me.images.toLocaleString("en"), "images"), cell(fmtBytes(me.bytes), "on disk"),
      cell(smallest ? `${smallest}px` : "–", "smallest side"));
  }

  shapes(me) {
    const total = Math.max(1, Object.values(me.orientation).reduce((a, b) => a + b, 0));
    const bar = h("div", { class: "stack" }, ...SHAPES.map(([k]) => {
      const part = h("i", { class: `stack-part ${k}`, title: `${me.orientation[k]} ${k}` });
      part.style.flexGrow = me.orientation[k];
      return part;
    }));
    const legend = h("div", { class: "legend" }, ...SHAPES.map(([k, t]) => h("span", { class: `legend-item ${k}` }, `${t} ${pct(me.orientation[k], total)}%`)));
    return this.section("Shapes", me.images ? h("div", {}, bar, legend) : h("p", { class: "text-dim", text: "No images yet." }));
  }

  coverage(me) {
    if (!me.images) return this.section("Captions", h("p", { class: "text-dim", text: "Nothing to caption yet." }));
    const rows = CHANNELS.map((c) => {
      const n = me.captioned[c] || 0;
      const fill = h("i", { class: "meter-fill" });
      fill.style.width = `${pct(n, me.images)}%`;
      const missing = me.images - n;
      return h("div", { class: "ov-row" },
        h("span", { class: "ov-key", text: c }), h("div", { class: `meter${missing === 0 && me.images ? " full" : ""}` }, fill),
        h("span", { class: "ov-val", text: `${n} / ${me.images}` }),
        missing > 0 ? h("button", { class: "link", text: `${missing} missing`, onclick: () => this.state.setView({ missing: c, set: "all" }) }) : h("span", { class: "ov-val ok", text: "complete" }));
    });
    return this.section("Captions", ...rows);
  }

  problems(me) {
    const r = this.report;
    const title = `Check · ${this.state.channel} captions`;
    if (!r) return this.section(title, h("p", { class: "text-dim", text: "Reading…" }));
    if (r.error) return this.section(title, h("p", { class: "error-inline", text: r.error }));
    if (!me.images) return this.section(title, h("p", { class: "text-dim", text: "Nothing to check yet." }));
    const out = [];
    const search = (q) => this.state.setView({ q, set: "all", missing: "" });
    if (r.repeated_captions.length) {
      out.push(h("div", { class: "ov-note warn" }, `${plural(r.repeated_captions.length, "caption")} used by more than one image`),
        ...r.repeated_captions.slice(0, 4).map((c) => h("button", { class: "ov-line", title: "Search for it", onclick: () => search(c.text) }, h("span", { class: "ov-snip", text: c.text }), h("span", { class: "ov-count", text: `×${c.count}` }))));
    }
    if (r.empty_captions.length) {
      out.push(h("div", { class: "ov-note warn" }, `${plural(r.empty_captions.length, "image")} without a ${this.state.channel} caption`,
        h("button", { class: "link", text: "show", onclick: () => this.state.setView({ missing: this.state.channel, set: "all" }) })));
    }
    if (!out.length) out.push(h("div", { class: "ov-note good" }, icon("check", 14), "Every image has its own caption."));
    out.push(h("div", { class: "ov-sub", text: `Sizes (${plural(r.distinct_sizes, "distinct size")})` }),
      h("div", { class: "ov-sizes" }, ...r.sizes.slice(0, 6).map((s) => h("span", { class: "chip", text: `${s.width}×${s.height} · ${s.count}` }))));
    if (r.tags.length) {
      out.push(h("div", { class: "ov-sub", text: `Most used tags (${plural(r.distinct_tags, "tag")})` }),
        h("div", { class: "ov-sizes" }, ...r.tags.slice(0, 16).map(([tag, n]) => h("button", { class: "chip chip-btn", title: "Search for it", onclick: () => search(tag) }, tag, h("span", { class: "chip-n", text: n })))));
    }
    return this.section(title, ...out);
  }

  setList(me) {
    const names = Object.keys(me.sets).filter((n) => n !== "all");
    const rows = names.map((n) => h("div", { class: "ov-row" },
      h("button", { class: "link ov-name", text: n, title: `Show set ${n}`, onclick: () => this.state.setView({ set: n }) }),
      h("span", { class: "ov-val ov-grow", text: plural(me.sets[n], "image") }),
      h("button", { class: "icon-btn", title: `Delete set ${n}`, "aria-label": `Delete set ${n}`, onclick: () => this.removeSet(n) }, icon("trash", 14))));
    return this.section("Sets", ...(rows.length ? rows : [h("p", { class: "text-dim", text: "A set is a subset a run can train on. Select images and add them to one, or drag them onto a set tab." })]));
  }

  async removeSet(name) {
    if (await confirmDialog({ title: `Delete set “${name}”?`, text: "The images stay in the dataset; only the grouping goes." })) this.state.deleteSet(name);
  }

  cacheList(me) {
    const rows = me.caches.map((c) => h("div", { class: "ov-row" },
      h("span", { class: "ov-name", text: c.vae, title: c.vae }), h("span", { class: "ov-val ov-grow ov-right", text: `${plural(c.files, "file")} · ${fmtBytes(c.bytes)}` }),
      h("button", { class: "icon-btn", title: "Delete this cache", "aria-label": `Delete cache ${c.vae}`, onclick: () => this.removeCache(c.vae) }, icon("trash", 14))));
    return this.section("Latent caches", ...(rows.length ? rows : [h("p", { class: "text-dim", text: "None yet. A run that trains on this dataset builds one per VAE." })]),
      rows.length ? h("p", { class: "text-dim ov-hint", text: "Rebuilt by the next run; delete one after changing the images or the bucket settings." }) : "");
  }

  async removeCache(vae) {
    if (await confirmDialog({ title: `Delete the cache for ${vae}?`, text: "The next run that trains on this dataset builds it again." })) this.state.deleteCache(vae);
  }
}
