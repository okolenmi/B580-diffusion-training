/* datasets/library_state.js -- everything the open dataset's panels share: the view (set, search, filter), the loaded
   images, the selection and the operations on them. Panels read it and listen for events; they never call the API
   for the dataset themselves, so one change reaches every panel.

   Events: "summary" (counts changed), "items" (the loaded images changed), "item" (one caption changed, detail.id),
   "selection", "channel". */

import { errText } from "../lib/errors.js";
import { datasets, CHANNELS } from "./client.js";
import { Selection } from "./selection.js";

const PAGE = 80;
const FETCH_MAX = 300;

export class LibraryState extends EventTarget {
  constructor(name, note) {
    super();
    this.name = name;
    this.note = note;
    this.selection = new Selection();
    this.view = { set: "all", q: "", missing: "" };
    this.channel = CHANNELS[0];
    this.items = [];
    this.display = null;
    this.total = 0;
    this.summary = null;
    this.loading = false;
    this.token = 0;
  }

  emit(type, detail) { this.dispatchEvent(new CustomEvent(type, { detail })); }

  /** Ids as they read on screen: the mosaic may deal pictures out of import order (packing.js). */
  get order() { return this.display ?? this.items.map((i) => i.id); }
  shown() { return this.order.map((id) => this.find(id)).filter(Boolean); }
  find(id) { return this.items.find((i) => i.id === id) ?? null; }
  setNames() { return Object.keys(this.summary?.sets ?? {}).filter((n) => n !== "all"); }

  async start() {
    await this.refreshSummary();
    await this.load();
  }

  async refreshSummary() {
    const me = (await datasets.list()).find((d) => d.name === this.name);
    if (!me) throw new Error(`dataset ${this.name} no longer exists`);
    this.summary = me;
    if (!(this.view.set in me.sets)) this.view = { ...this.view, set: "all" };
    this.emit("summary");
  }

  /** First page of the current view; a newer call makes an older one that is still in flight irrelevant. */
  async load() {
    const token = ++this.token;
    this.loading = true;
    try {
      const page = await datasets.images(this.name, this.view, { limit: PAGE });
      if (token !== this.token) return;
      this.items = page.items;
      this.total = page.total;
    } catch (err) {
      this.note(errText(err), "error");
    } finally {
      if (token === this.token) this.loading = false;
    }
    this.emit("items");
  }

  async loadMore() {
    if (this.loading || this.items.length >= this.total) return;
    const token = this.token;
    this.loading = true;
    try {
      const page = await datasets.images(this.name, this.view, { offset: this.items.length, limit: PAGE });
      if (token !== this.token) return;
      this.items.push(...page.items);
      this.total = page.total;
    } catch (err) {
      this.note(errText(err), "error");
    } finally {
      this.loading = false;
    }
    this.emit("items");
  }

  /** After an edit: counts, and the images already on screen, fresh. */
  async refresh() {
    await this.refreshSummary();
    const token = ++this.token;
    const want = Math.max(this.items.length, PAGE);
    const items = [];
    let total = 0;
    for (let offset = 0; offset < want; offset += FETCH_MAX) {
      const page = await datasets.images(this.name, this.view, { offset, limit: Math.min(FETCH_MAX, want - offset) });
      items.push(...page.items);
      total = page.total;
      if (items.length >= total) break;
    }
    if (token !== this.token) return;
    this.items = items;
    this.total = total;
    const loaded = new Set(items.map((i) => i.id));
    for (const id of this.selection.list()) if (!loaded.has(id) && items.length >= total) this.selection.ids.delete(id);
    this.emit("items");
    this.emit("selection");
  }

  /* ---- view ---- */

  async setView(patch) {
    this.view = { ...this.view, ...patch };
    this.selection.clear();
    this.emit("selection");
    await this.load();
  }

  setChannel(channel) {
    if (channel === this.channel) return;
    this.channel = channel;
    this.emit("channel");
  }

  /* ---- selection ---- */

  select(id, how) {
    if (how === "toggle") this.selection.toggle(id);
    else if (how === "range") this.selection.range(this.order, id);
    else this.selection.only(id);
    this.emit("selection");
  }

  clearSelection() {
    if (!this.selection.size) return;
    this.selection.clear();
    this.emit("selection");
  }

  /** Every image the view selects, also the ones not loaded yet. */
  async selectAll() {
    this.selection.fill(await datasets.ids(this.name, this.view));
    this.emit("selection");
  }

  selectedItems() { return this.selection.list().map((id) => this.find(id)).filter(Boolean); }

  /* ---- operations; each one tells the user what happened and throws nothing the panels must handle ---- */

  async run(work, done) {
    try {
      const result = await work();
      this.note(done(result));
      await this.refresh();
      return true;
    } catch (err) {
      this.note(errText(err), "error");
      return false;
    }
  }

  /** One image's caption: patched in place, no reload, because the user is typing. */
  async saveCaption(id, channel, text) {
    await datasets.editCaptions(this.name, { ids: [id], channel, mode: "set", text });
    const item = this.find(id);
    if (item) {
      item.captions[channel] = text;
      this.emit("item", { id });
    }
    this.refreshSummary().catch(() => {});
  }

  editCaptions(ids, edit) {
    return this.run(() => datasets.editCaptions(this.name, { ids, channel: this.channel, ...edit }), (r) => `${r.changed} ${this.channel} captions changed.`);
  }

  addToSet(ids, setName) {
    return this.run(() => datasets.addToSet(this.name, setName, ids), (r) => `Set "${r.name}" now has ${r.members} images.`);
  }

  async removeFromSet(ids, setName) {
    const ok = await this.run(() => datasets.removeImages(this.name, ids, setName), (r) => `${r.removed} images left set "${setName}".`);
    if (ok) this.clearSelection();
    return ok;
  }

  async removeImages(ids) {
    const ok = await this.run(() => datasets.removeImages(this.name, ids, null), (r) => `${r.removed} images removed from the dataset (files untouched).`);
    if (ok) this.clearSelection();
    return ok;
  }

  deleteSet(setName) {
    return this.run(() => datasets.deleteSet(this.name, setName), () => `Set "${setName}" deleted (its images stay).`);
  }

  deleteCache(vae) {
    return this.run(() => datasets.deleteCache(this.name, vae), () => `Cache ${vae} deleted; the next run rebuilds it.`);
  }
}
