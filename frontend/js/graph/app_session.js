/* ---------------------------------------------------------------------------
   graph/app_session.js -- the editor's second mode: editing the graph an app owns.

   The editor has two sessions over the one document. The *free* session is the editor as it always was: the canvas is kept in this
   browser, graphs go to and from the library. An *app* session (`/graph?app=<name>`) puts the app's own graph on the canvas, keeps
   nothing in the browser (the free canvas waits untouched), and gives the page one more job: Save writes the graph back to the app
   (`PUT /apps/{name}/graph`; its page, style and description stay). Library and import/export are hidden, since the app's graph is
   not a library graph.

   Like the instance view this only coordinates: the document, canvas and draft saver are the editor's own.
   --------------------------------------------------------------------------- */

import { api } from "../api.js";
import { errText } from "../lib/errors.js";

const enc = encodeURIComponent;

export class AppSession {
  /**
   * @param {{name: string, doc, canvas, saver: {hold: (on: boolean) => void}, bar: HTMLElement, title: HTMLElement,
   *   saveBtn: HTMLElement, backLink: HTMLAnchorElement, onNote: Function, onChange?: (open: boolean) => void,
   *   timers?: {set: Function, clear: Function}}} parts
   */
  constructor({ name, doc, canvas, saver, bar, title, saveBtn, backLink, onNote, onChange = () => {}, timers = { set: (f, ms) => setTimeout(f, ms), clear: (id) => clearTimeout(id) } }) {
    Object.assign(this, { name, doc, canvas, saver, bar, title, saveBtn, onNote, onChange, timers });
    this.saved = null;     // the document as last loaded or saved, to tell what is unsaved
    this.dirty = false;
    this.version = 0;
    this.active = false;
    this._timer = null;
    backLink.href = `/apps/${enc(name)}`;
    saveBtn.addEventListener("click", () => this.save());
    doc.on("change", () => this._soon());
    const win = globalThis.window; // absent where the session is tested without a page
    win?.addEventListener("beforeunload", (e) => { if (this.dirty) { e.preventDefault(); e.returnValue = ""; } });
    win?.addEventListener("keydown", (e) => {
      if (this.active && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") { e.preventDefault(); this.save(); }
    });
  }

  /** Put the app's graph on the canvas. False (and the free session stays) when the app cannot be read. */
  async open() {
    let app;
    try {
      app = await api(`/apps/${enc(this.name)}`);
    } catch (err) {
      this.onNote(`Could not open the graph of app "${this.name}": ${errText(err)}`, "error");
      return false;
    }
    this.saver.hold(true); // the free canvas in this browser stays as it was
    this.doc.load(app.doc.graph || {});
    this.version = app.version;
    this.saved = this._key();
    this.active = true;
    this.bar.hidden = false;
    globalThis.document?.body.classList.add("is-app-edit");
    this._show();
    this.canvas.fit();
    this.onChange(true);
    return true;
  }

  _key() { return JSON.stringify(this.doc.toJSON()); }

  _soon() {
    if (!this.active) return;
    if (this._timer) this.timers.clear(this._timer);
    this._timer = this.timers.set(() => { this._timer = null; this.refresh(); }, 150);
  }

  /** Whether the canvas differs from what the app holds, and what the bar says about it. */
  refresh() {
    if (!this.active) return;
    this.dirty = this._key() !== this.saved;
    this._show();
  }

  _show() {
    this.title.textContent = `Graph of app “${this.name}” (version ${this.version})${this.dirty ? " · unsaved changes" : ""}`;
    this.saveBtn.textContent = this.dirty ? "Save •" : "Save";
    this.saveBtn.classList.toggle("is-dirty", this.dirty);
  }

  async save() {
    if (!this.active) return false;
    this.refresh();
    try {
      const out = await api(`/apps/${enc(this.name)}/graph`, { method: "PUT", body: { graph: this.doc.toJSON() } });
      this.version = out.app.version;
      this.saved = this._key();
      this.refresh();
      const lost = out.orphaned_cards.length;
      this.onNote(
        `Saved the graph of "${this.name}" (version ${this.version}).${lost ? ` ${lost} card${lost === 1 ? "" : "s"} on its page lost their target; the app page shows them.` : ""}`,
        lost ? "warn" : "success");
      return true;
    } catch (err) {
      this.onNote(`Could not save the graph of "${this.name}": ${errText(err)}`, "error");
      return false;
    }
  }
}

/** In the free session: one link per app to edit its graph (the way into an app session from the editor itself). */
export async function renderAppLinks(box, { onNote }) {
  try {
    const apps = (await api("/apps")).apps || [];
    box.replaceChildren();
    if (!apps.length) {
      box.append(Object.assign(document.createElement("div"), { className: "console-line info", textContent: "No apps yet." }));
      return;
    }
    for (const app of apps) {
      const row = Object.assign(document.createElement("div"), { className: "exec-row" });
      const link = Object.assign(document.createElement("a"), {
        href: `/graph?app=${enc(app.name)}`, textContent: app.name, title: `Edit the graph of app "${app.name}"`,
      });
      row.append(link, Object.assign(document.createElement("span"), { className: "exec-meta", textContent: `${app.node_count} nodes` }));
      box.append(row);
    }
  } catch (err) {
    onNote(errText(err), "error");
  }
}
