/* ---------------------------------------------------------------------------
   graph2/library.js -- saved graphs: list, save the open one, load, delete.

   The library stores the document as it is (graph-json.md); a saved graph of
   an older format is listed but loading it explains that the editor reads
   format 3 only -- there is no silent conversion.
   --------------------------------------------------------------------------- */

import { api } from "../api.js";
import { errText } from "../lib/errors.js";

const button = (label, onClick, extra = "") => {
  const b = Object.assign(document.createElement("button"), { className: `btn btn-secondary btn-small ${extra}`.trim(), textContent: label });
  b.style.cssText = "padding:0.1rem 0.4rem;font-size:0.68rem";
  b.addEventListener("click", (e) => { e.stopPropagation(); onClick(); });
  return b;
};

export class Library {
  constructor({ list, nameInput, saveBtn }, doc, { onNote, onLoaded }) {
    this.root = list;
    this.nameInput = nameInput;
    this.doc = doc;
    this.onNote = onNote;
    this.onLoaded = onLoaded;
    saveBtn.addEventListener("click", () => this.save());
    nameInput.addEventListener("keydown", (e) => { if (e.key === "Enter") this.save(); });
  }

  async refresh() {
    try {
      this.render((await api("/graphs/library")).graphs || []);
    } catch (err) {
      this.onNote(errText(err), "error");
    }
  }

  render(graphs) {
    this.root.replaceChildren();
    if (!graphs.length) {
      this.root.append(Object.assign(document.createElement("div"), { className: "console-line info", textContent: "No saved graphs." }));
      return;
    }
    for (const g of graphs) {
      const row = Object.assign(document.createElement("div"), { className: "exec-row" });
      const name = Object.assign(document.createElement("span"), { textContent: g.name, title: g.description || g.name });
      const meta = Object.assign(document.createElement("span"), { className: "exec-meta", textContent: `${g.node_count} nodes` });
      const actions = Object.assign(document.createElement("span"), {});
      actions.style.cssText = "display:flex;gap:0.3rem";
      actions.append(button("Load", () => this.load(g.name)), button("Del", () => this.remove(g.name), "btn-danger"));
      row.append(name, meta, actions);
      this.root.append(row);
    }
  }

  async save() {
    const name = this.nameInput.value.trim();
    if (!name || name.length > 120) return this.onNote("Graph name must be 1-120 characters.", "warn");
    if (!this.doc.nodes.size) return this.onNote("Canvas is empty -- nothing to save.", "warn");
    try {
      const res = await api(`/graphs/library/${encodeURIComponent(name)}`, { method: "PUT", body: this.doc.toJSON() });
      this.onNote(`Saved "${res.name}" (${res.node_count} nodes).`, "success");
      this.nameInput.value = "";
      await this.refresh();
    } catch (err) {
      this.onNote(errText(err), "error");
    }
  }

  async load(name) {
    if (this.doc.nodes.size && !window.confirm(`Replace the current canvas with "${name}"?`)) return;
    try {
      const saved = await api(`/graphs/library/${encodeURIComponent(name)}`);
      this.doc.load(saved.graph || {});
      this.onNote(`Loaded "${name}" (${this.doc.nodes.size} nodes).`, "success");
      this.onLoaded?.();
    } catch (err) {
      this.onNote(`"${name}": ${errText(err)}`, "error");
    }
  }

  async remove(name) {
    if (!window.confirm(`Delete saved graph "${name}"?`)) return;
    try {
      await api(`/graphs/library/${encodeURIComponent(name)}`, { method: "DELETE" });
      this.onNote(`Deleted "${name}".`, "warn");
      await this.refresh();
    } catch (err) {
      this.onNote(errText(err), "error");
    }
  }
}
