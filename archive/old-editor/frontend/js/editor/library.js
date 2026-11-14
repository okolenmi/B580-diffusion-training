/* ---------------------------------------------------------------------------
   editor/library.js -- saved-graph storage + legacy draft import (M7).

   The backend library (`/graphs/library`) stores submitted graphs
   verbatim -- validation happens at run time, never at save (docs 05 §2),
   so an unknown class is preserved and flagged on load, not rejected.

   The legacy editor's canvas lived in localStorage (`ng_graph_v1`); the
   new frontend never writes that key -- it reads it once per explicit
   import click (docs 03 §5): {nodes:[{id,class_name,x,y,paramValues}],
   connections:[...]} -> GraphDoc.load(), which accepts both spellings.
   --------------------------------------------------------------------------- */

import { api, ApiError } from "../api.js";
import { errText } from "../lib/errors.js";

export class Library {
  constructor({ list, nameInput, saveBtn, importBtn }, doc, { onNote, onLoaded }) {
    this.root = list;
    this.nameInput = nameInput;
    this.doc = doc;
    this.onNote = onNote;
    this.onLoaded = onLoaded;   // () => after the canvas content is replaced
    saveBtn.addEventListener("click", () => this.save());
    importBtn.addEventListener("click", () => this.importLegacy());
    this.nameInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") this.save();
    });
  }

  async refresh() {
    try {
      const data = await api("/graphs/library");
      this.render(data.graphs || []);
    } catch (err) {
      this.onNote(errText(err), "error");
    }
  }

  render(graphs) {
    this.root.replaceChildren();
    if (!graphs.length) {
      const line = document.createElement("div");
      line.className = "console-line info";
      line.textContent = "No saved graphs.";
      this.root.appendChild(line);
      return;
    }
    for (const g of graphs) {
      const row = document.createElement("div");
      row.className = "exec-row";
      const main = document.createElement("span");
      main.textContent = g.name;
      main.title = g.description || g.name;
      const meta = document.createElement("span");
      meta.className = "exec-meta";
      meta.textContent = `${g.node_count} nodes`;
      const actions = document.createElement("span");
      actions.style.display = "flex";
      actions.style.gap = "0.3rem";
      actions.appendChild(_miniBtn("Load", () => this.load(g.name)));
      actions.appendChild(_miniBtn("Del", () => this.remove(g.name), "btn-danger"));
      row.append(main, meta, actions);
      this.root.appendChild(row);
    }
  }

  async save() {
    const name = this.nameInput.value.trim();
    if (!name || name.length > 120) {
      this.onNote("Graph name must be 1-120 characters.", "warn");
      return;
    }
    if (!this.doc.size) {
      this.onNote("Canvas is empty -- nothing to save.", "warn");
      return;
    }
    try {
      const res = await api(`/graphs/library/${encodeURIComponent(name)}`, {
        method: "PUT",
        body: this.doc.toLibraryPayload(),
      });
      this.onNote(`Saved "${res.name}" (${res.node_count} nodes).`, "success");
      this.nameInput.value = "";
      await this.refresh();
    } catch (err) {
      this.onNote(errText(err), "error");
    }
  }

  async load(name) {
    if (this.doc.size && !window.confirm(`Replace the current canvas with "${name}"?`)) return;
    try {
      const saved = await api(`/graphs/library/${encodeURIComponent(name)}`);
      this.doc.load(saved.graph || {});
      const unknown = this.doc.unknownClasses();
      this.onNote(
        `Loaded "${name}" (${this.doc.size} nodes).` +
          (unknown.length ? ` Unknown classes kept: ${unknown.join(", ")}.` : ""),
        unknown.length ? "warn" : "success",
      );
      if (this.onLoaded) this.onLoaded();
    } catch (err) {
      this.onNote(errText(err), "error");
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

  importLegacy() {
    const raw = window.localStorage.getItem("ng_graph_v1");
    if (!raw) {
      this.onNote('No legacy draft found in localStorage ("ng_graph_v1").', "warn");
      return;
    }
    if (this.doc.size && !window.confirm("Replace the current canvas with the legacy draft?")) return;
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch {
      this.onNote("Legacy draft is not valid JSON -- not imported.", "error");
      return;
    }
    this.doc.load(parsed);
    const unknown = this.doc.unknownClasses();
    this.onNote(
      `Imported legacy draft (${this.doc.size} nodes).` +
        (unknown.length ? ` Unknown classes kept: ${unknown.join(", ")}.` : ""),
      unknown.length ? "warn" : "success",
    );
    if (this.onLoaded) this.onLoaded();
  }
}

function _miniBtn(label, onClick, extraClass = "") {
  const b = document.createElement("button");
  b.className = `btn btn-secondary btn-small ${extraClass}`.trim();
  b.style.padding = "0.1rem 0.4rem";
  b.style.fontSize = "0.68rem";
  b.textContent = label;
  b.addEventListener("click", (e) => {
    e.stopPropagation();
    onClick();
  });
  return b;
}

