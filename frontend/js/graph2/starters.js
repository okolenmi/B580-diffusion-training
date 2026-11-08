/* ---------------------------------------------------------------------------
   graph2/starters.js -- "New from starter": the few choices made before a graph
   exists (checkpoint, dataset, output, length), drawn by the same controls a
   node port uses; the server returns the whole graph and the document loads it.
   The answers are kept per starter in this browser so the next graph starts
   from the last one.
   --------------------------------------------------------------------------- */

import { api } from "../api.js";
import { errText } from "../lib/errors.js";
import { buildControl } from "./widgets.js";

const KEY = "g2.starter.v1";

const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

function remembered(name) {
  try {
    return JSON.parse(localStorage.getItem(KEY))?.[name] || {};
  } catch {
    return {};
  }
}

function remember(name, values) {
  try {
    const all = JSON.parse(localStorage.getItem(KEY)) || {};
    localStorage.setItem(KEY, JSON.stringify({ ...all, [name]: values }));
  } catch { /* a convenience only */ }
}

export class StarterDialog {
  constructor(doc, { options, note, onBuilt }) {
    this.doc = doc;
    this.options = options;
    this.note = note;
    this.onBuilt = onBuilt;
    this.dialog = make("dialog", "g2-dialog");
    document.body.append(this.dialog);
    this._starters = null;
  }

  async open() {
    try {
      this._starters ||= (await api("/starters")).starters;
    } catch (err) {
      return this.note(`Starters unavailable: ${errText(err)}`, "error");
    }
    this._draw(this._starters[0]);
    this.dialog.showModal();
  }

  _draw(starter) {
    const values = { ...Object.fromEntries(starter.options.filter((o) => o.default !== undefined).map((o) => [o.name, o.default])), ...remembered(starter.name) };
    this.dialog.replaceChildren(make("h3", "", starter.label), make("p", "g2-insp-doc", starter.doc));
    for (const spec of starter.options) {
      const row = make("div", "g2-dialog-row");
      const label = make("label", "gs-label", spec.required ? `${spec.label} *` : spec.label);
      label.title = spec.doc || "";
      row.append(label, buildControl(spec, {
        value: values[spec.name],
        onDraft: () => {},
        onCommit: (value) => { values[spec.name] = value; },
        options: () => this.options(spec, { scope: spec.scope }),
        note: this.note,
      }));
      this.dialog.append(row);
    }
    const create = make("button", "btn btn-small", "Create");
    const cancel = make("button", "btn btn-secondary btn-small", "Cancel");
    create.type = cancel.type = "button";
    cancel.addEventListener("click", () => this.dialog.close());
    create.addEventListener("click", () => this._create(starter, values));
    this.dialog.append(make("div", "g2-dialog-actions"));
    this.dialog.lastChild.append(cancel, create);
  }

  async _create(starter, values) {
    if (this.doc.nodes.size && !window.confirm("Replace the current canvas with the new graph?")) return;
    try {
      const built = await api(`/starters/${encodeURIComponent(starter.name)}`, { method: "POST", body: { options: values } });
      remember(starter.name, values);
      this.doc.load(built);
      this.dialog.close();
      this.note(`${starter.label}: ${this.doc.nodes.size} nodes.`, "success");
      this.onBuilt?.();
    } catch (err) {
      this.note(errText(err), "error");
    }
  }
}
