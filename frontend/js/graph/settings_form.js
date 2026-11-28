/* ---------------------------------------------------------------------------
   graph/settings_form.js -- the rows of the graph's own settings (settings.md): memory budget, precision, runtime
   options, seed. The list comes from the catalog's SettingSpecs and each row is drawn by the same `buildControl` a
   node port uses; values live in the document's `settings` (a key left out takes its default). Cross-key rules are
   the server's: it names them when the graph is checked.

   A row says what it does in words under its control (not in a tooltip), marks a value that differs from the default
   and can be put back with one click.
   --------------------------------------------------------------------------- */

import { buildControl } from "./widgets.js";

const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** Whether the document holds a value for `spec` that is not its default. */
export const isChanged = (doc, spec) => spec.key in doc.settings && !same(doc.settings[spec.key], spec.default);

export class SettingsForm {
  /** @param {HTMLElement} root @param {object} doc @param {object[]} specs @param {{note: Function}} deps */
  constructor(root, doc, specs, { note }) {
    this.root = root;
    this.doc = doc;
    this.specs = specs;
    this.note = note;
    this.showAdvanced = false;
    this.open = new Set(["Memory"]); // groups the person left open, kept across redraws
    this.render();
    doc.on("reset", () => this.render());
  }

  changed() {
    return this.specs.filter((spec) => isChanged(this.doc, spec));
  }

  /** Every setting back to its default. */
  resetAll() {
    for (const spec of this.changed()) this.doc.setSetting(spec.key, undefined);
    this.render();
  }

  setAdvanced(on) {
    this.showAdvanced = on;
    this.render();
  }

  render() {
    this.root.replaceChildren();
    const groups = new Map();
    for (const spec of this.specs) {
      if (spec.advanced && !this.showAdvanced && !isChanged(this.doc, spec)) continue;
      const name = spec.group || "Other";
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(spec);
    }
    for (const [name, list] of groups) {
      const details = make("details", "g2-setting-group");
      details.open = this.open.has(name);
      details.addEventListener("toggle", () => { if (details.open) this.open.add(name); else this.open.delete(name); });
      const summary = make("summary", "", name);
      const count = list.filter((spec) => isChanged(this.doc, spec)).length;
      if (count) summary.append(make("span", "gs-count", `${count} changed`));
      details.append(summary);
      for (const spec of list) details.append(this._row(spec));
      this.root.append(details);
    }
  }

  _row(spec) {
    const row = make("div", `gs-row${isChanged(this.doc, spec) ? " is-changed" : ""}`);
    const head = make("div", "gs-head");
    head.append(make("label", "gs-label", spec.unit ? `${spec.label} (${spec.unit})` : spec.label));
    if (isChanged(this.doc, spec)) {
      const back = make("button", "gs-reset", "Reset");
      back.type = "button";
      back.title = `Back to the default (${spec.default})`;
      back.addEventListener("click", () => { this.doc.setSetting(spec.key, undefined); this.render(); });
      head.append(back);
    }
    row.append(head, buildControl(spec, {
      value: this.doc.settings[spec.key],
      onDraft: () => {},
      onCommit: (value) => { this.doc.setSetting(spec.key, value); this._refreshMarks(); },
      note: this.note,
    }));
    if (spec.doc) row.append(make("div", "gs-doc", spec.doc));
    return row;
  }

  /** After a commit the control keeps focus: only the marks (changed rows, counts) are drawn again, by redrawing when the changed set moved. */
  _refreshMarks() {
    const now = this.changed().map((spec) => spec.key).join("|");
    if (now === this._marked) return;
    this._marked = now;
    this.render();
  }
}
