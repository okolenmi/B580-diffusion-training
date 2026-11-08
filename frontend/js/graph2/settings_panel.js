/* ---------------------------------------------------------------------------
   graph2/settings_panel.js -- the graph's own settings (settings.md): memory
   budget, precision, runtime options, seed. The list comes from the catalog's
   SettingSpecs and each row is drawn by the same `buildControl` a node port
   uses; values live in the document's `settings` (a key left out takes its
   default). Cross-key rules are the server's: it names them when the graph
   is checked.
   --------------------------------------------------------------------------- */

import { buildControl } from "./widgets.js";

const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

export class SettingsPanel {
  constructor(root, doc, specs, { note }) {
    this.root = root;
    this.doc = doc;
    this.specs = specs;
    this.note = note;
    this.showAdvanced = false;
    this.render();
    doc.on("reset", () => this.render());
  }

  render() {
    this.root.replaceChildren();
    const groups = new Map();
    for (const spec of this.specs) {
      if (spec.advanced && !this.showAdvanced) continue;
      const name = spec.group || "Other";
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(spec);
    }
    for (const [name, list] of groups) {
      const details = make("details", "g2-setting-group");
      details.open = name === "Memory";
      details.append(make("summary", "", name));
      for (const spec of list) {
        const row = make("div", "gs-row");
        const label = make("label", "gs-label", spec.unit ? `${spec.label} (${spec.unit})` : spec.label);
        label.title = spec.doc || spec.key;
        row.append(label, buildControl(spec, {
          value: this.doc.settings[spec.key],
          onDraft: () => {},
          onCommit: (value) => this.doc.setSetting(spec.key, value),
          note: this.note,
        }));
        details.append(row);
      }
      this.root.append(details);
    }
    if (this.specs.some((s) => s.advanced)) {
      const toggle = make("button", "btn btn-secondary btn-small gs-adv", this.showAdvanced ? "Hide advanced" : "Show advanced");
      toggle.type = "button";
      toggle.addEventListener("click", () => { this.showAdvanced = !this.showAdvanced; this.render(); });
      this.root.append(toggle);
    }
  }
}
