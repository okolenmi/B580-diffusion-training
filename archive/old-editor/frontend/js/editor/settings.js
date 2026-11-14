/* ---------------------------------------------------------------------------
   editor/settings.js -- the graph's own settings (memory budget), as a
   section of the editor.

   Until now a graph had no place to say how much device memory it may use:
   the editor sent only nodes and edges, so every run took the server's
   default whatever the person wanted. These settings are stored *in the
   graph* (GraphDoc.memory), so a saved graph, a run, and a template built
   on the graph all carry the same numbers.

   The list of settings and how each is drawn comes from the backend
   (`graph_fields` of POST /templates/modules -- backend/domain/modules.py),
   and the controls are the Templates page's own (board/fields.js), so the
   editor and a template show the same setting the same way. Nothing about
   a setting is written down a second time here.
   --------------------------------------------------------------------------- */

import { buildFieldControl } from "../board/fields.js";
import { graphRef } from "../board/model.js";

/** Hold only these in the section; `advanced` ones sit behind a toggle. */
export class GraphSettings {
  /**
   * @param {HTMLElement} root
   * @param {import("./state.js").GraphDoc} graphDoc
   * @param {object[]} fields   `graph_fields` from /templates/modules
   * @param {(msg: string, kind?: string) => void} [onNote]
   */
  constructor(root, graphDoc, fields, onNote) {
    this.root = root;
    this.gd = graphDoc;
    this.fields = fields;
    this.onNote = onNote;
    this.showAdvanced = false;
    // Adapter: board/model.js reads `doc.graph.memory`; point it at the
    // editor's own document so there is one copy of the values.
    const gd = graphDoc;
    this.adapter = {
      graph: {
        nodes: [],
        get memory() { return gd.memory; },
        set memory(v) { gd.memory = v; },
      },
    };
    this.render();
  }

  render() {
    this.root.replaceChildren();
    const visible = this.fields.filter((f) => this.showAdvanced || !f.advanced);
    for (const f of visible) {
      const row = document.createElement("div");
      row.className = "gs-row";
      const label = document.createElement("label");
      label.className = "gs-label";
      label.textContent = f.label || f.name;
      label.title = f.doc || "";
      const ref = graphRef("field", f.name);
      const { el } = buildFieldControl({
        block: { ref },
        descriptor: f,
        siblings: this.fields,
        doc: this.adapter,
        // buildFieldControl commits through setFieldValue on the adapter;
        // tell the editor (so it re-asks the memory preview).
        onChange: () => this.gd.changed("memory", "settings"),
        onNote: this.onNote,
      });
      row.append(label, el);
      this.root.append(row);
    }
    if (this.fields.some((f) => f.advanced)) {
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "btn btn-secondary btn-small gs-adv";
      toggle.textContent = this.showAdvanced ? "Hide advanced" : "Show advanced";
      toggle.addEventListener("click", () => { this.showAdvanced = !this.showAdvanced; this.render(); });
      this.root.append(toggle);
    }
  }
}
