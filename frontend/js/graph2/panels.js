/* ---------------------------------------------------------------------------
   graph2/panels.js -- the two read-only side panels: what the server found
   wrong with the graph (issues) and whether it will fit the card (memory).
   Every string reaches the DOM through textContent: holder names and
   messages come from a running graph and are never markup.
   --------------------------------------------------------------------------- */

import { api } from "../api.js";
import { errText } from "../lib/errors.js";
import { describeRefusal, panelModel } from "../lib/memory_panel.js";

const row = (className, text) => {
  const node = document.createElement("div");
  node.className = className;
  node.textContent = text;
  return node;
};

export class IssuesPanel {
  constructor(root, { onFocus }) {
    this.root = root;
    this.onFocus = onFocus;
  }

  show(issues) {
    this.root.replaceChildren();
    if (!issues.length) return void this.root.append(row("console-line success", "Graph is valid."));
    for (const issue of issues) {
      const line = row(`issue ${issue.severity === "error" ? "error" : "warn"}`, "");
      if (issue.node_id) {
        const chip = row("issue-node", issue.node_id);
        chip.title = "Select this node";
        chip.addEventListener("click", () => this.onFocus(issue.node_id));
        line.append(chip);
      }
      line.append(Object.assign(document.createElement("span"), { textContent: issue.message + (issue.param ? ` [${issue.param}]` : "") }));
      this.root.append(line);
    }
  }
}

export class MemoryPanel {
  constructor(root, doc, { onNote, wait = 400 }) {
    this.root = root;
    this.doc = doc;
    this.onNote = onNote;
    this.wait = wait;
    this._timer = null;
    this._seq = 0;
  }

  /** The answer depends on the live ledger, so it is asked again after every change and every run. */
  refreshLater() {
    clearTimeout(this._timer);
    this._timer = setTimeout(() => this.refresh(), this.wait);
  }

  async refresh() {
    if (!this.doc.nodes.size) return void this.root.replaceChildren(row("console-line info", "Empty graph."));
    const seq = ++this._seq;
    try {
      const preview = await api("/graphs/memory-preview", { method: "POST", body: { graph: this.doc.runPayload() } });
      if (seq === this._seq) this.show(preview);
    } catch (err) {
      if (seq !== this._seq) return;
      this.show(null);
      this.onNote(`Memory preview unavailable: ${errText(err)}`, "warn");
    }
  }

  show(preview) {
    const model = panelModel(preview);
    this.root.replaceChildren();
    for (const [label, value] of model.settings) this.root.append(row("mem-row", `${label}: ${value}`));
    this.root.append(row(`issue ${model.fit.level === "bad" ? "error" : model.fit.level}`, model.fit.headline));
    this.root.append(row("console-line info", model.fit.detail), row("console-line info", model.peak));
    for (const holder of model.holders) this.root.append(row("mem-row", `${holder.owner} holds ${holder.mb}`));
  }

  refusal(details, reason) {
    const refusal = describeRefusal(details, reason);
    this.root.replaceChildren(row("issue error", "Run refused"), row("console-line info", refusal.headline));
    for (const [label, value] of refusal.rows) this.root.append(row("mem-row", `${label}: ${value}`));
    for (const holder of refusal.holders) this.root.append(row("mem-row", `${holder.owner} holds ${holder.mb}`));
  }
}
