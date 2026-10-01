/* ---------------------------------------------------------------------------
   editor/inspector.js -- selected-node panel (M7).

   Renders the params form from the class's INPUTS:
     - widget_only  -> widget, no socket (checkboxes etc.)
     - primitive    -> socket + widget (wired value overrides the param)
     - otherwise    -> pure handle: read-only "wire" row, never a param
   `visible_when` gates rows (value is preserved while hidden -- docs
   resources-controller, Node.Port contract). The controls themselves
   come from editor/widgets.js -- the SAME builder the node body uses,
   so a value looks and behaves identically on canvas and here
   (path_kind ports get the server-folder picker + upload).

   Also: node id rename (with edge rewiring), delete, diagnostics button
   (POST /graphs/nodes/{class}/diagnostics) when the class declares them.

   Committing a value writes node.params and calls
   doc.changed("params", "inspector"): the canvas re-renders (its node
   widgets follow), but the inspector does NOT rebuild itself on every
   keystroke -- it rebuilds after select/checkbox/JSON/path commits (a
   visible_when gate or the picker catalog may have moved), and
   editor.js refreshes it when the change came from anywhere else.
   --------------------------------------------------------------------------- */

import { api, ApiError } from "../api.js";
import { isWidgetInput } from "./state.js";
import { buildWidget, rowVisible } from "./widgets.js";

export class Inspector {
  constructor(root, doc, { onNote }) {
    this.root = root;
    this.doc = doc;
    this.onNote = onNote;
    this.nodeId = null;
  }

  select(nodeId) {
    this.nodeId = nodeId;
    this.render();
  }

  /** Called on structural doc changes: drop stale selection, re-render. */
  refresh() {
    if (this.nodeId && !this.doc.nodes.has(this.nodeId)) this.nodeId = null;
    this.render();
  }

  render() {
    const node = this.nodeId ? this.doc.nodes.get(this.nodeId) : null;
    this.root.replaceChildren();
    if (!node) {
      const line = document.createElement("div");
      line.className = "console-line info";
      line.textContent = "Select a node.";
      this.root.appendChild(line);
      return;
    }
    const cls = this.doc.classOf(node);

    const title = document.createElement("div");
    title.className = "insp-title";
    title.textContent = cls ? cls.display_name : node.class_name;
    this.root.appendChild(title);

    // -- identity: editable id + class name
    const idRow = this._row("id");
    const idInput = document.createElement("input");
    idInput.className = "cfg-input";
    idInput.type = "text";
    idInput.value = node.id;
    idInput.title = "Node id (used by edges and the run payload)";
    idInput.addEventListener("change", () => {
      const newId = idInput.value.trim();
      if (!newId) {
        idInput.value = node.id;
        this.onNote("Node id must not be empty.", "warn");
        return;
      }
      // Point the inspector at the NEW id before mutating: renameNode fires
      // onChange synchronously and the refresh would otherwise drop a stale id.
      if (newId !== node.id) this.nodeId = newId;
      const ok = this.doc.renameNode(node.id, newId);
      if (!ok) {
        this.nodeId = node.id;
        idInput.value = node.id;
        this.onNote(`Cannot use "${newId}" as a node id (spaces or already taken).`, "warn");
        return;
      }
      if (newId !== node.id) this.onNote(`Node id -> ${newId}.`);
      // structure change re-renders canvas + inspector via doc.onChange
    });
    idRow.appendChild(idInput);
    this.root.appendChild(idRow);

    const clsLine = document.createElement("div");
    clsLine.className = "insp-class";
    clsLine.textContent = node.class_name + (cls ? "" : "  (unknown class)");
    this.root.appendChild(clsLine);

    if (cls && cls.doc) {
      const docEl = document.createElement("div");
      docEl.className = "insp-doc";
      docEl.textContent = cls.doc;
      this.root.appendChild(docEl);
    }

    if (!cls) {
      const warn = document.createElement("div");
      warn.className = "console-line error";
      warn.textContent = "Class not in the catalog -- validation will flag it; delete and re-add from the palette.";
      this.root.appendChild(warn);
      this._actions(node);
      return;
    }

    // -- params
    for (const p of cls.inputs) this.root.appendChild(this._inputRow(node, cls, p));

    if (cls.has_diagnostics) {
      const actions = document.createElement("div");
      actions.className = "insp-actions";
      const btn = document.createElement("button");
      btn.className = "btn btn-secondary btn-small";
      btn.textContent = "Run diagnostics";
      btn.addEventListener("click", () => this._diagnose(node, cls));
      actions.appendChild(btn);
      this.root.appendChild(actions);
    }

    this._actions(node);
  }

  _row(labelText) {
    const row = document.createElement("div");
    row.className = "param-row";
    const label = document.createElement("span");
    label.className = "param-label";
    label.textContent = labelText;
    row.appendChild(label);
    return row;
  }

  _inputRow(node, cls, p) {
    const row = this._row(p.name);
    const wired = this.doc.edgeInto(node.id, p.name);

    if (!isWidgetInput(p)) {
      // pure handle: wire only
      const span = document.createElement("span");
      span.className = "param-wired";
      span.textContent = wired
        ? `<- ${wired.from_node}.${wired.from_port}`
        : p.required
          ? "not connected (required)"
          : "not connected";
      row.appendChild(span);
      return row;
    }

    if (p.required) {
      const star = document.createElement("span");
      star.className = "req";
      star.textContent = " *";
      row.querySelector(".param-label").appendChild(star);
    }
    row.querySelector(".param-label").title = p.doc ? `${p.name}: ${p.doc}` : p.name;
    if (!rowVisible(node, p, cls.inputs)) row.classList.add("gated");

    row.appendChild(
      buildWidget({
        doc: this.doc,
        node,
        port: p,
        origin: "inspector",
        onNote: this.onNote,
        onRebuild: () => this.render(),
      }),
    );

    if (wired) {
      const hint = document.createElement("span");
      hint.className = "param-wired";
      hint.textContent = `overridden by ${wired.from_node}.${wired.from_port}`;
      row.appendChild(hint);
    }
    return row;
  }

  _actions(node) {
    const bar = document.createElement("div");
    bar.className = "insp-actions";
    const del = document.createElement("button");
    del.className = "btn btn-danger btn-small";
    del.textContent = "Delete node";
    del.addEventListener("click", () => {
      const id = node.id;
      this.doc.removeNode(id);
      this.nodeId = null;
      this.onNote(`Node ${id} removed.`);
    });
    bar.appendChild(del);
    this.root.appendChild(bar);
  }

  async _diagnose(node, cls) {
    try {
      const res = await api(`/graphs/nodes/${encodeURIComponent(cls.class_name)}/diagnostics`, {
        method: "POST",
        body: { params: this.doc.paramsFor(node) },
      });
      const lines = [];
      for (const [input, msgs] of Object.entries(res.messages || {})) {
        for (const m of msgs) lines.push(`${input}: ${m}`);
      }
      const box = document.createElement("div");
      box.className = "diag-box";
      box.textContent = lines.length ? lines.join("\n") : "No diagnostics messages.";
      const old = this.root.querySelector(".diag-box");
      if (old) old.remove();
      this.root.appendChild(box);
      this.onNote(`Diagnostics for ${node.id}: ${lines.length} message(s).`);
    } catch (err) {
      if (err instanceof ApiError) this.onNote(`${err.code}: ${err.message}`, "error");
      else this.onNote(String(err), "error");
    }
  }
}
