/* ---------------------------------------------------------------------------
   editor/inspector.js -- selected-node panel (M7).

   Renders the params form from the class's INPUTS:
     - widget_only  -> widget, no socket (checkboxes etc.)
     - primitive    -> socket + widget (wired value overrides the param)
     - otherwise    -> pure handle: read-only "wire" row, never a param
   `visible_when` gates rows (value is preserved while hidden -- docs
   resources-controller, Node.Port contract). Choices -> select, bool ->
   checkbox, int/float -> number, list/dict/tuple -> JSON textarea, Path/
   str -> text (path_kind is a picker hint; the picker itself is deferred).

   Also: node id rename (with edge rewiring), delete, diagnostics button
   (POST /graphs/nodes/{class}/diagnostics) when the class declares them.

   Committing a value writes node.params and calls doc.changed("params"):
   the canvas summary refreshes, the inspector does NOT rebuild itself on
   every keystroke (it re-renders after select/checkbox commits, where a
   visible_when gate may have moved).
   --------------------------------------------------------------------------- */

import { api, ApiError } from "../api.js";
import { isWidgetInput } from "./state.js";

const JSON_TYPES = new Set(["list", "dict", "tuple"]);

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
    for (const p of cls.inputs) this.root.appendChild(this._inputRow(node, p));

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

  _inputRow(node, p) {
    const row = this._row(p.name);
    const wired = this.doc.edgeInto(node.id, p.name);

    // gated visibility: (other_param, value | [values])
    let gated = false;
    if (p.visible_when) {
      const [gate, accepted] = p.visible_when;
      const cur = node.params[gate];
      const want = Array.isArray(accepted) ? accepted.includes(cur) : cur === accepted;
      gated = !want;
    }

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
    if (gated) row.classList.add("gated");

    const input = this._widget(node, p);
    row.appendChild(input);

    if (wired) {
      const hint = document.createElement("span");
      hint.className = "param-wired";
      hint.textContent = `overridden by ${wired.from_node}.${wired.from_port}`;
      row.appendChild(hint);
    } else if (p.path_kind) {
      const hint = document.createElement("span");
      hint.className = "param-hint";
      hint.textContent = `path (${p.path_kind})`;
      row.appendChild(hint);
    }
    return row;
  }

  _widget(node, p) {
    const commit = (value, rerender) => {
      if (value === undefined || value === null || value === "") delete node.params[p.name];
      else node.params[p.name] = value;
      this.doc.changed("params");
      if (rerender) this.render();
    };

    if (p.choices && p.choices.length) {
      const sel = document.createElement("select");
      sel.className = "cfg-input";
      const def = document.createElement("option");
      def.value = "";
      def.textContent = p.default !== null && p.default !== undefined && p.default !== ""
        ? `(default: ${p.default})`
        : "(default)";
      sel.appendChild(def);
      for (const c of p.choices) {
        const o = document.createElement("option");
        o.value = c;
        o.textContent = c;
        sel.appendChild(o);
      }
      const cur = node.params[p.name];
      sel.value = cur === undefined || cur === null ? "" : String(cur);
      sel.addEventListener("change", () => commit(sel.value, true));
      return sel;
    }

    if (p.type === "bool") {
      const wrap = document.createElement("label");
      wrap.className = "param-label";
      wrap.style.justifySelf = "start";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = node.params[p.name] === true;
      box.addEventListener("change", () => {
        // store explicit false too: unchecking must override an earlier true
        node.params[p.name] = box.checked;
        this.doc.changed("params");
        this.render(); // a gated sibling may show/hide
      });
      wrap.append(box, document.createTextNode(" " + (box.checked ? "true" : "false")));
      return wrap;
    }

    if (p.type === "int" || p.type === "float") {
      const input = document.createElement("input");
      input.className = "cfg-input";
      input.type = "number";
      if (p.type === "float") input.step = "any";
      const cur = node.params[p.name];
      input.value = cur === undefined || cur === null ? "" : String(cur);
      if (p.default !== null && p.default !== undefined && input.value === "") {
        input.placeholder = String(p.default);
      }
      input.addEventListener("change", () => {
        if (input.value === "") return commit(undefined);
        const n = p.type === "int" ? parseInt(input.value, 10) : parseFloat(input.value);
        if (Number.isNaN(n)) {
          input.value = "";
          this.onNote(`${p.name}: not a ${p.type}, value cleared.`, "warn");
          return commit(undefined);
        }
        commit(n);
      });
      return input;
    }

    if (JSON_TYPES.has(p.type)) {
      const input = document.createElement("textarea");
      input.className = "cfg-input";
      input.rows = 2;
      const cur = node.params[p.name];
      input.value = cur === undefined || cur === null
        ? ""
        : typeof cur === "object"
          ? JSON.stringify(cur)
          : String(cur);
      input.addEventListener("change", () => {
        const raw = input.value.trim();
        if (raw === "") return commit(undefined);
        try {
          commit(JSON.parse(raw), true);
        } catch {
          this.onNote(`${p.name}: invalid JSON -- keeping previous value.`, "warn");
          input.value = typeof cur === "object" ? JSON.stringify(cur) : String(cur ?? "");
        }
      });
      return input;
    }

    // str / Path / anything else: text
    const input = document.createElement("input");
    input.className = "cfg-input";
    input.type = "text";
    const cur = node.params[p.name];
    input.value = cur === undefined || cur === null ? "" : String(cur);
    if (p.default_repr !== null && p.default_repr !== undefined && input.value === "") {
      input.placeholder = p.default_repr;
    }
    input.title = p.doc || p.name;
    input.addEventListener("change", () => commit(input.value.trim() === "" ? undefined : input.value));
    return input;
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
