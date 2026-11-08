/* ---------------------------------------------------------------------------
   graph2/inspector.js -- what the open selection says beyond the card:
   the class and its description, what the server said when it resolved the
   node (message, choices still to make), the node's last result, and live
   advice on the values being edited (`diagnostics`, one request per pause
   in typing; a failure shows as one line, never as a broken panel).
   --------------------------------------------------------------------------- */

import { api } from "../api.js";
import { errText } from "../lib/errors.js";

const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

export class Inspector {
  constructor(root, { doc, shapes, views, wait = 500 }) {
    this.root = root;
    this.doc = doc;
    this.shapes = shapes;
    this.views = views;
    this.wait = wait;
    this.uid = null;
    this._timer = null;
    this._seq = 0;
    doc.on("node:params", (e) => { if (e.uid === this.uid) this._askLater(); });
    doc.on("node:remove", (e) => { if (e.uid === this.uid) this.show(null); });
    shapes.onShape((e) => { if (e.uid === this.uid) this._draw(); });
  }

  show(uid) {
    this.uid = uid;
    this._hints = null;
    this._draw();
    if (uid) this._ask();
  }

  _askLater() {
    clearTimeout(this._timer);
    this._timer = setTimeout(() => this._ask(), this.wait);
  }

  async _ask() {
    const node = this.doc.nodes.get(this.uid);
    if (!node) return;
    const seq = ++this._seq;
    try {
      const hints = await api(`/graphs/nodes/${encodeURIComponent(node.class_name)}/diagnostics`, { method: "POST", body: { params: node.params } });
      if (seq !== this._seq) return;
      this._hints = { hints: hints.hints || hints, error: null };
    } catch (err) {
      if (seq !== this._seq) return;
      this._hints = { hints: {}, error: errText(err) };
    }
    this._draw();
  }

  _draw() {
    this.root.replaceChildren();
    const node = this.doc.nodes.get(this.uid);
    if (!node) {
      this.root.append(make("div", "console-line info", "Select a node."));
      return;
    }
    const info = this.shapes.info(node.class_name);
    this.root.append(make("div", "g2-insp-title", info ? info.display_name : node.class_name));
    this.root.append(make("div", "g2-insp-uid", `${node.class_name} · ${node.uid}`));
    if (info?.doc) this.root.append(make("div", "g2-insp-doc", info.doc));
    const resolved = this.shapes.stateOf(node.uid);
    if (resolved.message) this.root.append(make("div", `console-line ${resolved.state === "error" ? "error" : "warn"}`, resolved.message));
    const status = this.views.get(node.uid)?.status;
    if (status && status.state !== "idle") this.root.append(make("div", `console-line ${status.state === "error" ? "error" : "info"}`, `last run: ${status.state}${status.text ? ` ${status.text}` : ""}${status.title ? ` -- ${status.title}` : ""}`));
    if (this._hints?.error) this.root.append(make("div", "console-line warn", `advice unavailable: ${this._hints.error}`));
    for (const [input, lines] of Object.entries(this._hints?.hints || {})) {
      for (const line of [].concat(lines)) this.root.append(make("div", "console-line info", `${input}: ${line}`));
    }
  }
}
