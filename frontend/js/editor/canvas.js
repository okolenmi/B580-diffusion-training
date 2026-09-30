/* ---------------------------------------------------------------------------
   editor/canvas.js -- graph surface rendering + interaction (M7).

   DOM nodes absolutely positioned on a scrollable plane, edges in one SVG
   layer. Structural change -> full re-render; a node drag moves the element
   directly and only touches the `d` of the edges that node participates in
   (socket offsets are node-relative, so they don't shift while it moves).

   Socket centers are measured with offsetLeft/offsetTop -- the node is the
   offsetParent (only positioned ancestor), so offsets are node-relative and
   absolute edge coords are node.x/y + offset.

   Callbacks: onSelect(selection | null) with {kind:"node", id} or
   {kind:"edge", edge}; onNote(message, kind) for the page console.
   --------------------------------------------------------------------------- */

import { hasSocket, typesCompatible } from "./state.js";

const SVG_NS = "http://www.w3.org/2000/svg";

export class Canvas {
  constructor(inner, doc, { onSelect, onNote } = {}) {
    this.inner = inner;   // #canvas-inner -- nodes + <svg> live here
    this.svg = inner.querySelector("#edge-layer");
    this.doc = doc;
    this.onSelect = onSelect || (() => {});
    this.onNote = onNote || (() => {});
    this.selection = null;          // {kind, id | edge}
    this._sockets = new Map();      // nodeId -> Map("dir:port" -> {x, y})
    this._badges = new Map();       // nodeId -> {kind, text} execution status
    this._connect = null;           // active wire drag
    this._drag = null;              // active node drag
    this._suppressClick = false;    // set after a wire drop, eats the click
    this._bind();
  }

  /* ================= rendering ================= */

  render() {
    if (this.selection && !this._selectionAlive()) this.selection = null;
    for (const el of [...this.inner.querySelectorAll(".gnode")]) el.remove();
    this._sockets.clear();
    for (const node of this.doc.nodes.values()) {
      const el = this._nodeEl(node);
      this.inner.appendChild(el);
      this._measure(el, node);
    }
    this._applySelection();
    this._redrawEdges();
    this._restoreBadges();
  }

  _selectionAlive() {
    if (this.selection.kind === "node") return this.doc.nodes.has(this.selection.id);
    return this.doc.edges.includes(this.selection.edge);
  }

  _nodeEl(node) {
    const cls = this.doc.classOf(node);
    const el = document.createElement("div");
    el.className = "gnode" + (cls ? "" : " unknown");
    el.dataset.id = node.id;
    el.style.left = node.x + "px";
    el.style.top = node.y + "px";

    const head = document.createElement("div");
    head.className = "gnode-head";
    const title = document.createElement("span");
    title.className = "gnode-title";
    title.textContent = cls ? cls.display_name : node.class_name;
    const badge = document.createElement("span");
    badge.className = "gnode-badge";
    badge.dataset.badge = node.id;
    head.append(title, badge);
    el.appendChild(head);

    const body = document.createElement("div");
    body.className = "gnode-body";
    const colIn = document.createElement("div");
    colIn.className = "gnode-col in";
    const mid = document.createElement("div");
    mid.className = "gnode-params";
    const colOut = document.createElement("div");
    colOut.className = "gnode-col out";
    body.append(colIn, mid, colOut);

    if (!cls) {
      mid.textContent = "(unknown class)";
    } else {
      for (const p of cls.inputs) colIn.appendChild(this._portRow(node, p, "in"));
      for (const p of cls.outputs) colOut.appendChild(this._portRow(node, p, "out"));
      const summary = Object.entries(this.doc.paramsFor(node))
        .slice(0, 3)
        .map(([k, v]) => `${k}=${typeof v === "object" ? "…" : String(v)}`)
        .join("\n");
      const more = Object.keys(this.doc.paramsFor(node)).length > 3 ? "\n…" : "";
      mid.textContent = summary + more || "no params";
    }
    el.appendChild(body);
    return el;
  }

  _portRow(node, p, dir) {
    const row = document.createElement("div");
    row.className = "gport-row";
    if (dir === "in" && hasSocket(p)) row.appendChild(this._socket(p, "in"));
    const label = document.createElement("span");
    label.className = "gport-label";
    label.textContent = p.name + (dir === "in" && p.required ? " *" : "");
    label.title = p.doc ? `${p.name} (${p.type}) -- ${p.doc}` : `${p.name} (${p.type})`;
    row.appendChild(label);
    if (dir === "out") row.appendChild(this._socket(p, "out"));
    return row;
  }

  _socket(p, dir) {
    const s = document.createElement("span");
    s.className = "gport " + dir;
    s.dataset.port = p.name;
    s.dataset.dir = dir;
    s.title = `${p.name} (${p.type})`;
    return s;
  }

  _measure(el, node) {
    const map = new Map();
    for (const s of el.querySelectorAll(".gport")) {
      map.set(s.dataset.dir + ":" + s.dataset.port, {
        x: node.x + s.offsetLeft + s.offsetWidth / 2,
        y: node.y + s.offsetTop + s.offsetHeight / 2,
      });
    }
    this._sockets.set(node.id, map);
  }

  _redrawEdges() {
    this.svg.replaceChildren();
    this.doc.edges.forEach((edge, index) => {
      const path = document.createElementNS(SVG_NS, "path");
      path.setAttribute("class", "edge");
      path._edge = edge;
      path._index = index;
      const a = this._pos(edge.from_node, "out", edge.from_port);
      const b = this._pos(edge.to_node, "in", edge.to_port);
      if (a && b) path.setAttribute("d", this._curve(a, b));
      this.svg.appendChild(path);
    });
  }

  _pos(nodeId, dir, port) {
    return (this._sockets.get(nodeId) || {}).get(dir + ":" + port) || null;
  }

  _curve(a, b) {
    const dx = Math.max(40, Math.abs(b.x - a.x) / 2);
    return `M ${a.x} ${a.y} C ${a.x + dx} ${a.y}, ${b.x - dx} ${b.y}, ${b.x} ${b.y}`;
  }

  _updateEdgesFor(nodeId) {
    for (const path of this.svg.querySelectorAll("path.edge")) {
      const e = path._edge;
      if (!e || (e.from_node !== nodeId && e.to_node !== nodeId)) continue;
      const a = this._pos(e.from_node, "out", e.from_port);
      const b = this._pos(e.to_node, "in", e.to_port);
      if (a && b) path.setAttribute("d", this._curve(a, b));
    }
  }

  /* ================= selection ================= */

  selectNode(id) {
    if (!this.doc.nodes.has(id)) return;
    this.selection = { kind: "node", id };
    this._applySelection();
    this.onSelect(this.selection);
  }

  clearSelection() {
    this.selection = null;
    this._applySelection();
    this.onSelect(null);
  }

  _applySelection() {
    for (const el of this.inner.querySelectorAll(".gnode")) {
      el.classList.toggle(
        "selected",
        !!this.selection && this.selection.kind === "node" && this.selection.id === el.dataset.id,
      );
    }
    for (const p of this.svg.querySelectorAll("path.edge")) {
      p.classList.toggle(
        "selected",
        !!this.selection && this.selection.kind === "edge" && this.selection.edge === p._edge,
      );
    }
  }

  focusNode(id) {
    const el = [...this.inner.querySelectorAll(".gnode")].find((n) => n.dataset.id === id);
    if (el) el.scrollIntoView({ block: "center", inline: "center", behavior: "smooth" });
    this.selectNode(id);
  }

  /* ================= execution status badges ================= */

  setNodeStatus(id, kind, text) {
    this._badges.set(id, { kind, text });
    this._applyBadge(id);
  }

  clearStatuses() {
    this._badges.clear();
    for (const badge of this.inner.querySelectorAll(".gnode-badge")) {
      badge.className = "gnode-badge";
      badge.textContent = "";
    }
  }

  _applyBadge(id) {
    const badge = this.inner.querySelector(`[data-badge="${CSS.escape(id)}"]`);
    const state = this._badges.get(id);
    if (!badge) return;
    badge.className = state ? `gnode-badge ${state.kind}` : "gnode-badge";
    badge.textContent = state ? state.text : "";
  }

  /** Badges live in a map so structural/param re-renders keep live status. */
  _restoreBadges() {
    for (const id of this._badges.keys()) this._applyBadge(id);
  }

  /* ================= interaction ================= */

  _bind() {
    this.inner.addEventListener("pointerdown", (e) => this._pointerDown(e));
    this.inner.addEventListener("click", (e) => this._click(e));
    document.addEventListener("pointermove", (e) => this._pointerMove(e));
    document.addEventListener("pointerup", (e) => this._pointerUp(e));
    document.addEventListener("keydown", (e) => this._keyDown(e));
  }

  _pointInInner(e) {
    const rect = this.inner.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  _pointerDown(e) {
    // Any new press supersedes a pending suppress flag (e.g. a wire
    // released off-canvas never produces the trailing canvas click).
    this._suppressClick = false;
    const socket = e.target.closest(".gport");
    if (socket) {
      e.preventDefault();
      this._startConnect(socket);
      return;
    }
    const nodeEl = e.target.closest(".gnode");
    if (nodeEl) {
      const node = this.doc.nodes.get(nodeEl.dataset.id);
      if (!node) return;
      this.selectNode(node.id);
      if (e.target.closest(".gnode-head")) {
        e.preventDefault();
        const start = this._pointInInner(e);
        this._drag = { node, el: nodeEl, startX: start.x, startY: start.y, origX: node.x, origY: node.y };
      }
    }
  }

  _pointerMove(e) {
    if (this._drag) {
      const p = this._pointInInner(e);
      const node = this._drag.node;
      node.x = Math.max(0, Math.round(this._drag.origX + (p.x - this._drag.startX)));
      node.y = Math.max(0, Math.round(this._drag.origY + (p.y - this._drag.startY)));
      this._drag.el.style.left = node.x + "px";
      this._drag.el.style.top = node.y + "px";
      this._updateEdgesFor(node.id);
      return;
    }
    if (this._connect) {
      const p = this._pointInInner(e);
      this._connect.temp.setAttribute("d", this._curve(this._connect.origin, p));
    }
  }

  _pointerUp(e) {
    if (this._drag) {
      const moved = this._drag.node.x !== this._drag.origX || this._drag.node.y !== this._drag.origY;
      this._drag = null;
      if (moved) this.doc.changed("params"); // position only: canvas edges already updated
      return;
    }
    if (this._connect) {
      const target = document.elementFromPoint(e.clientX, e.clientY);
      const socket = target && target.closest ? target.closest(".gport") : null;
      this._finishConnect(socket);
    }
  }

  /* ---- wire dragging ---- */

  _startConnect(socket) {
    const row = socket.closest(".gnode");
    const node = this.doc.nodes.get(row.dataset.id);
    if (!node) return;
    const dir = socket.dataset.dir;
    const cls = this.doc.classOf(node);
    if (!cls) {
      this.onNote("Cannot wire an unknown class.", "warn");
      return;
    }
    const port = (dir === "in" ? cls.inputs : cls.outputs).find((p) => p.name === socket.dataset.port);
    if (!port) return;

    const origin = this._pos(node.id, dir, port.name);
    const temp = document.createElementNS(SVG_NS, "path");
    temp.setAttribute("class", "edge-temp");
    this.svg.appendChild(temp);

    // precompute compatible opposite-side sockets for the green highlight
    const ok = new Set();
    for (const other of this.doc.nodes.values()) {
      if (other.id === node.id) continue;
      const otherCls = this.doc.classOf(other);
      if (!otherCls) continue;
      const targets = dir === "out" ? otherCls.inputs : otherCls.outputs;
      for (const t of targets) {
        if (dir === "in" && !hasSocket(t)) continue;
        const compatible =
          dir === "out" ? typesCompatible(port, t) : typesCompatible(t, port);
        if (compatible) ok.add(other.id + "|" + (dir === "out" ? "in" : "out") + "|" + t.name);
      }
    }
    for (const s of this.inner.querySelectorAll(".gport")) {
      const key = s.closest(".gnode").dataset.id + "|" + s.dataset.dir + "|" + s.dataset.port;
      if (ok.has(key)) s.classList.add("ok");
    }
    socket.classList.add("src");
    this.inner.classList.add("connecting");
    this._connect = { node, port, dir, origin, temp, ok };
  }

  _finishConnect(targetSocket) {
    const c = this._connect;
    this._connect = null;
    c.temp.remove();
    this.inner.classList.remove("connecting");
    for (const s of this.inner.querySelectorAll(".gport.ok, .gport.src")) {
      s.classList.remove("ok", "src");
    }
    this._suppressClick = true;

    if (!targetSocket || !targetSocket.classList.contains("gport")) return;
    const targetNodeEl = targetSocket.closest(".gnode");
    const targetDir = targetSocket.dataset.dir;
    if (targetDir === c.dir || targetNodeEl.dataset.id === c.node.id) {
      this.onNote("Wires run output -> input, between different nodes.", "warn");
      return;
    }
    const key = targetNodeEl.dataset.id + "|" + targetDir + "|" + targetSocket.dataset.port;
    if (!c.ok.has(key)) {
      this.onNote("Incompatible port types.", "warn");
      return;
    }
    if (c.dir === "out") {
      this.doc.addEdge(c.node.id, c.port.name, targetNodeEl.dataset.id, targetSocket.dataset.port);
    } else {
      this.doc.addEdge(targetNodeEl.dataset.id, targetSocket.dataset.port, c.node.id, c.port.name);
    }
    // structure change re-rendered everything; restore selection
    this.selectNode(c.node.id);
  }

  /* ---- click / keys ---- */

  _click(e) {
    if (this._suppressClick) {
      this._suppressClick = false;
      return;
    }
    const path = e.target.closest ? e.target.closest("path.edge") : null;
    if (path && path._edge) {
      this.selection = { kind: "edge", edge: path._edge };
      this._applySelection();
      this.onSelect(this.selection);
      return;
    }
    if (e.target.closest(".gnode")) return; // handled on pointerdown
    // empty plane
    this.clearSelection();
  }

  _keyDown(e) {
    const tag = (document.activeElement && document.activeElement.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (e.key === "Escape" && this.selection) {
      this.clearSelection();
      return;
    }
    if ((e.key === "Delete" || e.key === "Backspace") && this.selection) {
      e.preventDefault();
      if (this.selection.kind === "node") {
        const id = this.selection.id;
        this.doc.removeNode(id);
        this.onNote(`Node ${id} removed.`);
      } else {
        this.doc.removeEdge(this.selection.edge);
        this.onNote("Wire removed.");
      }
      this.selection = null;
      this.onSelect(null);
    }
  }
}
