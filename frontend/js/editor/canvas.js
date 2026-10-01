/* ---------------------------------------------------------------------------
   editor/canvas.js -- graph surface rendering + interaction (M7).

   An INFINITE plane: the viewport never scrolls (overflow: hidden, no
   scrollbars); pan comes from wheel + dragging the empty background, and
   the view translates via one CSS transform on #canvas-inner. A dashed
   circle marks the origin (0, 0) so "infinite" still has a visible
   center; nodes may live anywhere, including negative coordinates.

   DOM nodes absolutely positioned on the plane, edges in one SVG
   layer. Structural change -> full re-render; a node drag moves the element
   directly and only touches the `d` of the edges that node participates in
   (socket offsets are node-relative, so they don't shift while it moves).

   Node bodies are EDITABLE: each input row carries its widget from
   editor/widgets.js (the inspector builds the same controls), wired
   inputs show `<- node.port` instead, required-unconnected sockets
   read red, and classes with diagnostics render the server's per-input
   lines live under the row (debounced, see _scheduleDiagnostics).

   Socket centers are measured with offsetLeft/offsetTop -- the node is the
   offsetParent (only positioned ancestor), so offsets are node-relative and
   absolute edge coords are node.x/y + offset.

   Callbacks: onSelect(selection | null) with {kind:"node", id} or
   {kind:"edge", edge}; onNote(message, kind) for the floating console.
   --------------------------------------------------------------------------- */

import { api } from "../api.js";
import { hasSocket, isWidgetInput, typesCompatible } from "./state.js";
import { buildWidget, rowVisible } from "./widgets.js";

const SVG_NS = "http://www.w3.org/2000/svg";

export class Canvas {
  constructor(inner, doc, { onSelect, onNote } = {}) {
    this.inner = inner;   // #canvas-inner -- nodes + <svg> live here
    this.viewport = inner.parentElement; // #graph-canvas -- the fixed window
    this.svg = inner.querySelector("#edge-layer");
    this.doc = doc;
    this.onSelect = onSelect || (() => {});
    this.onNote = onNote || (() => {});
    this.selection = null;          // {kind, id | edge}
    this.pan = { x: 0, y: 0 };      // translate of the plane (px)
    this._sockets = new Map();      // nodeId -> Map("dir:port" -> {x, y})
    this._badges = new Map();       // nodeId -> {kind, text} execution status
    this._connect = null;           // active wire drag
    this._drag = null;              // active node drag
    this._panDrag = null;           // active background pan
    this._suppressClick = false;    // set after a wire drop, eats the click
    this._diagTimers = new Map();   // nodeId -> debounce timer (live diagnostics)
    this._diagSeq = new Map();      // nodeId -> latest request token
    this._diagLast = new Map();     // nodeId -> {sig, messages} last result
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
    this._scheduleDiagnostics();
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
    const colOut = document.createElement("div");
    colOut.className = "gnode-col out";
    body.append(colIn, colOut);

    if (!cls) {
      const ph = document.createElement("div");
      ph.className = "gnode-unknown";
      ph.textContent = "(unknown class)";
      colIn.appendChild(ph);
    } else {
      for (const p of cls.inputs) colIn.appendChild(this._inputRow(node, cls, p));
      for (const p of cls.outputs) colOut.appendChild(this._outRow(p));
    }
    el.appendChild(body);
    return el;
  }

  /**
   * Input row: socket + name on one line, then the editable widget
   * below it (or the `<- node.port` wire hint when an edge feeds the
   * input -- edges override params by design, docs 05 §4). Pure handles
   * are just the line. `visible_when` hides the row (the value stays in
   * params while hidden). Socket states: filled = wired, red ring =
   * required and unconnected.
   */
  _inputRow(node, cls, p) {
    const row = document.createElement("div");
    row.className = "gport-row in";
    row.dataset.port = p.name;
    row.dataset.dir = "in";
    const wired = this.doc.edgeInto(node.id, p.name);

    const line = document.createElement("div");
    line.className = "gport-line";
    if (hasSocket(p)) {
      const s = this._socket(p, "in");
      if (wired) s.classList.add("connected");
      else if (p.required) s.classList.add("req-unmet");
      line.appendChild(s);
    }
    const label = document.createElement("span");
    label.className = "gport-label";
    label.textContent = p.name + (p.required ? " *" : "");
    label.title = p.doc ? `${p.name} (${p.type}) -- ${p.doc}` : `${p.name} (${p.type})`;
    line.appendChild(label);
    if (wired && isWidgetInput(p)) {
      const hint = document.createElement("span");
      hint.className = "gwire";
      hint.textContent = `<- ${wired.from_node}.${wired.from_port}`;
      hint.title = `value overridden by ${wired.from_node}.${wired.from_port}`;
      line.appendChild(hint);
    }
    row.appendChild(line);

    if (isWidgetInput(p)) {
      if (!rowVisible(node, p, cls.inputs)) row.classList.add("gated");
      else if (!wired) {
        row.appendChild(
          buildWidget({ doc: this.doc, node, port: p, origin: "canvas", onNote: this.onNote }),
        );
      }
    }
    return row;
  }

  _outRow(p) {
    const row = document.createElement("div");
    row.className = "gport-row out";
    row.dataset.port = p.name;
    row.dataset.dir = "out";
    const line = document.createElement("div");
    line.className = "gport-line";
    const label = document.createElement("span");
    label.className = "gport-label";
    label.textContent = p.name;
    label.title = p.doc ? `${p.name} (${p.type}) -- ${p.doc}` : `${p.name} (${p.type})`;
    line.append(label, this._socket(p, "out"));
    row.appendChild(line);
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
    const node = this.doc.nodes.get(id);
    const el = [...this.inner.querySelectorAll(".gnode")].find((n) => n.dataset.id === id);
    if (node && el) {
      // glide, not scrollIntoView: the viewport itself never scrolls
      this.centerOn(node.x + el.offsetWidth / 2, node.y + el.offsetHeight / 2, true);
    }
    this.selectNode(id);
  }

  /* ================= panning (infinite plane) ================= */

  /** Plane coordinates of the viewport center (where drops land). */
  viewportCenter() {
    const r = this.viewport.getBoundingClientRect();
    const ir = this.inner.getBoundingClientRect();
    return { x: r.left + r.width / 2 - ir.left, y: r.top + r.height / 2 - ir.top };
  }

  /** Put a plane point at the viewport center; glide animates the move. */
  centerOn(x, y, glide = false) {
    const r = this.viewport.getBoundingClientRect();
    this._setPan(r.width / 2 - x, r.height / 2 - y, glide);
  }

  /** Frame every node (or the origin when the canvas is empty). */
  frameAll(glide = false) {
    const els = [...this.inner.querySelectorAll(".gnode")];
    if (!els.length) { this.centerOn(0, 0, glide); return; }
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const el of els) {
      const l = parseFloat(el.style.left) || 0;
      const t = parseFloat(el.style.top) || 0;
      x0 = Math.min(x0, l); y0 = Math.min(y0, t);
      x1 = Math.max(x1, l + el.offsetWidth);
      y1 = Math.max(y1, t + el.offsetHeight);
    }
    this.centerOn((x0 + x1) / 2, (y0 + y1) / 2, glide);
  }

  panBy(dx, dy) {
    this._endGlide();
    this._setPan(this.pan.x + dx, this.pan.y + dy);
  }

  _setPan(x, y, glide = false) {
    this.pan.x = x;
    this.pan.y = y;
    if (glide) {
      this.viewport.classList.add("glide");
      clearTimeout(this._glideTimer);
      this._glideTimer = setTimeout(() => this._endGlide(), 320);
    }
    this.inner.style.transform = `translate(${x}px, ${y}px)`;
    // the dot grid rides the plane, so panning reads as movement
    this.viewport.style.backgroundPosition = `${x}px ${y}px`;
  }

  _endGlide() {
    clearTimeout(this._glideTimer);
    this.viewport.classList.remove("glide");
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

  /* ================= live diagnostics (server -> node) =================
     Classes that declare diagnostics (has_diagnostics) get a debounced
     POST after any re-render: the response's per-input lines render
     under that input's row on the node -- the legacy "node calculates
     and shows content" behaviour. Empty results clear the boxes (a box
     only ever shows what the server sent); failures are silent here
     (legacy parity) and the inspector's "Run diagnostics" button
     reports errors to the console. */

  _scheduleDiagnostics() {
    for (const node of this.doc.nodes.values()) {
      const cls = this.doc.classOf(node);
      if (!cls || !cls.has_diagnostics) continue;
      const sig = JSON.stringify(this.doc.paramsFor(node));
      const last = this._diagLast.get(node.id);
      if (last && last.sig === sig) {
        this._renderDiagnostics(node, last.messages); // rebuild restores from cache
        continue;
      }
      clearTimeout(this._diagTimers.get(node.id));
      this._diagTimers.set(node.id, setTimeout(() => this._runDiagnostics(node.id), 400));
    }
  }

  async _runDiagnostics(id) {
    const node = this.doc.nodes.get(id);
    if (!node) return;
    const cls = this.doc.classOf(node);
    if (!cls || !cls.has_diagnostics) return;
    const sig = JSON.stringify(this.doc.paramsFor(node));
    if (this._diagLast.get(id) && this._diagLast.get(id).sig === sig) return;
    const seq = (this._diagSeq.get(id) || 0) + 1;
    this._diagSeq.set(id, seq);
    let messages;
    try {
      const res = await api(`/graphs/nodes/${encodeURIComponent(cls.class_name)}/diagnostics`, {
        method: "POST",
        body: { params: this.doc.paramsFor(node) },
      });
      messages = res.messages || {};
    } catch {
      return; // transient failure: the next change schedules again
    }
    if (this._diagSeq.get(id) !== seq) return;      // superseded by a newer request
    const fresh = this.doc.nodes.get(id);
    if (!fresh) return;
    if (JSON.stringify(this.doc.paramsFor(fresh)) !== sig) return; // params moved on
    this._diagLast.set(id, { sig, messages });
    this._renderDiagnostics(fresh, messages);
  }

  _renderDiagnostics(node, messages) {
    const el = this.inner.querySelector(`.gnode[data-id="${CSS.escape(node.id)}"]`);
    if (!el) return;
    const before = el.querySelectorAll(".gdiag").length;
    for (const row of el.querySelectorAll('.gport-row[data-dir="in"]')) {
      const old = row.querySelector(":scope > .gdiag");
      if (old) old.remove();
    }
    for (const [input, lines] of Object.entries(messages)) {
      if (!lines || !lines.length) continue;
      const row = el.querySelector(
        `.gport-row[data-dir="in"][data-port="${CSS.escape(input)}"]`,
      );
      if (!row) continue;
      const box = document.createElement("div");
      box.className = "gdiag";
      for (const text of lines) {
        const line = document.createElement("div");
        line.className = "gdiag-line";
        line.textContent = text;
        box.appendChild(line);
      }
      row.appendChild(box);
    }
    if (el.querySelectorAll(".gdiag").length !== before) {
      this._measure(el, node); // rows shifted: sockets move with them
      this._updateEdgesFor(node.id);
    }
  }

  /* ================= interaction ================= */

  _bind() {
    // bound on the VIEWPORT, not #canvas-inner: once the plane is panned,
    // strips of the viewport are no longer covered by inner's box and must
    // still pan / deselect / recenter like empty plane
    this.viewport.addEventListener("pointerdown", (e) => this._pointerDown(e));
    this.viewport.addEventListener("click", (e) => this._click(e));
    this.viewport.addEventListener("dblclick", (e) => this._doubleClick(e));
    document.addEventListener("pointermove", (e) => this._pointerMove(e));
    document.addEventListener("pointerup", (e) => this._pointerUp(e));
    document.addEventListener("keydown", (e) => this._keyDown(e));
    // wheel = pan the plane (passive:false: the viewport must not scroll)
    this.viewport.addEventListener("wheel", (e) => {
      e.preventDefault();
      this._endGlide();
      if (e.shiftKey && e.deltaX === 0) this.panBy(-e.deltaY, 0);
      else this.panBy(-e.deltaX, -e.deltaY);
    }, { passive: false });
  }

  _pointInInner(e) {
    const rect = this.inner.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  _pointerDown(e) {
    // Any new press supersedes a pending suppress flag (e.g. a wire
    // released off-canvas never produces the trailing canvas click).
    this._suppressClick = false;
    this._endGlide();
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
      return;
    }
    // empty plane: left-button drag pans the view
    if (e.button !== 0) return;
    e.preventDefault();
    this._panDrag = {
      sx: e.clientX, sy: e.clientY,
      ox: this.pan.x, oy: this.pan.y, moved: false,
    };
    this.viewport.classList.add("panning");
  }

  _pointerMove(e) {
    if (this._drag) {
      const p = this._pointInInner(e);
      const node = this._drag.node;
      // infinite plane: negative coordinates are allowed
      node.x = Math.round(this._drag.origX + (p.x - this._drag.startX));
      node.y = Math.round(this._drag.origY + (p.y - this._drag.startY));
      this._drag.el.style.left = node.x + "px";
      this._drag.el.style.top = node.y + "px";
      this._updateEdgesFor(node.id);
      return;
    }
    if (this._connect) {
      const p = this._pointInInner(e);
      this._connect.temp.setAttribute("d", this._curve(this._connect.origin, p));
      return;
    }
    if (this._panDrag) {
      const dx = e.clientX - this._panDrag.sx;
      const dy = e.clientY - this._panDrag.sy;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) this._panDrag.moved = true;
      this._setPan(this._panDrag.ox + dx, this._panDrag.oy + dy);
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
      return;
    }
    if (this._panDrag) {
      // a real pan eats the click that follows (no accidental deselect)
      if (this._panDrag.moved) this._suppressClick = true;
      this._panDrag = null;
      this.viewport.classList.remove("panning");
    }
  }

  /** Double-click on the empty plane glides back to the origin circle. */
  _doubleClick(e) {
    if (e.target.closest(".gnode") || e.target.closest("path.edge")) return;
    this.centerOn(0, 0, true);
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
