/* ---------------------------------------------------------------------------
   graph2/canvas.js -- the editing surface: pan and zoom, nodes (one NodeView
   each), wires (EdgeLayer), selection, drag a node, drag a wire.

   Every document event patches only what it names:
     node:add/remove   one view
     node:move         one element's position + the wires that end at it
     node:params/title/ui, shape change    one view (+ its wires)
     edge:*            the views at both ends (their sockets show "wired") + that wire
   so a 200-node graph pans and drags without rebuilding anything.

   Wiring: drag from an output (or an event output) to a socket; drag from an
   input to pick its wire up; drop on empty canvas asks `hooks.onDropEmpty`
   (the quick-add list of nodes with a compatible port). While a wire is
   dragged, a folded group under the pointer opens after a short hover and
   closes again when the drag ends; compatible sockets are lit, the others dimmed.
   --------------------------------------------------------------------------- */

import { endpointHost } from "./doc.js";
import { EdgeLayer } from "./edge_layer.js";
import { clampZoom, eventPath, toCanvas, wirePath, zoomAround } from "./geometry.js";
import { NodeView } from "./node_view.js";

const SPRING_MS = 400;
const SVG_NS = "http://www.w3.org/2000/svg";
const TYPING = /^(INPUT|TEXTAREA|SELECT)$/;

export class Canvas {
  /**
   * @param {{root: HTMLElement, doc, shapes, options: Function, note: Function, hooks?: {onSelect?, onDropEmpty?, onQuickAdd?}}} deps
   */
  constructor({ root, doc, shapes, options, note, hooks = {} }) {
    this.root = root;
    this.doc = doc;
    this.shapes = shapes;
    this.options = options;
    this.note = note;
    this.hooks = hooks;
    this.views = new Map();
    this.selection = new Set();
    this.pan = { x: 0, y: 0 };
    this.zoom = 1;
    this.clipboard = null;
    this._pending = new Set();
    this._drag = null;
    this._edge = null;

    this.inner = document.createElement("div");
    this.inner.className = "g2-inner";
    this.svg = document.createElementNS(SVG_NS, "svg");
    this.svg.setAttribute("class", "g2-edges");
    this.layer = document.createElement("div");
    this.layer.className = "g2-nodes";
    this.inner.append(this.svg, this.layer);
    this.root.replaceChildren(this.inner);
    this.root.tabIndex = 0;

    this.edges = new EdgeLayer(this.svg, {
      doc, views: this.views, innerRect: () => this.inner.getBoundingClientRect(), zoom: () => this.zoom,
      onPick: (edge, kind, event) => this._pickEdge(edge, kind, event),
    });
    this.observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver((entries) => {
      for (const entry of entries) this._schedule(entry.target.dataset.uid);
    });
    this._bindDocument();
    this._bindPointer();
    this._bindKeys();
    this._apply();
    this.rebuild();
  }

  /* ----- views follow the document ----- */

  _bindDocument() {
    const { doc, shapes } = this;
    doc.on("reset", () => this.rebuild());
    doc.on("node:add", (e) => { this._addView(doc.nodes.get(e.uid)); shapes.resolveNow(doc.nodes.get(e.uid)); });
    doc.on("node:remove", (e) => { this._dropView(e.uid); shapes.forget(e.uid); this.edges.update(); });
    doc.on("node:move", (e) => { this.views.get(e.uid)?.place(); this._schedule(e.uid); });
    doc.on("node:title", (e) => this.views.get(e.uid)?.render());
    doc.on("node:ui", (e) => this._refresh(e.uid));
    doc.on("node:addons", (e) => this._refresh(e.uid));
    doc.on("node:params", (e) => {
      const node = doc.nodes.get(e.uid);
      if (node && shapes.dependsOn(node.class_name, e.name)) shapes.schedule(node);
      this.views.get(e.uid)?.paramChanged(e.name, e.origin);
      this._schedule(e.uid);
    });
    for (const type of ["edge:add", "edge:remove"]) {
      doc.on(type, (e) => {
        const ends = new Set([endpointHost(e.edge.from_node), endpointHost(e.edge.to_node)]);
        for (const uid of ends) this.views.get(uid)?.render();
        this.edges.update(ends);
      });
    }
    doc.on("edge:order", (e) => this._refresh(e.uid));
    for (const type of ["event_edge:add", "event_edge:remove"]) doc.on(type, () => this.edges.update());
    shapes.onShape((e) => this._refresh(e.uid));
  }

  _refresh(uid) {
    this.views.get(uid)?.render();
    this._schedule(uid);
  }

  /** Wires are redrawn once per frame for every node that changed. */
  _schedule(uid) {
    if (!uid) return;
    this._pending.add(uid);
    if (this._frame) return;
    const run = () => {
      this._frame = null;
      const uids = this._pending;
      this._pending = new Set();
      this.edges.update(uids);
    };
    this._frame = typeof requestAnimationFrame === "function" ? requestAnimationFrame(run) : setTimeout(run, 0);
  }

  rebuild() {
    for (const uid of [...this.views.keys()]) this._dropView(uid);
    for (const node of this.doc.nodes.values()) this._addView(node);
    this.edges.update();
    for (const node of this.doc.nodes.values()) this.shapes.resolveNow(node);
  }

  _addView(node) {
    if (!node) return;
    const view = new NodeView({
      doc: this.doc, shapes: this.shapes, node, options: this.options, note: this.note,
      onTitle: (uid, element) => this._editTitle(uid, element),
    });
    this.views.set(node.uid, view);
    this.layer.append(view.el);
    this.observer?.observe(view.el);
    this._schedule(node.uid);
  }

  _dropView(uid) {
    const view = this.views.get(uid);
    if (!view) return;
    this.observer?.unobserve(view.el);
    view.el.remove();
    this.views.delete(uid);
    this.selection.delete(uid);
  }

  _editTitle(uid, element) {
    const node = this.doc.nodes.get(uid);
    const input = document.createElement("input");
    input.className = "g2-title-edit";
    input.value = node.title || element.textContent;
    element.replaceWith(input);
    input.focus();
    input.select();
    let finished = false;
    const done = (keep) => {
      if (finished) return;
      finished = true;
      if (keep && input.value.trim() !== (node.title || element.textContent)) this.doc.setTitle(uid, input.value.trim());
      else this.views.get(uid)?.render();
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") done(true); if (e.key === "Escape") done(false); });
    input.addEventListener("blur", () => done(true), { once: true });
  }

  /* ----- selection ----- */

  select(uids, { add = false } = {}) {
    if (!add) this.selection.clear();
    for (const uid of uids) this.selection.add(uid);
    for (const [uid, view] of this.views) view.setSelected(this.selection.has(uid));
    this._edge = null;
    this.edges.select(null);
    this.hooks.onSelect?.([...this.selection]);
  }

  _pickEdge(edge, kind, event) {
    event.stopPropagation();
    this.select([]);
    this._edge = { edge, kind };
    this.edges.select(this.edges.keyOf(kind, edge));
    this.root.focus();
  }

  deleteSelection() {
    if (this._edge) {
      const { edge, kind } = this._edge;
      this._edge = null;
      if (kind === "data") this.doc.disconnect(edge);
      else this.doc.disconnectEvent(edge);
      return;
    }
    for (const uid of [...this.selection]) this.doc.removeNode(uid);
    this.select([]);
  }

  /** Selects a node and brings it to the middle of the view (an issue chip names it). */
  focusNode(uid) {
    const node = this.doc.nodes.get(uid);
    if (!node) return;
    this.select([uid]);
    const box = this.root.getBoundingClientRect();
    this.pan = { x: box.width / 2 - (node.layout.x + 140) * this.zoom, y: box.height / 3 - node.layout.y * this.zoom };
    this._apply();
    this.edges.update();
  }

  /* ----- coordinates, pan, zoom ----- */

  _apply() {
    this.inner.style.transform = `translate(${this.pan.x}px, ${this.pan.y}px) scale(${this.zoom})`;
  }

  point(event) {
    return toCanvas(event.clientX, event.clientY, this.root.getBoundingClientRect(), this.pan, this.zoom);
  }

  /** The canvas point at the middle of the view: where a node added from the palette goes. */
  viewCenter() {
    const box = this.root.getBoundingClientRect();
    return toCanvas(box.left + box.width / 2, box.top + box.height / 2, box, this.pan, this.zoom);
  }

  /** Fits the whole graph in the view (the origin when empty). */
  fit() {
    const nodes = [...this.doc.nodes.values()];
    if (!nodes.length) { this.pan = { x: 40, y: 40 }; this.zoom = 1; this._apply(); return; }
    const xs = nodes.map((n) => n.layout.x), ys = nodes.map((n) => n.layout.y);
    const right = Math.max(...nodes.map((n) => n.layout.x + (n.layout.w || 280))), bottom = Math.max(...ys) + 260;
    const box = this.root.getBoundingClientRect();
    const zoom = clampZoom(Math.min(box.width / (right - Math.min(...xs) + 80), box.height / (bottom - Math.min(...ys) + 80), 1));
    this.zoom = zoom;
    this.pan = { x: 40 - Math.min(...xs) * zoom, y: 40 - Math.min(...ys) * zoom };
    this._apply();
    this.edges.update();
  }

  /* ----- pointer ----- */

  _bindPointer() {
    this.root.addEventListener("pointerdown", (e) => this._down(e));
    this.root.addEventListener("pointermove", (e) => this._move(e));
    this.root.addEventListener("pointerup", (e) => this._up(e));
    this.root.addEventListener("pointercancel", (e) => this._up(e));
    this.root.addEventListener("click", (e) => this._click(e));
    this.root.addEventListener("dblclick", (e) => {
      if (e.target === this.root || e.target === this.inner || e.target === this.layer) this.hooks.onQuickAdd?.(this.point(e), e);
    });
    this.root.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = this.root.getBoundingClientRect();
      const cursor = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      const next = clampZoom(this.zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1));
      this.pan = zoomAround(cursor, this.pan, this.zoom, next);
      this.zoom = next;
      this._apply();
      this.edges.update();
    }, { passive: false });
  }

  _uidOf(element) {
    return element.closest(".g2-node")?.dataset.uid;
  }

  _down(e) {
    if (e.button !== 0 && e.button !== 1) return;
    const socket = e.target.closest(".g2-socket");
    if (socket && e.button === 0) { this._startWire(socket, e); return; }
    if (e.target.closest("input, textarea, select, button, label, [data-action]")) return;
    const uid = this._uidOf(e.target);
    if (uid) {
      if (e.shiftKey) this.select(this.selection.has(uid) ? [...this.selection].filter((u) => u !== uid) : [...this.selection, uid]);
      else if (!this.selection.has(uid)) this.select([uid]);
      if (!e.target.closest(".g2-head") || !this.selection.has(uid)) return;
      this._drag = { kind: "node", start: this.point(e), origin: new Map([...this.selection].map((u) => [u, { ...this.doc.nodes.get(u).layout }])) };
    } else if (e.shiftKey && e.button === 0) {
      this._drag = { kind: "band", from: this.point(e), client: { x: e.clientX, y: e.clientY } };
      this._band = this._band || Object.assign(document.createElement("div"), { className: "g2-band" });
      this.root.append(this._band);
    } else {
      this.select([]);
      this._drag = { kind: "pan", start: { x: e.clientX, y: e.clientY }, pan: { ...this.pan } };
    }
    this.root.setPointerCapture?.(e.pointerId);
    e.preventDefault();
  }

  _move(e) {
    const drag = this._drag;
    if (!drag) return;
    if (drag.kind === "pan") {
      this.pan = { x: drag.pan.x + e.clientX - drag.start.x, y: drag.pan.y + e.clientY - drag.start.y };
      this._apply();
    } else if (drag.kind === "node") {
      const p = this.point(e);
      for (const [uid, from] of drag.origin) this.doc.moveNode(uid, Math.round(from.x + p.x - drag.start.x), Math.round(from.y + p.y - drag.start.y));
    } else if (drag.kind === "wire") {
      this._moveWire(e);
    } else if (drag.kind === "band") {
      const box = this.root.getBoundingClientRect();
      const x = Math.min(drag.client.x, e.clientX) - box.left, y = Math.min(drag.client.y, e.clientY) - box.top;
      Object.assign(this._band.style, { left: `${x}px`, top: `${y}px`, width: `${Math.abs(e.clientX - drag.client.x)}px`, height: `${Math.abs(e.clientY - drag.client.y)}px` });
      drag.now = this.point(e);
    }
  }

  _up(e) {
    const drag = this._drag;
    this._drag = null;
    if (!drag) return;
    this.root.releasePointerCapture?.(e.pointerId);
    if (drag.kind === "wire") this._endWire(e, drag);
    if (drag.kind === "band") {
      this._band.remove();
      const a = drag.from, b = drag.now || a;
      const [x0, x1] = [Math.min(a.x, b.x), Math.max(a.x, b.x)], [y0, y1] = [Math.min(a.y, b.y), Math.max(a.y, b.y)];
      const inside = [...this.doc.nodes.values()].filter((n) => n.layout.x < x1 && n.layout.x + (n.layout.w || 280) > x0 && n.layout.y < y1 && n.layout.y + 60 > y0).map((n) => n.uid);
      this.select([...this.selection, ...inside]);
    }
  }

  _click(e) {
    const action = e.target.closest("[data-action]");
    const uid = this._uidOf(e.target);
    if (!action || !uid) return;
    const node = this.doc.nodes.get(uid);
    if (action.dataset.action === "fold") this.doc.setLayout(uid, { collapsed: !node.layout.collapsed });
    else if (action.dataset.action === "advanced") this.doc.setLayout(uid, { advanced: !node.layout.advanced });
    else if (action.dataset.action === "group") {
      const id = action.dataset.group;
      const group = (this.shapes.info(node.class_name)?.groups || []).find((g) => g.id === id);
      const open = node.layout.groups && id in node.layout.groups ? node.layout.groups[id] : !(group && group.collapsed);
      this.doc.setGroupOpen(uid, id, !open);
    }
  }

  /* ----- wiring ----- */

  _startWire(socket, e) {
    const uid = this._uidOf(socket);
    const side = socket.dataset.side;
    const port = socket.dataset.port;
    const events = side.startsWith("event");
    let from = { uid, side, port };
    let reverse = side === "in" || side === "event-in";
    let picked = null;
    if (reverse) {
      const wires = events ? this.doc.eventEdges.filter((w) => w.to_node === uid && w.to_port === port) : this.doc.edgesInto(uid, port);
      const wire = wires[wires.length - 1];
      if (wire) {
        picked = wire;
        from = { uid: wire.from_node, side: events ? "event-out" : "out", port: wire.from_port };
        reverse = false;
        if (events) this.doc.disconnectEvent(wire); else this.doc.disconnect(wire);
      }
    }
    this._drag = { kind: "wire", from, reverse, picked, events, spring: [], hover: null, timer: null };
    this._lightSockets();
    this.root.setPointerCapture?.(e.pointerId);
    e.preventDefault();
    this._moveWire(e);
  }

  /** Marks every socket of the other kind as compatible (`ok`) or not (`bad`) for the wire being dragged. */
  _lightSockets() {
    const { from, reverse, events } = this._drag;
    const want = events ? (reverse ? "event-out" : "event-in") : reverse ? "out" : "in";
    for (const el of this.root.querySelectorAll(".g2-socket")) {
      el.classList.remove("ok", "bad");
      if (el.dataset.side !== want) continue;
      const uid = this._uidOf(el);
      const [a, b] = reverse ? [{ uid, port: el.dataset.port }, from] : [from, { uid, port: el.dataset.port }];
      el.classList.add(!events && this.doc.connectProblem(a.uid, a.port, b.uid, b.port) ? "bad" : "ok");
    }
  }

  _moveWire(e) {
    const drag = this._drag;
    const here = this.point(e);
    const socket = this.views.get(drag.from.uid)?.socket(drag.from.side, drag.from.port);
    if (socket) {
      const anchor = this.edges.center(socket);
      const [a, b] = drag.reverse ? [here, anchor] : [anchor, here];
      this.edges.drag((drag.events ? eventPath : wirePath)(a.x, a.y, b.x, b.y), drag.events ? "event" : "data");
    }
    const head = document.elementFromPoint(e.clientX, e.clientY)?.closest?.(".g2-group-head");
    const key = head ? `${this._uidOf(head)}|${head.dataset.group}` : null;
    if (key === drag.hover) return;
    clearTimeout(drag.timer);
    drag.hover = key;
    if (!head) return;
    const uid = this._uidOf(head), group = head.dataset.group;
    drag.timer = setTimeout(() => {
      if (head.querySelector(".g2-chev")?.textContent !== "▸") return;
      const groups = this.doc.nodes.get(uid)?.layout.groups;
      drag.spring.push({ uid, group, was: groups && group in groups ? groups[group] : undefined });
      this.doc.setGroupOpen(uid, group, true);
      this._lightSockets();
    }, SPRING_MS);
  }

  _endWire(e, drag) {
    clearTimeout(drag.timer);
    this.edges.endDrag();
    const under = document.elementFromPoint(e.clientX, e.clientY);
    const target = under?.closest?.(".g2-socket");
    const want = drag.events ? (drag.reverse ? "event-out" : "event-in") : drag.reverse ? "out" : "in";
    if (target && target.dataset.side === want) {
      const there = { uid: this._uidOf(target), port: target.dataset.port };
      const [a, b] = drag.reverse ? [there, drag.from] : [drag.from, there];
      const result = drag.events ? this.doc.connectEvent(a.uid, a.port, b.uid, b.port) : this.doc.connect(a.uid, a.port, b.uid, b.port);
      if (!result.ok) this.note?.(result.reason, "warn");
    } else if (!target && !under?.closest?.(".g2-node") && !drag.picked) {
      this.hooks.onDropEmpty?.({ from: drag.from, reverse: drag.reverse, events: drag.events, point: this.point(e), client: { x: e.clientX, y: e.clientY } });
    }
    for (const { uid, group, was } of drag.spring.reverse()) {
      const node = this.doc.nodes.get(uid);
      if (!node) continue;
      if (was === undefined) {
        const groups = { ...(node.layout.groups || {}) };
        delete groups[group];
        this.doc.setLayout(uid, { groups });
      } else this.doc.setGroupOpen(uid, group, was);
    }
    for (const el of this.root.querySelectorAll(".g2-socket.ok, .g2-socket.bad")) el.classList.remove("ok", "bad");
  }

  /* ----- keyboard ----- */

  _bindKeys() {
    this.root.addEventListener("keydown", (e) => {
      if (TYPING.test(e.target.tagName)) return;
      const mod = e.ctrlKey || e.metaKey;
      if (e.key === "Delete" || e.key === "Backspace") { this.deleteSelection(); e.preventDefault(); }
      else if (mod && e.key === "a") { this.select([...this.doc.nodes.keys()]); e.preventDefault(); }
      else if (mod && e.key === "c") this.clipboard = this.doc.copy([...this.selection]);
      else if (mod && e.key === "v" && this.clipboard) this.select(this.doc.paste(this.clipboard));
      else if (e.key === "Escape") this.select([]);
    });
  }
}
