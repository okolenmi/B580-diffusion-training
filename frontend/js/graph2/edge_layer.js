/* ---------------------------------------------------------------------------
   graph2/edge_layer.js -- the wires, one SVG path per edge, kept by key so a
   moved node updates only the paths that end at it.

   Data wires leave an output on the right and enter an input on the left;
   event wires are dashed and run from a node's bottom strip to another's top
   strip. A wire whose port is gone (dangling) is drawn dashed red to the
   node's edge instead of disappearing. Several wires into one `multi` port
   leave the socket slightly apart.
   --------------------------------------------------------------------------- */

import { endpointHost } from "./doc.js";
import { eventPath, fanOffset, wirePath } from "./geometry.js";

const SVG = "http://www.w3.org/2000/svg";
const key = (kind, e) => `${kind}|${e.from_node}|${e.from_port}|${e.to_node}|${e.to_port}`;

export class EdgeLayer {
  /**
   * @param {SVGElement} svg
   * @param {{doc, views: Map, innerRect: ()=>DOMRect, zoom: ()=>number, onPick: (edge, kind, event)=>void}} deps
   */
  constructor(svg, { doc, views, innerRect, zoom, onPick }) {
    this.svg = svg;
    this.doc = doc;
    this.views = views;
    this.innerRect = innerRect;
    this.zoom = zoom;
    this.onPick = onPick;
    this.paths = new Map(); // key -> {group, hit, line, edge, kind}
    this.selected = null;
    this.svg.addEventListener("pointerdown", (e) => {
      const hit = e.target.closest?.("[data-key]");
      if (!hit) return;
      const entry = this.paths.get(hit.dataset.key);
      if (entry) this.onPick(entry.edge, entry.kind, e);
    });
  }

  select(edgeKey) {
    this.selected = edgeKey;
    for (const [k, entry] of this.paths) entry.group.classList.toggle("selected", k === edgeKey);
  }

  keyOf(kind, edge) {
    return key(kind, edge);
  }

  center(element) {
    const rect = element.getBoundingClientRect();
    const origin = this.innerRect();
    const z = this.zoom();
    return { x: (rect.left + rect.width / 2 - origin.left) / z, y: (rect.top + rect.height / 2 - origin.top) / z };
  }

  /** The point a wire attaches to: the port's socket, else (dangling, or an addon end) the node's own edge. */
  _anchor(uid, side, port) {
    const view = this.views.get(endpointHost(uid));
    if (!view) return null;
    const socket = port && view.socket(side, port);
    if (socket) return { ...this.center(socket), loose: false };
    const rect = view.el.getBoundingClientRect();
    const origin = this.innerRect();
    const z = this.zoom();
    const left = (rect.left - origin.left) / z, top = (rect.top - origin.top) / z, w = rect.width / z, h = rect.height / z;
    const point = { in: { x: left, y: top + h / 2 }, out: { x: left + w, y: top + h / 2 }, "event-in": { x: left + w / 2, y: top }, "event-out": { x: left + w / 2, y: top + h } }[side];
    return { ...point, loose: true };
  }

  _shape(kind, edge) {
    const data = kind === "data";
    const a = this._anchor(edge.from_node, data ? "out" : "event-out", edge.from_port);
    const b = this._anchor(edge.to_node, data ? "in" : "event-in", edge.to_port);
    if (!a || !b) return null;
    let by = b.y;
    if (data) {
      const into = this.doc.edgesInto(edge.to_node, edge.to_port);
      if (into.length > 1) by += fanOffset(into.findIndex((e) => e.from_node === edge.from_node && e.from_port === edge.from_port), into.length);
    }
    // An event wire may end at an addon (`uid.id`), which has no socket: that is not a broken wire.
    const addonEnd = (end, uid) => end.loose && endpointHost(uid) !== uid;
    const broken = data ? a.loose || b.loose : (a.loose && !addonEnd(a, edge.from_node)) || (b.loose && !addonEnd(b, edge.to_node));
    return { d: (data ? wirePath : eventPath)(a.x, a.y, b.x, by), broken };
  }

  _make(kind, edge) {
    const group = document.createElementNS(SVG, "g");
    const hit = document.createElementNS(SVG, "path");
    const line = document.createElementNS(SVG, "path");
    hit.setAttribute("class", "g2-edge-hit");
    line.setAttribute("class", `g2-edge ${kind}`);
    for (const p of [hit, line]) p.dataset.key = key(kind, edge);
    group.append(hit, line);
    this.svg.append(group);
    return { group, hit, line, edge, kind, fresh: true };
  }

  /** Redraws every edge whose end is in `uids` (all when omitted); adds and removes paths to match the document. */
  update(uids) {
    const wanted = new Set();
    for (const [kind, edges] of [["data", this.doc.edges], ["event", this.doc.eventEdges]]) {
      for (const edge of edges) {
        const k = key(kind, edge);
        wanted.add(k);
        let entry = this.paths.get(k);
        if (!entry) { entry = this._make(kind, edge); this.paths.set(k, entry); }
        if (uids && !entry.fresh && !uids.has(endpointHost(edge.from_node)) && !uids.has(endpointHost(edge.to_node))) continue;
        entry.fresh = false;
        const shape = this._shape(kind, edge);
        if (!shape) { entry.group.style.display = "none"; continue; }
        entry.group.style.display = "";
        entry.hit.setAttribute("d", shape.d);
        entry.line.setAttribute("d", shape.d);
        entry.line.classList.toggle("dangling", shape.broken);
      }
    }
    for (const [k, entry] of [...this.paths]) {
      if (!wanted.has(k)) { entry.group.remove(); this.paths.delete(k); }
    }
    if (this.selected && !this.paths.has(this.selected)) this.selected = null;
  }

  /** The rubber-band wire of a drag in progress. */
  drag(d, kind = "data") {
    if (!this._ghost) {
      this._ghost = document.createElementNS(SVG, "path");
      this.svg.append(this._ghost);
    }
    this._ghost.setAttribute("class", `g2-edge ghost ${kind}`);
    this._ghost.setAttribute("d", d);
  }

  endDrag() {
    this._ghost?.remove();
    this._ghost = null;
  }
}
