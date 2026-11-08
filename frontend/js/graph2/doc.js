/* ---------------------------------------------------------------------------
   graph2/doc.js -- GraphDoc: the editable graph document, format 3
   (rework/contracts/graph-json.md).

   One source of truth. Every mutation goes through a method, changes the
   data, and emits a fine-grained event so views patch only what changed:

     node:add  node:remove  node:move  node:title  node:params  node:ui
     node:addons  edge:add  edge:remove  event_edge:add  event_edge:remove
     settings  graph:addons  views

   plus `change` after each of them (dirty flag, draft saving). An event's
   argument is `{type, uid?, ...}`; handlers must not mutate the document.

   Wire rules (the server validates the same ones at run time):
     - an ordinary input takes one wire, a new one replaces it; a `multi`
       input takes several, in list order (that order is the argument order);
     - no self loops and no cycles in the data plane;
     - a wire whose port vanished after a resolve stays in the document
       ("dangling") and is reported, never silently deleted.

   The document knows nothing about the catalog: a `shapeOf(node)` function
   given at construction returns a node's resolved ports
   (`{inputs, outputs, events_in, events_out}`) or null for an unknown class.
   --------------------------------------------------------------------------- */

import { eventWireProblem, wireProblem } from "./ports.js";

export const GRAPH_FORMAT = 3;
const ALPHABET = "abcdefghijklmnopqrstuvwxyz234567";
const KNOWN_KEYS = new Set(["format", "settings", "nodes", "edges", "event_edges", "views", "addons"]);

/** `n_` plus six base32 characters, not in `taken`. */
export function newUid(taken, random = Math.random) {
  for (;;) {
    let uid = "n_";
    for (let i = 0; i < 6; i++) uid += ALPHABET[Math.floor(random() * ALPHABET.length) % ALPHABET.length];
    if (!taken.has(uid)) return uid;
  }
}

/** The host of an event wire's end: `n_ab12cd.a1` is the addon `a1` of node `n_ab12cd`, `graph.g1` an addon of the graph. */
export function endpointHost(endpoint) {
  return endpoint.split(".", 1)[0];
}

const clone = (value) => (value === undefined ? undefined : JSON.parse(JSON.stringify(value)));
const sameEdge = (a, b) => a.from_node === b.from_node && a.from_port === b.from_port && a.to_node === b.to_node && a.to_port === b.to_port;

export class GraphDoc {
  /** @param {{shapeOf?: (node:object)=>object|null, random?: ()=>number}} [options] */
  constructor({ shapeOf = () => null, random = Math.random } = {}) {
    this.shapeOf = shapeOf;
    this.random = random;
    this.nodes = new Map(); // uid -> {uid, title, class_name, params, layout, addons}
    this.edges = [];
    this.eventEdges = [];
    this.settings = {};
    this.views = [];
    this.addons = [];
    this.extras = {};
    this._handlers = new Map();
  }

  /* ----- events ----- */

  on(type, handler) {
    if (!this._handlers.has(type)) this._handlers.set(type, new Set());
    this._handlers.get(type).add(handler);
    return () => this._handlers.get(type).delete(handler);
  }

  _emit(event) {
    for (const handler of [...(this._handlers.get(event.type) || [])]) handler(event);
    for (const handler of [...(this._handlers.get("change") || [])]) handler(event);
  }

  /* ----- reading and writing the document ----- */

  /** Reads a document of this version; any other format is refused with the reason (there is no migration). */
  static fromJSON(raw, options) {
    if (!raw || typeof raw !== "object") throw new Error("the graph is not a document");
    if (raw.format !== GRAPH_FORMAT) {
      throw new Error(`this graph is format ${raw.format ?? "unknown"}; the editor reads format ${GRAPH_FORMAT} only`);
    }
    const doc = new GraphDoc(options);
    for (const n of raw.nodes || []) {
      doc.nodes.set(n.uid, {
        uid: n.uid, title: n.title ?? "", class_name: n.class_name, params: clone(n.params) || {},
        layout: { x: 0, y: 0, ...clone(n.layout) }, addons: clone(n.addons) || [],
      });
    }
    doc.edges = clone(raw.edges) || [];
    doc.eventEdges = clone(raw.event_edges) || [];
    doc.settings = clone(raw.settings) || {};
    doc.views = clone(raw.views) || [];
    doc.addons = clone(raw.addons) || [];
    for (const key of Object.keys(raw)) if (!KNOWN_KEYS.has(key)) doc.extras[key] = clone(raw[key]);
    return doc;
  }

  /** The whole document (what the library stores); empty optional parts are omitted. */
  toJSON() {
    const out = { ...clone(this.extras), format: GRAPH_FORMAT, settings: clone(this.settings), nodes: [], edges: clone(this.edges), event_edges: clone(this.eventEdges), views: clone(this.views) };
    for (const node of this.nodes.values()) {
      const entry = { uid: node.uid, title: node.title, class_name: node.class_name, params: clone(node.params), layout: clone(node.layout) };
      if (node.addons.length) entry.addons = clone(node.addons);
      out.nodes.push(entry);
    }
    if (this.addons.length) out.addons = clone(this.addons);
    return out;
  }

  /** What a run or a check is sent: the document without `layout` and `views`. */
  runPayload() {
    const out = this.toJSON();
    delete out.views;
    for (const node of out.nodes) delete node.layout;
    return out;
  }

  /** Replaces everything with another document's content. */
  load(raw) {
    const fresh = GraphDoc.fromJSON(raw, { shapeOf: this.shapeOf, random: this.random });
    for (const key of ["nodes", "edges", "eventEdges", "settings", "views", "addons", "extras"]) this[key] = fresh[key];
    this._emit({ type: "reset" });
  }

  clear() {
    this.load({ format: GRAPH_FORMAT });
  }

  /* ----- nodes ----- */

  addNode(className, { x = 0, y = 0, title = "", params = {}, uid, layout = {}, addons = [] } = {}) {
    const id = uid || newUid(new Set(this.nodes.keys()), this.random);
    if (this.nodes.has(id)) throw new Error(`node ${id} exists`);
    const node = { uid: id, title, class_name: className, params: clone(params), layout: { x, y, ...layout }, addons: clone(addons) };
    this.nodes.set(id, node);
    this._emit({ type: "node:add", uid: id });
    return node;
  }

  removeNode(uid) {
    if (!this.nodes.has(uid)) return;
    for (const edge of this.edges.filter((e) => e.from_node === uid || e.to_node === uid)) this.disconnect(edge);
    for (const edge of this.eventEdges.filter((e) => endpointHost(e.from_node) === uid || endpointHost(e.to_node) === uid)) this.disconnectEvent(edge);
    this.nodes.delete(uid);
    this._emit({ type: "node:remove", uid });
  }

  moveNode(uid, x, y) {
    const node = this.nodes.get(uid);
    if (!node || (node.layout.x === x && node.layout.y === y)) return;
    node.layout.x = x;
    node.layout.y = y;
    this._emit({ type: "node:move", uid });
  }

  setTitle(uid, title) {
    const node = this.nodes.get(uid);
    if (!node || node.title === title) return;
    node.title = title;
    this._emit({ type: "node:title", uid });
  }

  /** Presentation state kept in the document's layout (size, collapsed, group overrides); never part of a run. */
  setLayout(uid, patch) {
    const node = this.nodes.get(uid);
    if (!node) return;
    Object.assign(node.layout, patch);
    this._emit({ type: "node:ui", uid });
  }

  setGroupOpen(uid, group, expanded) {
    const node = this.nodes.get(uid);
    if (!node) return;
    node.layout.groups = { ...(node.layout.groups || {}), [group]: expanded };
    this._emit({ type: "node:ui", uid, group });
  }

  /** Sets a field's value; `undefined` removes it (the port's default applies). `origin` tells a view whether it was the editor. */
  setParam(uid, name, value, origin = null) {
    const node = this.nodes.get(uid);
    if (!node) return;
    const had = Object.prototype.hasOwnProperty.call(node.params, name);
    if (value === undefined ? !had : had && JSON.stringify(node.params[name]) === JSON.stringify(value)) return;
    if (value === undefined) delete node.params[name];
    else node.params[name] = clone(value);
    this._emit({ type: "node:params", uid, name, origin });
  }

  /* ----- data wires ----- */

  edgesInto(uid, port) {
    return this.edges.filter((e) => e.to_node === uid && e.to_port === port);
  }

  edgesFrom(uid, port) {
    return this.edges.filter((e) => e.from_node === uid && (port === undefined || e.from_port === port));
  }

  _port(uid, side, name) {
    const node = this.nodes.get(uid);
    const shape = node && this.shapeOf(node);
    return shape ? (shape[side] || []).find((p) => p.name === name) : undefined;
  }

  /** Whether a wire from `from` into `to` would close a loop. */
  wouldCycle(from, to) {
    const seen = new Set();
    const stack = [to];
    while (stack.length) {
      const uid = stack.pop();
      if (uid === from) return true;
      if (seen.has(uid)) continue;
      seen.add(uid);
      for (const e of this.edges) if (e.from_node === uid) stack.push(e.to_node);
    }
    return false;
  }

  /** Why this wire is not allowed, or null. A wire to or from an unknown class is allowed (the server reports it). */
  connectProblem(fromUid, fromPort, toUid, toPort) {
    if (fromUid === toUid) return "a node cannot feed itself";
    const out = this._port(fromUid, "outputs", fromPort);
    const inp = this._port(toUid, "inputs", toPort);
    if (out && inp) {
      const why = wireProblem(out, inp);
      if (why) return why;
    } else if (this.shapeOf(this.nodes.get(fromUid) || {}) && this.shapeOf(this.nodes.get(toUid) || {})) {
      return "no such port";
    }
    if (this.wouldCycle(fromUid, toUid)) return "this wire would make a loop";
    return null;
  }

  /** Adds a wire; returns `{ok, reason?, replaced?}`. A single-input port loses its earlier wire; a `multi` port appends. */
  connect(fromUid, fromPort, toUid, toPort) {
    const reason = this.connectProblem(fromUid, fromPort, toUid, toPort);
    if (reason) return { ok: false, reason };
    const edge = { from_node: fromUid, from_port: fromPort, to_node: toUid, to_port: toPort };
    if (this.edges.some((e) => sameEdge(e, edge))) return { ok: true, replaced: [] };
    const inp = this._port(toUid, "inputs", toPort);
    const replaced = inp && inp.multi ? [] : this.edgesInto(toUid, toPort);
    for (const old of replaced) this.disconnect(old);
    this.edges.push(edge);
    this._emit({ type: "edge:add", edge });
    return { ok: true, replaced };
  }

  disconnect(edge) {
    const index = this.edges.findIndex((e) => sameEdge(e, edge));
    if (index < 0) return;
    const [gone] = this.edges.splice(index, 1);
    this._emit({ type: "edge:remove", edge: gone });
  }

  /** Moves one wire of a `multi` input to another place in that input's order (the argument order). */
  reorderInto(uid, port, from, to) {
    const into = this.edgesInto(uid, port);
    if (from === to || from < 0 || to < 0 || from >= into.length || to >= into.length) return;
    const order = into.slice();
    order.splice(to, 0, order.splice(from, 1)[0]);
    const slots = this.edges.map((e, i) => (e.to_node === uid && e.to_port === port ? i : -1)).filter((i) => i >= 0);
    slots.forEach((slot, i) => { this.edges[slot] = order[i]; });
    this._emit({ type: "edge:order", uid, port });
  }

  /** Wires whose port is not in the node's current resolved shape. */
  dangling() {
    return this.edges.filter((e) => {
      const from = this.nodes.get(e.from_node);
      const to = this.nodes.get(e.to_node);
      const a = from && this.shapeOf(from);
      const b = to && this.shapeOf(to);
      return (a && !(a.outputs || []).some((p) => p.name === e.from_port)) || (b && !(b.inputs || []).some((p) => p.name === e.to_port));
    });
  }

  /* ----- event wires ----- */

  connectEvent(fromNode, fromPort, toNode, toPort) {
    const edge = { from_node: fromNode, from_port: fromPort, to_node: toNode, to_port: toPort };
    const out = this._port(endpointHost(fromNode), "events_out", fromPort);
    const inp = this._port(endpointHost(toNode), "events_in", toPort);
    if (out && inp) {
      const why = eventWireProblem(out, inp);
      if (why) return { ok: false, reason: why };
    }
    if (this.eventEdges.some((e) => sameEdge(e, edge))) return { ok: true };
    this.eventEdges.push(edge);
    this._emit({ type: "event_edge:add", edge });
    return { ok: true };
  }

  disconnectEvent(edge) {
    const index = this.eventEdges.findIndex((e) => sameEdge(e, edge));
    if (index < 0) return;
    const [gone] = this.eventEdges.splice(index, 1);
    this._emit({ type: "event_edge:remove", edge: gone });
  }

  /* ----- addons (a node's, or the graph's with host "graph") ----- */

  _addonList(host) {
    return host === "graph" ? this.addons : (this.nodes.get(host) || {}).addons;
  }

  addAddon(host, kind, params = {}) {
    const list = this._addonList(host);
    if (!list) return null;
    const taken = new Set(list.map((a) => a.id));
    const base = kind.replace(/([a-z0-9])([A-Z])/g, "$1_$2").replace(/([A-Z]+)([A-Z][a-z])/g, "$1_$2").toLowerCase();
    let id = base;
    for (let i = 2; taken.has(id); i++) id = `${base}${i}`;
    const addon = { id, kind, enabled: true, params: clone(params) };
    list.push(addon);
    this._emit({ type: host === "graph" ? "graph:addons" : "node:addons", uid: host });
    return addon;
  }

  setAddon(host, id, patch) {
    const addon = (this._addonList(host) || []).find((a) => a.id === id);
    if (!addon) return;
    if ("enabled" in patch) addon.enabled = Boolean(patch.enabled);
    if (patch.params) Object.assign(addon.params, clone(patch.params));
    this._emit({ type: host === "graph" ? "graph:addons" : "node:addons", uid: host });
  }

  removeAddon(host, id) {
    const list = this._addonList(host);
    if (!list) return;
    const index = list.findIndex((a) => a.id === id);
    if (index < 0) return;
    list.splice(index, 1);
    const end = `${host}.${id}`;
    for (const edge of this.eventEdges.filter((e) => e.from_node === end || e.to_node === end)) this.disconnectEvent(edge);
    for (const edge of this.edges.filter((e) => e.to_node === host && e.to_port.startsWith(`${id}.`))) this.disconnect(edge);
    this._emit({ type: host === "graph" ? "graph:addons" : "node:addons", uid: host });
  }

  /* ----- settings and views ----- */

  /** `null` removes the key (the setting's default applies). */
  setSetting(key, value) {
    if (value === null || value === undefined) {
      if (!(key in this.settings)) return;
      delete this.settings[key];
    } else {
      if (JSON.stringify(this.settings[key]) === JSON.stringify(value)) return;
      this.settings[key] = clone(value);
    }
    this._emit({ type: "settings", key });
  }

  setViews(views) {
    this.views = clone(views) || [];
    this._emit({ type: "views" });
  }

  /* ----- copy and paste ----- */

  /** The nodes and the wires between them, as a clipboard value. */
  copy(uids) {
    const chosen = new Set(uids);
    return {
      nodes: [...this.nodes.values()].filter((n) => chosen.has(n.uid)).map((n) => clone(n)),
      edges: this.edges.filter((e) => chosen.has(e.from_node) && chosen.has(e.to_node)).map((e) => ({ ...e })),
    };
  }

  /** Adds the clipboard's nodes under new uids, moved by (dx, dy), with their inner wires; returns the new uids. */
  paste(clip, dx = 40, dy = 40) {
    const rename = new Map();
    const taken = new Set(this.nodes.keys());
    for (const n of clip.nodes) {
      const uid = newUid(taken, this.random);
      taken.add(uid);
      rename.set(n.uid, uid);
    }
    for (const n of clip.nodes) {
      this.addNode(n.class_name, { uid: rename.get(n.uid), title: n.title, params: n.params, addons: n.addons, layout: { ...n.layout, x: n.layout.x + dx, y: n.layout.y + dy } });
    }
    for (const e of clip.edges) this.connect(rename.get(e.from_node), e.from_port, rename.get(e.to_node), e.to_port);
    return [...rename.values()];
  }
}
