/* ---------------------------------------------------------------------------
   editor/state.js -- GraphDoc: the editable graph document (M7).

   One source of truth for nodes/edges/positions. Mutations go through
   methods and notify `onChange(reason, origin)`:
     - "structure"  nodes/edges added/removed/replaced -> full re-render
     - "params"     a param value committed -> node bodies + inspector
                    refresh; `origin` ("inspector" | "canvas" | null)
                    tells editor.js who committed, so the inspector
                    skips rebuilding only when IT was the editor (focus
                    survives keystrokes; every other origin refreshes it)

   Wire forms (docs 02 §graphs, 05 §6):
     run/validate: {nodes:[{id,class_name,params}],
                    edges:[{from_node,from_port,to_node,to_port}]}
     library:      same + top-level `layout:{id:{x,y}}` -- node positions
                    ride as an extra top-level key because LibraryGraphIn
                    allows extras through while GraphNodeIn does not.
     legacy draft (localStorage `ng_graph_v1`, import only):
                    {nodes:[{id,class_name,x,y,paramValues}],
                     connections:[{fromNode,fromPort,toNode,toPort}], nextId}

   Rules inherited from the legacy editor (behavior, not code):
     - one wire per input (a new edge replaces the old one), no self loops;
     - a param is submitted unless it is empty or an edge feeds that input
       (edges overwrite params by design -- docs 05 §4);
     - pure handle inputs (non-primitive, socketable) never carry params;
     - unknown node classes are KEPT (rendered as placeholders) -- the
       library stores graphs verbatim, validation happens at run time.
   --------------------------------------------------------------------------- */

const PRIMITIVE_TYPES = new Set(["int", "float", "str", "bool", "Path", "Any", "Callable"]);

/** This input gets a value widget (and a param slot in the payload). */
export function isWidgetInput(port) {
  if (port.widget_only) return true;      // checkbox-style: widget, no socket
  return PRIMITIVE_TYPES.has(port.type);  // primitive: socket + widget
}

/** This input gets a wire socket (everything except widget_only). */
export function hasSocket(port) {
  return !port.widget_only;
}

/** Legacy wire-type rule: `from`'s MRO must contain `to`'s type; Any passes. */
export function typesCompatible(fromPort, toPort) {
  const fromMro = fromPort.type_mro || [];
  const toType = toPort.type;
  const toMro = toPort.type_mro || [];
  if (toType === "Any" || toMro.includes("Any")) return true;
  if (fromMro.includes("Any")) return true;
  return fromMro.includes(toType);
}

const GRID_X = 270;
const GRID_Y = 210;

export class GraphDoc {
  constructor() {
    this.nodes = new Map();   // id -> {id, class_name, x, y, params}
    this.edges = [];          // [{from_node, from_port, to_node, to_port}]
    this.nextId = 1;
    // Graph-wide settings (the memory budget). Stored in the graph, so a
    // saved graph, a run and a template built on it all agree. Keys other
    // than the ones the Settings section edits (policy, device) are kept
    // verbatim: this document must not silently drop what it did not draw.
    this.memory = {};
    this.classByName = {};    // editor.js fills this after /graphs/nodes
    this.onChange = null;     // (reason, origin) => void
  }

  changed(reason, origin = null) {
    if (this.onChange) this.onChange(reason, origin);
  }

  classOf(node) {
    return this.classByName[node.class_name] || null;
  }

  get size() {
    return this.nodes.size;
  }

  /* ---- nodes ---- */

  addNode(className, x, y) {
    let id;
    do {
      id = "n" + this.nextId++;
    } while (this.nodes.has(id));
    const node = {
      id,
      class_name: className,
      x: Math.round(x), // infinite plane: negative coordinates allowed
      y: Math.round(y),
      params: {},
    };
    this.nodes.set(id, node);
    this.changed("structure");
    return node;
  }

  removeNode(id) {
    if (!this.nodes.delete(id)) return;
    this.edges = this.edges.filter((e) => e.from_node !== id && e.to_node !== id);
    this.changed("structure");
  }

  /** Rename a node id and rewire its edges. false = rejected (empty,
      whitespace, or duplicate); true = applied or no-op. */
  renameNode(oldId, newId) {
    const target = (newId || "").trim();
    if (!target) return false;
    if (target === oldId) return true;
    if (/\s/.test(target) || this.nodes.has(target)) return false;
    const node = this.nodes.get(oldId);
    if (!node) return false;
    this.nodes.delete(oldId);
    node.id = target;
    this.nodes.set(target, node);
    for (const e of this.edges) {
      if (e.from_node === oldId) e.from_node = target;
      if (e.to_node === oldId) e.to_node = target;
    }
    this.changed("structure");
    return true;
  }

  /* ---- edges ---- */

  edgeInto(nodeId, port) {
    return this.edges.find((e) => e.to_node === nodeId && e.to_port === port) || null;
  }

  addEdge(fromNode, fromPort, toNode, toPort) {
    if (fromNode === toNode) return null;
    // one wire per input: the newest edge wins (legacy rule)
    this.edges = this.edges.filter((e) => !(e.to_node === toNode && e.to_port === toPort));
    const edge = { from_node: fromNode, from_port: fromPort, to_node: toNode, to_port: toPort };
    this.edges.push(edge);
    this.changed("structure");
    return edge;
  }

  removeEdge(edge) {
    const i = this.edges.indexOf(edge);
    if (i >= 0) {
      this.edges.splice(i, 1);
      this.changed("structure");
    }
  }

  clear() {
    this.nodes.clear();
    this.edges = [];
    this.nextId = 1;
    this.memory = {};
    this.changed("structure");
  }

  /* ---- payload forms ---- */

  paramsFor(node) {
    const cls = this.classOf(node);
    if (!cls) return { ...node.params }; // unknown class: keep verbatim, validate flags it
    const out = {};
    for (const p of cls.inputs) {
      if (!isWidgetInput(p)) continue;               // pure handle: edges only
      if (this.edgeInto(node.id, p.name)) continue;  // fed by wire, param redundant
      const v = node.params[p.name];
      if (v === undefined || v === null || v === "") continue; // empty never submitted
      out[p.name] = v;
    }
    return out;
  }

  toRunPayload() {
    const payload = {
      nodes: [...this.nodes.values()].map((n) => ({
        id: n.id,
        class_name: n.class_name,
        params: this.paramsFor(n),
      })),
      edges: this.edges.map((e) => ({ ...e })),
    };
    // Sent only when something was set: an untouched graph submits exactly
    // what it always did, and the server's defaults apply.
    if (Object.keys(this.memory).length) payload.memory = { ...this.memory };
    return payload;
  }

  /** Everything needed to rebuild the canvas exactly, for the browser-side
      draft (editor/draft.js). Unlike the run payload this keeps params that
      are empty-looking or fed by a wire, verbatim: a draft that dropped
      them would hand back a different graph than the one that was open. */
  toDraft() {
    const layout = {};
    const nodes = [...this.nodes.values()].map((n) => {
      layout[n.id] = { x: n.x, y: n.y };
      return { id: n.id, class_name: n.class_name, params: { ...n.params } };
    });
    return {
      format: 1,
      nodes,
      edges: this.edges.map((e) => ({ ...e })),
      layout,
      memory: { ...this.memory },
    };
  }

  toLibraryPayload(description = "") {
    const run = this.toRunPayload();
    const layout = {};
    for (const n of this.nodes.values()) layout[n.id] = { x: n.x, y: n.y };
    const out = { format: 1, nodes: run.nodes, edges: run.edges, layout, description };
    // The budget is part of the graph: a saved graph that forgot it would
    // run with the server default the next time it is loaded.
    if (run.memory) out.memory = run.memory;
    return out;
  }

  /* ---- loading ---- */

  /**
   * Replace the document from a library payload / execution snapshot /
   * legacy draft. Positions come from `layout`, else from x/y on the node
   * itself (legacy drafts), else a grid. Legacy `connections`/`paramValues`
   * keys are accepted alongside `edges`/`params`.
   */
  load(payload) {
    this.nodes.clear();
    this.edges = [];
    const positions = payload.layout || {};
    const nodes = payload.nodes || [];
    nodes.forEach((n, i) => {
      const pos = positions[n.id] || n;
      const x = Number.isFinite(pos.x) ? pos.x : 40 + (i % 4) * GRID_X;
      const y = Number.isFinite(pos.y) ? pos.y : 40 + Math.floor(i / 4) * GRID_Y;
      this.nodes.set(n.id, {
        id: n.id,
        class_name: n.class_name,
        x: Math.round(x),
        y: Math.round(y),
        params: { ...(n.params || n.paramValues || {}) },
      });
    });
    const raw = payload.edges || payload.connections || [];
    for (const c of raw) {
      const e =
        c.from_node !== undefined
          ? c
          : { from_node: c.fromNode, from_port: c.fromPort, to_node: c.toNode, to_port: c.toPort };
      if (this.nodes.has(e.from_node) && this.nodes.has(e.to_node)) {
        this.edges.push({ from_node: e.from_node, from_port: e.from_port, to_node: e.to_node, to_port: e.to_port });
      }
    }
    this.nextId = this._scanNextId();
    const mem = payload.memory;
    this.memory = mem && typeof mem === "object" && !Array.isArray(mem) ? { ...mem } : {};
    this.changed("structure");
  }

  /** Set (or, for undefined/null/"", clear) one graph setting. */
  setMemory(key, value) {
    const clear = value === undefined || value === null || value === "";
    if (clear) delete this.memory[key];
    else this.memory[key] = value;
    this.changed("memory", "settings");
  }

  /** Loaded node ids whose class is absent from the catalog (kept, not dropped). */
  unknownClasses() {
    const seen = new Set();
    for (const n of this.nodes.values()) {
      if (!this.classByName[n.class_name]) seen.add(n.class_name);
    }
    return [...seen];
  }

  _scanNextId() {
    let max = 0;
    for (const id of this.nodes.keys()) {
      const m = /^n(\d+)$/.exec(id);
      if (m) max = Math.max(max, parseInt(m[1], 10));
    }
    return max + 1;
  }
}
