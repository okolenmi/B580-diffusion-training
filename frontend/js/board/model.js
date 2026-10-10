/* ---------------------------------------------------------------------------
   board/model.js -- the template document, as pure functions.

   A template document (backend/domain/template.py) is a board of *blocks*
   plus a snapshot of the graph they belong to:

     {format, kind: "template"|"monitor", columns, source_graph, graph,
      blocks: [{id, ref: {scope: "node"|"graph", node?, kind: "field"|
                "display", name}, x, y, w, h, collapsed, title, options}]}

   Nothing in this file touches the DOM, the network or a clock, so every
   rule a person relies on -- "what can I add", "which fields does the
   starter page expose", "what exactly will Run send" -- is checked by
   number in frontend/tests/board_model.test.mjs.

   The *modules* argument is what the backend's POST /templates/modules
   returns: {nodes: [{id, class_name, display_name, known, fields,
   displays, stream}], graph_fields, graph_displays}. The backend, not this
   file, decides what a node offers; this file only arranges it.
   --------------------------------------------------------------------------- */

import { addAt, resolve } from "./layout.js";

export const GRAPH_STREAM = "graph:run";

/* ---- identity ------------------------------------------------------------ */

/** A stable string for a block's target: used as the "is it placed" key. */
export function refKey(ref) {
  return ref.scope === "graph"
    ? `graph/${ref.kind}/${ref.name}`
    : `node/${ref.node}/${ref.kind}/${ref.name}`;
}

export const nodeRef = (node, kind, name) => ({ scope: "node", node, kind, name });
export const graphRef = (kind, name) => ({ scope: "graph", kind, name });

/** A block id not used in `blocks` (ids are [A-Za-z0-9_-]{1,40}). */
export function newBlockId(blocks) {
  const used = new Set(blocks.map((b) => b.id));
  let n = blocks.length + 1;
  while (used.has(`b${n}`)) n++;
  return `b${n}`;
}

export function emptyDoc(kind, graph = null, columns = 12) {
  return {
    format: 1, kind, columns, source_graph: null,
    graph: kind === "template" ? (graph || { nodes: [], edges: [] }) : graph,
    blocks: [],
  };
}

/* ---- finding what a block points at ------------------------------------- */

/** `{node, descriptor}` for a ref against the modules answer. `descriptor`
    is null when the target is gone; `node` is null for graph scope. */
export function findTarget(ref, modules) {
  if (!modules) return { node: null, descriptor: null };
  if (ref.scope === "graph") {
    const list = ref.kind === "field" ? modules.graph_fields : modules.graph_displays;
    return { node: null, descriptor: (list || []).find((d) => d.name === ref.name) || null };
  }
  const node = (modules.nodes || []).find((n) => n.id === ref.node) || null;
  if (!node) return { node: null, descriptor: null };
  const list = ref.kind === "field" ? node.fields : node.displays;
  return { node, descriptor: (list || []).find((d) => d.name === ref.name) || null };
}

/** Heading and sub-heading of a block. A saved custom title wins. */
export function titleFor(block, modules) {
  const { node, descriptor } = findTarget(block.ref, modules);
  const base = descriptor
    ? descriptor.label || descriptor.title || descriptor.name
    : block.ref.name;
  const where = block.ref.scope === "graph"
    ? "Graph"
    : node ? `${node.display_name} · ${node.id}` : block.ref.node;
  // A node's live view shares its name with the setting it displays
  // ("Learning rate"); say which one this is.
  const live = block.ref.kind === "display" && block.ref.scope === "node" ? " \u00b7 live view" : "";
  return { title: block.title || base, subtitle: where + live, defaultTitle: base };
}

/* ---- sizes --------------------------------------------------------------- */

const JSON_TYPES = new Set(["list", "dict", "tuple"]);

/** Suggested [w, h] for a new block, in grid units, clamped to `columns`. */
export function defaultSize(ref, modules, columns) {
  const { descriptor } = findTarget(ref, modules);
  let w = 4, h = 2;
  if (ref.kind === "display") {
    const size = descriptor && Array.isArray(descriptor.size) ? descriptor.size : [6, 4];
    [w, h] = size;
  } else if (descriptor) {
    if (JSON_TYPES.has(descriptor.type)) h = 3;
  }
  if (columns < 8) w = columns;
  return [Math.max(1, Math.min(w, columns)), Math.max(1, h)];
}

/* ---- the "what can I add" tray ------------------------------------------ */

/**
 * Groups of things the graph offers, each item flagged `placed` when the
 * document already shows it. A field the graph feeds from upstream is not
 * offered at all (nothing to type into it); advanced fields are offered
 * only when `showAdvanced`, except that an already-placed one always
 * appears so it can be found again.
 *
 * @returns {{key: string, title: string, scope: string, items: object[]}[]}
 */
export function buildAvailable(modules, blocks, { showAdvanced = false } = {}) {
  if (!modules) return [];
  const placed = new Set(blocks.map((b) => refKey(b.ref)));
  const item = (ref, label, sub, extra = {}) => ({
    ref, key: refKey(ref), label, sub, placed: placed.has(refKey(ref)),
    kind: ref.kind, advanced: false, ...extra,
  });
  const keep = (it) => it.placed || showAdvanced || !it.advanced;

  const groups = [];

  const graphItems = [
    ...(modules.graph_displays || []).map((d) =>
      item(graphRef("display", d.name), d.title, d.doc || "")),
    ...(modules.graph_fields || []).map((f) =>
      item(graphRef("field", f.name), f.label || f.name, f.doc || "",
        { advanced: !!f.advanced, group: f.group || "" })),
  ].filter(keep);
  if (graphItems.length) {
    groups.push({ key: "graph", title: "This graph", scope: "graph", items: graphItems });
  }

  for (const node of modules.nodes || []) {
    if (!node.known) continue;
    const items = [
      ...node.displays.map((d) =>
        item(nodeRef(node.id, "display", d.name), d.title, d.doc || "")),
      ...node.fields
        .filter((f) => !f.wired)
        .map((f) =>
          item(nodeRef(node.id, "field", f.name), f.label || f.name, f.doc || "",
            { advanced: !!f.advanced, group: f.group || "", type: f.type })),
    ].filter(keep);
    if (items.length) {
      groups.push({
        key: `node:${node.id}`, title: `${node.display_name} · ${node.id}`,
        scope: "node", items,
      });
    }
  }
  return groups;
}

/* ---- adding / removing blocks ------------------------------------------- */

/** The document with a new block for `ref` at the first free slot
    (a no-op, returning the same doc, when the ref is already placed). */
export function addBlock(doc, ref, modules) {
  const key = refKey(ref);
  if (doc.blocks.some((b) => refKey(b.ref) === key)) return doc;
  const [w, h] = defaultSize(ref, modules, doc.columns);
  const block = {
    id: newBlockId(doc.blocks), ref, x: 0, y: 0, w, h,
    collapsed: false, title: null, options: {},
  };
  return { ...doc, blocks: addAt(doc.blocks, block, doc.columns) };
}

export function removeBlock(doc, id) {
  return { ...doc, blocks: resolve(doc.blocks.filter((b) => b.id !== id), doc.columns) };
}

export function patchBlock(doc, id, patch) {
  return { ...doc, blocks: doc.blocks.map((b) => (b.id === id ? { ...b, ...patch } : b)) };
}

/* ---- the starter page --------------------------------------------------- */

/**
 * A first page for a graph, so "new from this graph" is not a blank board.
 *
 * Exposed: the fields the graph's author actually set (or that cannot run
 * without a value), skipping advanced and wired ones, capped at
 * `maxFields` so a big graph does not become a wall of inputs -- plus the
 * graph's VRAM budget. Shown: every live view the nodes declare, the run
 * status and the memory chart. Fields stack on the left, live views on the
 * right; the person rearranges it in edit mode.
 *
 * Deliberately a heuristic and a *starting point*: it cannot know which
 * knobs a beginner should see. Everything it leaves out is one click away
 * in the tray.
 */
export function autoFill(modules, columns = 12, { maxFields = 16 } = {}) {
  const narrow = columns < 8;
  const leftW = narrow ? columns : Math.max(3, Math.round(columns / 3));
  const rightW = narrow ? columns : columns - leftW;
  const rightX = narrow ? 0 : leftW;

  const fieldRefs = [];
  const graphBudget = (modules.graph_fields || []).find((f) => f.name === "memory.vram_max_mb");
  if (graphBudget) fieldRefs.push(graphRef("field", graphBudget.name));
  let nodeFieldCount = 0;
  for (const node of modules.nodes || []) {
    if (!node.known) continue;
    for (const f of node.fields) {
      if (nodeFieldCount >= maxFields) break;
      if (f.wired || f.advanced) continue;
      const needed = f.required && (f.default === null || f.default === undefined);
      if (f.has_value || needed) {
        fieldRefs.push(nodeRef(node.id, "field", f.name));
        nodeFieldCount++;
      }
    }
  }

  const displayRefs = [graphRef("display", "run")];
  for (const node of modules.nodes || []) {
    if (!node.known) continue;
    for (const d of node.displays) displayRefs.push(nodeRef(node.id, "display", d.name));
  }
  if ((modules.graph_displays || []).some((d) => d.name === "memory")) {
    displayRefs.push(graphRef("display", "memory"));
  }

  const blocks = [];
  let yLeft = 0;
  for (const ref of fieldRefs) {
    const [, h] = defaultSize(ref, modules, columns);
    blocks.push({
      id: `b${blocks.length + 1}`, ref, x: 0, y: yLeft, w: leftW, h,
      collapsed: false, title: null, options: {},
    });
    yLeft += h;
  }
  let yRight = narrow ? yLeft : 0;
  for (const ref of displayRefs) {
    const [w, h] = defaultSize(ref, modules, columns);
    const bw = Math.min(w, rightW);
    blocks.push({
      id: `b${blocks.length + 1}`, ref, x: rightX, y: yRight, w: bw, h,
      collapsed: false, title: null, options: {},
    });
    yRight += h;
  }
  return resolve(blocks, columns);
}

/* ---- values: the draft graph is where a field's value lives ------------- */

function nodeOf(doc, id) {
  return ((doc.graph && doc.graph.nodes) || []).find((n) => n.id === id) || null;
}

/** Current value of a field block's target in the draft graph, or
    `undefined` when unset (the node's default applies). */
export function getFieldValue(doc, ref) {
  if (ref.scope === "graph") {
    const key = ref.name.split(".", 2)[1];
    const mem = (doc.graph && doc.graph.memory) || {};
    return mem[key];
  }
  const node = nodeOf(doc, ref.node);
  return node && node.params ? node.params[ref.name] : undefined;
}

/** Set (or, for undefined/null/"", clear) a field in the draft graph.
    Mutates `doc.graph` in place -- the page owns its draft -- and returns
    whether anything changed. */
export function setFieldValue(doc, ref, value) {
  const clear = value === undefined || value === null || value === "";
  const before = getFieldValue(doc, ref);
  if (ref.scope === "graph") {
    const key = ref.name.split(".", 2)[1];
    if (!doc.graph) doc.graph = { nodes: [], edges: [] };
    const mem = (doc.graph.memory = doc.graph.memory || {});
    if (clear) delete mem[key];
    else mem[key] = value;
  } else {
    const node = nodeOf(doc, ref.node);
    if (!node) return false;
    node.params = node.params || {};
    if (clear) delete node.params[ref.name];
    else node.params[ref.name] = value;
  }
  return JSON.stringify(before) !== JSON.stringify(getFieldValue(doc, ref));
}

/**
 * The `values` body of POST /templates/{name}/run: the current value of
 * every field block the page places. A cleared field is sent as null so the
 * server clears it too (the saved graph may hold a value the person just
 * emptied); a graph-memory field that is unset is left out, because the
 * server merges memory settings rather than clearing them.
 *
 * `statuses` (block id -> status from /resolve) filters out blocks the
 * server would refuse: wired fields and orphans.
 */
export function valuesPayload(doc, statuses = {}) {
  const nodes = {};
  const memory = {};
  for (const b of doc.blocks) {
    if (b.ref.kind !== "field") continue;
    const st = statuses[b.id];
    if (st && st !== "ok") continue;
    const v = getFieldValue(doc, b.ref);
    if (b.ref.scope === "graph") {
      if (v !== undefined) memory[b.ref.name.split(".", 2)[1]] = v;
    } else {
      (nodes[b.ref.node] = nodes[b.ref.node] || {})[b.ref.name] = v === undefined ? null : v;
    }
  }
  const out = {};
  if (Object.keys(nodes).length) out.nodes = nodes;
  if (Object.keys(memory).length) out.memory = memory;
  return out;
}

/* ---- dirty detection ----------------------------------------------------- */

/** What "layout" means for unsaved-changes purposes: where blocks are and
    how they are labelled -- not their live data. */
export function layoutKey(doc) {
  return JSON.stringify({
    c: doc.columns,
    b: doc.blocks.map((b) => [
      b.id, refKey(b.ref), b.x, b.y, b.w, b.h, !!b.collapsed, b.title || null,
      b.options || {},
    ]),
  });
}

export function graphKey(doc) {
  return JSON.stringify(doc.graph || null);
}

export const docKey = (doc) => `${layoutKey(doc)}|${graphKey(doc)}`;

/** The document as it should be PUT: format stamped, nothing live in it. */
export function toSaveDoc(doc) {
  return {
    ...doc,
    format: 1,
    blocks: doc.blocks.map((b) => ({
      ...b, collapsed: !!b.collapsed, title: b.title || null, options: b.options || {},
    })),
  };
}

/** block id -> status string from a /resolve answer. */
export function statusByBlock(resolution) {
  const out = {};
  for (const b of (resolution && resolution.blocks) || []) out[b.id] = b.status;
  return out;
}
