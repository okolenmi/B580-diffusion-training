/* ---------------------------------------------------------------------------
   board/model.js -- a board's document rules, as pure functions.

   A board is `{columns, blocks, graph}`: a grid of *blocks* (cards) that point at modules of a graph, and the "draft graph"
   whose parameter objects the field blocks edit (an App's snapshot, or a stand-in over the open graph document). The stored
   shapes are backend/domain/board.py (a view) and backend/domain/app.py (an app); a block is

     {id, ref: {scope: "node"|"graph"|"note", node?, kind: "field"|"display"|"note", name}, x, y, w, h, collapsed, title, style, options}

   or, for a section / tabbed card, `container: {columns, tabs: [{id, title, blocks}]}` in place of `ref`. Ids are unique across the whole
   document, containers included, so every id-minting function takes the set of ids in use.

   Nothing in this file touches the DOM, the network or a clock, so every
   rule a person relies on -- "what can I add", "which fields does the
   starter page expose", "what exactly will Run send" -- is checked by
   number in frontend/tests/board_model.test.mjs.

   The *modules* argument is what the backend's POST /boards/modules
   returns: {nodes: [{id, class_name, display_name, known, fields,
   displays, stream}], graph_fields, graph_displays}. The backend, not this
   file, decides what a node offers; this file only arranges it.
   --------------------------------------------------------------------------- */

import { addAt, resolve } from "./layout.js";

export const GRAPH_STREAM = "graph:run";

/* ---- identity ------------------------------------------------------------ */

/** A stable string for a block's target: used as the "is it placed" key. */
export function refKey(ref) {
  if (ref.scope === "note") return `note/${ref.name}`;
  return ref.scope === "graph"
    ? `graph/${ref.kind}/${ref.name}`
    : `node/${ref.node}/${ref.kind}/${ref.name}`;
}

export const nodeRef = (node, kind, name) => ({ scope: "node", node, kind, name });
export const noteRef = (name) => ({ scope: "note", kind: "note", name });
export const graphRef = (kind, name) => ({ scope: "graph", kind, name });

/** A block id not used in `blocks` (ids are [A-Za-z0-9_-]{1,40}). */
export function newBlockId(blocks, used = null) {
  const taken = used || collectIds(blocks);
  let n = taken.size + 1;
  while (taken.has(`b${n}`)) n++;
  return `b${n}`;
}

/* ---- walking a tree of cards -------------------------------------------- */

export const isContainer = (block) => !!block.container;

/** Every card at any depth, containers included, parents before children. */
export function* walkCards(blocks) {
  for (const b of blocks) {
    yield b;
    if (b.container) for (const tab of b.container.tabs) yield* walkCards(tab.blocks);
  }
}

/** The cards that point at something (`ref`), at any depth. */
export function* leafCards(blocks) {
  for (const b of walkCards(blocks)) if (b.ref) yield b;
}

export const collectIds = (blocks) => new Set([...walkCards(blocks)].map((b) => b.id));

/** How many containers deep `blocks` go (0 = no container). */
export function depthOf(blocks) {
  let deepest = 0;
  for (const b of blocks) {
    if (!b.container) continue;
    for (const tab of b.container.tabs) deepest = Math.max(deepest, 1 + depthOf(tab.blocks));
  }
  return deepest;
}

export function emptyDoc(graph = null, columns = 12) {
  return { columns, graph, blocks: [] };
}

/* ---- finding what a block points at ------------------------------------- */

/** `{node, descriptor}` for a ref against the modules answer. `descriptor`
    is null when the target is gone; `node` is null for graph scope. */
export function findTarget(ref, modules) {
  if (ref.scope === "note") return { node: null, descriptor: { name: ref.name, kind: "note" } };
  if (!modules) return { node: null, descriptor: null };
  if (ref.scope === "graph") {
    const list = ref.kind === "field" ? modules.graph_fields : modules.graph_displays;
    return { node: null, descriptor: (list || []).find((d) => d.name === ref.name) || null };
  }
  const node = (modules.nodes || []).find((n) => n.id === ref.node) || null;
  if (!node) return { node: null, descriptor: null };
  const list = ref.kind === "field" ? node.fields : ref.kind === "action" ? node.actions : node.displays;
  return { node, descriptor: (list || []).find((d) => d.name === ref.name) || null };
}

/** Heading and sub-heading of a block. A saved custom title wins. */
export function titleFor(block, modules) {
  if (block.container) {
    const n = block.container.tabs.length;
    return { title: block.title || "Section", subtitle: n > 1 ? `${n} tabs` : "Section", defaultTitle: "Section" };
  }
  if (block.ref.scope === "note") return { title: block.title || "Note", subtitle: "Note", defaultTitle: "Note" };
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
  } else if (ref.kind === "action") {
    [w, h] = [3, 2];
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
  const placed = new Set([...leafCards(blocks)].map((b) => refKey(b.ref)));
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
      ...(node.actions || []).map((a) =>
        item(nodeRef(node.id, "action", a.name), a.label, a.doc || "")),
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
export function addBlock(doc, ref, modules, used = null) {
  const key = refKey(ref);
  const taken = used || collectIds(doc.blocks);
  if ([...leafCards(doc.blocks)].some((b) => refKey(b.ref) === key)) return doc;
  const [w, h] = defaultSize(ref, modules, doc.columns);
  const block = {
    id: newBlockId(doc.blocks, taken), ref, x: 0, y: 0, w, h,
    collapsed: false, title: null, options: {},
  };
  return { ...doc, blocks: addAt(doc.blocks, block, doc.columns) };
}

/** A new empty note: free text on the board (instructions, a reminder), not tied to the graph. */
export function addNoteBlock(doc, used = null, look = "text") {
  const names = new Set([...leafCards(doc.blocks)].filter((b) => b.ref.scope === "note").map((b) => b.ref.name));
  let n = 1;
  while (names.has(`note${n}`)) n++;
  const heading = look === "heading";
  const block = {
    id: newBlockId(doc.blocks, used), ref: noteRef(`note${n}`), x: 0, y: 0,
    w: heading ? doc.columns : Math.min(6, doc.columns), h: heading ? 1 : 2,
    collapsed: false, title: null, style: heading ? "bare" : undefined, options: { text: "", look },
  };
  return { ...doc, blocks: addAt(doc.blocks, block, doc.columns) };
}

export function removeBlock(doc, id) {
  return { ...doc, blocks: resolve(doc.blocks.filter((b) => b.id !== id), doc.columns) };
}

/* ---- sections and tabs ---------------------------------------------------- */

export const MAX_TABS = 12;
export const MAX_DEPTH = 3;

/** A new section card: a titled box with one empty tab. `blocks` (already laid out for `columns`) fill that tab. */
export function addContainerBlock(doc, used = null, { title = null, tabTitle = "Tab 1", blocks = [], columns = 12, w = null, h = 4 } = {}) {
  const taken = used || collectIds(doc.blocks);
  const block = {
    id: newBlockId(doc.blocks, taken), container: { columns, tabs: [{ id: "t1", title: tabTitle, blocks }] },
    x: 0, y: 0, w: Math.min(w || doc.columns, doc.columns), h,
    collapsed: false, title, style: "outlined", options: {},
  };
  return { ...doc, blocks: addAt(doc.blocks, block, doc.columns) };
}

/**
 * A section holding every field a node offers (the ones the graph does not feed and that are not advanced): "give me this node's
 * settings as one form". Nothing new is bound -- they are ordinary field cards -- so Run and the exposure rules are unchanged.
 * Fields already placed elsewhere are skipped. Returns `doc` unchanged when nothing is left to add.
 */
export function addNodeSection(doc, nodeId, modules, used = null) {
  const node = ((modules && modules.nodes) || []).find((n) => n.id === nodeId);
  if (!node) return doc;
  const taken = new Set(used || collectIds(doc.blocks));
  const placed = new Set([...leafCards(doc.blocks)].map((b) => refKey(b.ref)));
  const columns = 6;
  const inner = [];
  for (const f of node.fields) {
    const ref = nodeRef(nodeId, "field", f.name);
    if (f.wired || f.advanced || placed.has(refKey(ref))) continue;
    const id = newBlockId(inner, taken);
    taken.add(id);
    const [, h] = defaultSize(ref, modules, columns);
    inner.push({ id, ref, x: 0, y: 0, w: columns, h, collapsed: false, title: null, options: {} });
  }
  if (!inner.length) return doc;
  const laid = resolve(inner, columns);
  const rows = Math.max(...laid.map((b) => b.y + b.h));
  return addContainerBlock(doc, taken, { title: node.display_name, tabTitle: "Settings", blocks: laid, columns, w: Math.min(6, doc.columns), h: rows + 2 });
}

/** The container with every tab re-laid for `columns`: cards narrowed to fit and pushed apart where they now collide. */
export function withColumns(container, columns) {
  const fit = (b) => ({ ...b, w: Math.min(b.w, columns), x: Math.max(0, Math.min(b.x, columns - Math.min(b.w, columns))) });
  return {
    ...container, columns,
    tabs: container.tabs.map((t) => ({ ...t, blocks: resolve(t.blocks.map(fit), columns) })),
  };
}

const slug = (n) => `t${n}`;

export function addTab(tabs, title = null) {
  if (tabs.length >= MAX_TABS) return tabs;
  const used = new Set(tabs.map((t) => t.id));
  let n = tabs.length + 1;
  while (used.has(slug(n))) n++;
  return [...tabs, { id: slug(n), title: title || `Tab ${tabs.length + 1}`, blocks: [] }];
}

export function removeTab(tabs, id) {
  return tabs.length <= 1 ? tabs : tabs.filter((t) => t.id !== id);
}

export function renameTab(tabs, id, title) {
  const text = (title || "").trim();
  return text ? tabs.map((t) => (t.id === id ? { ...t, title: text.slice(0, 120) } : t)) : tabs;
}

export function moveTab(tabs, id, delta) {
  const i = tabs.findIndex((t) => t.id === id);
  const j = i + delta;
  if (i < 0 || j < 0 || j >= tabs.length) return tabs;
  const out = [...tabs];
  [out[i], out[j]] = [out[j], out[i]];
  return out;
}

/** Every tab a card can be moved into, as `{tabId, owner, label, columns, depth}`; `owner` is the container card id ("" for the
    page). A card never goes into itself or its own descendants, and not past MAX_DEPTH. */
export function moveTargets(root, id) {
  const moving = [...walkCards(root.tabs.flatMap((t) => t.blocks))].find((b) => b.id === id);
  if (!moving) return [];
  const height = moving.container ? 1 + depthOf(moving.container.tabs.flatMap((t) => t.blocks)) : 0;
  const out = [];
  const visit = (container, owner, trail, depth) => {
    for (const tab of container.tabs) {
      const label = [...trail, tab.title].join(" \u203a ");
      if (depth + height <= MAX_DEPTH) out.push({ tabId: tab.id, owner, label, columns: container.columns, depth });
      for (const b of tab.blocks) {
        if (b.container && b.id !== id) visit(b.container, b.id, [...trail, tab.title, b.title || "Section"], depth + 1);
      }
    }
  };
  visit(root, "", [], 0);
  return out;
}

/** `{owner, tabId}` of the tab holding card `id` (owner "" = the page), or null. */
export function locateCard(root, id) {
  const visit = (container, owner) => {
    for (const tab of container.tabs) {
      for (const b of tab.blocks) {
        if (b.id === id) return { owner, tabId: tab.id };
        if (b.container) { const hit = visit(b.container, b.id); if (hit) return hit; }
      }
    }
    return null;
  };
  return visit(root, "");
}

/** The root with card `id` taken out of wherever it is and placed in tab `tabId` of container `owner` ("" = the page). A new tree;
    the root itself when the move is not allowed. */
export function moveCard(root, id, owner, tabId) {
  if (!moveTargets(root, id).some((t) => t.owner === owner && t.tabId === tabId)) return root;
  const tree = JSON.parse(JSON.stringify(root));
  let moving = null;
  const strip = (container) => {
    for (const tab of container.tabs) {
      const keep = [];
      for (const b of tab.blocks) {
        if (b.id === id) { moving = b; continue; }
        if (b.container) strip(b.container);
        keep.push(b);
      }
      tab.blocks = keep;
    }
  };
  strip(tree);
  if (!moving) return root;
  const find = (container) => {
    for (const tab of container.tabs) {
      for (const b of tab.blocks) {
        if (b.id === owner) return b.container;
        if (b.container) { const hit = find(b.container); if (hit) return hit; }
      }
    }
    return null;
  };
  const dest = owner ? find(tree) : tree;
  const tab = dest && dest.tabs.find((t) => t.id === tabId);
  if (!tab) return root;
  const w = Math.min(moving.w, dest.columns);
  tab.blocks = addAt(tab.blocks, { ...moving, x: 0, y: 0, w }, dest.columns);
  return tree;
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
  // A button the graph can act on goes first on the right: "Save now" is what a person looks for during a run.
  for (const node of modules.nodes || []) {
    if (!node.known) continue;
    for (const a of node.actions || []) if (a.available) displayRefs.unshift(nodeRef(node.id, "action", a.name));
  }
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
 * The `values` body of POST /apps/{name}/run: the current value of
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
  for (const b of leafCards(doc.blocks)) {
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
  return JSON.stringify({ c: doc.columns, b: doc.blocks.map(blockKey) });
}

/** The same for a whole container (an App's page): its columns, tabs and every card in them. */
export function containerKey(container) {
  return JSON.stringify(containerParts(container));
}

const blockKey = (b) => [
  b.id, b.ref ? refKey(b.ref) : null, b.x, b.y, b.w, b.h, !!b.collapsed, b.title || null, b.style || null, b.options || {},
  b.container ? containerParts(b.container) : null,
];
const containerParts = (c) => [c.columns, c.tabs.map((t) => [t.id, t.title, t.blocks.map(blockKey)])];

/** block id -> status string from a /resolve answer. */
export function statusByBlock(resolution) {
  const out = {};
  for (const b of (resolution && resolution.blocks) || []) out[b.id] = b.status;
  return out;
}
