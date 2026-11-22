/* Document identity: refs, ids, walking a tree of cards, finding what a block points at. Pure: no DOM, network or clock. See model.js for the document shape. */


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
