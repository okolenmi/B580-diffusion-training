/* Editing the card tree: add, remove, move, group, sections and tabs. Pure: no DOM, network or clock. See model.js for the document shape. */

import { collectIds, depthOf, leafCards, newBlockId, nodeRef, noteRef, refKey, walkCards } from "./doc.js";
import { addAt, resolve } from "./layout.js";
import { defaultSize } from "./card_kinds.js";

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
    const [, h] = defaultSize(ref, modules, columns, true);
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

/**
 * Put the cards `ids` into one new box (a section) that takes their place: the box has one tab holding the cards as they were
 * arranged, and more tabs can be added to it afterwards. The cards must share one tab of one container, and the box must not
 * nest deeper than allowed. A new tree; the root itself when that is not possible.
 */
export function groupCards(root, ids) {
  const wanted = new Set(ids);
  if (!wanted.size) return root;
  const spots = [...wanted].map((id) => locateCard(root, id));
  const first = spots[0];
  if (spots.some((s) => !s || s.owner !== first.owner || s.tabId !== first.tabId)) return root;
  const tree = JSON.parse(JSON.stringify(root));
  const find = (container, depth) => {
    for (const tab of container.tabs) {
      for (const b of tab.blocks) {
        if (b.id === first.owner) return { container: b.container, depth: depth + 1 };
        if (b.container) { const hit = find(b.container, depth + 1); if (hit) return hit; }
      }
    }
    return null;
  };
  const dest = first.owner ? find(tree, 0) : { container: tree, depth: 0 };
  const tab = dest && dest.container.tabs.find((t) => t.id === first.tabId);
  if (!tab) return root;
  const picked = tab.blocks.filter((b) => wanted.has(b.id));
  if (picked.length !== wanted.size) return root;
  if (dest.depth + 1 + depthOf(picked) > MAX_DEPTH) return root;
  const left = Math.min(...picked.map((b) => b.x));
  const top = Math.min(...picked.map((b) => b.y));
  const right = Math.max(...picked.map((b) => b.x + b.w));
  const bottom = Math.max(...picked.map((b) => b.y + b.h));
  const columns = Math.min(Math.max(right - left, 3), dest.container.columns);
  const box = {
    id: newBlockId(null, collectIds(tree.tabs.flatMap((t) => t.blocks))),
    container: { columns, tabs: [{ id: "t1", title: "Tab 1", blocks: picked.map((b) => ({ ...b, x: b.x - left, y: b.y - top })) }] },
    x: Math.min(left, dest.container.columns - columns), y: top, w: columns, h: bottom - top + 2,
    collapsed: false, title: null, style: "outlined", options: {},
  };
  tab.blocks = resolve([...tab.blocks.filter((b) => !wanted.has(b.id)), box], dest.container.columns);
  return tree;
}

export function patchBlock(doc, id, patch) {
  return { ...doc, blocks: doc.blocks.map((b) => (b.id === id ? { ...b, ...patch } : b)) };
}
