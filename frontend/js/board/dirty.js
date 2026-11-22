/* Dirty detection: what counts as a layout change. Pure: no DOM, network or clock. See model.js for the document shape. */

import { refKey } from "./doc.js";

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
