/* Values: the draft graph is where a field's value lives. Pure: no DOM, network or clock. See model.js for the document shape. */

import { leafCards } from "./doc.js";

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
 * server merges memory settings rather than clearing them. `keepCleared` sends it as null too: that is for the page's own copy of what
 * it shows, which must remember that the field was emptied.
 *
 * `statuses` (block id -> status from /resolve) filters out blocks the
 * server would refuse: wired fields and orphans.
 */
export function valuesPayload(doc, statuses = {}, { keepCleared = false } = {}) {
  const nodes = {};
  const memory = {};
  for (const b of leafCards(doc.blocks)) {
    if (b.ref.kind !== "field") continue;
    const st = statuses[b.id];
    if (st && st !== "ok") continue;
    const v = getFieldValue(doc, b.ref);
    if (b.ref.scope === "graph") {
      if (v !== undefined || keepCleared) memory[b.ref.name.split(".", 2)[1]] = v === undefined ? null : v;
    } else {
      (nodes[b.ref.node] = nodes[b.ref.node] || {})[b.ref.name] = v === undefined ? null : v;
    }
  }
  const out = {};
  if (Object.keys(nodes).length) out.nodes = nodes;
  if (Object.keys(memory).length) out.memory = memory;
  return out;
}

/**
 * Put a past run's `values` (the shape `valuesPayload` sent) back into the draft, for every field card the page still has:
 * a `null` clears the parameter, a key the run did not record is left alone (it was wired, or the card is newer than the run).
 * Returns how many fields changed.
 */
export function applyValues(doc, values) {
  let changed = 0;
  for (const b of leafCards(doc.blocks)) {
    if (b.ref.kind !== "field") continue;
    const bag = b.ref.scope === "graph" ? values.memory : values.nodes && values.nodes[b.ref.node];
    const key = b.ref.scope === "graph" ? b.ref.name.split(".", 2)[1] : b.ref.name;
    if (!bag || !Object.hasOwn(bag, key)) continue;
    if (setFieldValue(doc, b.ref, bag[key])) changed++;
  }
  return changed;
}
