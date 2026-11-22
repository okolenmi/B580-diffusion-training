/* The "what can I add" tray. Pure: no DOM, network or clock. See model.js for the document shape. */

import { graphRef, leafCards, nodeRef, refKey } from "./doc.js";

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
