/* ---------------------------------------------------------------------------
   graph2/node_tabs.js -- what a node's tabs hold (rework/31-node-addons.md,
   section 7). Pure functions over catalog data and the document's node; no DOM.

     Main      the node's own ports and fields
     Optional  optional inputs that take a wire; one is off until enabled
               (or wired), and an enabled one is a socket on the card
     Addons    the addons attached to the node, and the ones that fit
   --------------------------------------------------------------------------- */

import { hasSocket, isField } from "./ports.js";

/** Optional inputs that take a wire: not required, not a plain value field. */
export function optionalSockets(inputs) {
  return inputs.filter((p) => !p.required && !isField(p) && hasSocket(p));
}

/** The optional inputs drawn on the card: the ones the person enabled, plus every wired one (a wire never hides). */
export function shownOptional(node, inputs, wired) {
  const enabled = new Set(node.layout.optional || []);
  return optionalSockets(inputs).filter((p) => enabled.has(p.name) || wired(p.name));
}

/** The addon kinds the node's slots accept, minus kinds that would fill a single slot already filled. */
export function addonChoices(info, kinds, addons) {
  const slots = new Map((info?.addon_slots || []).map((s) => [s.name, s]));
  const filled = new Set(addons.map((a) => kinds.get(a.kind)?.slot));
  return [...kinds.values()].filter((k) => {
    const slot = slots.get(k.slot);
    if (!slot || (k.fits.length && !k.fits.includes(info.class_name))) return false;
    return slot.many || !filled.has(k.slot);
  });
}

/** `[{id, label, count}]` in display order; Main is always first and the others appear only when they have something. */
export function tabsOf({ shape, info, node, kinds }) {
  const tabs = [{ id: "main", label: "Main", count: 0 }];
  const optional = optionalSockets(shape.inputs);
  if (optional.length) tabs.push({ id: "optional", label: "Optional", count: (node.layout.optional || []).length });
  // Event-only addons fit every host; they alone do not earn a tab on every node.
  if (node.addons.length || addonChoices(info, kinds, node.addons).some((k) => k.slot !== "events")) tabs.push({ id: "addons", label: "Addons", count: node.addons.length });
  return tabs;
}

/** The tab to show: the one the node remembers if it still exists, else Main. */
export function activeTab(tabs, node) {
  return tabs.some((t) => t.id === node.layout.tab) ? node.layout.tab : "main";
}

/** Which tab holds a port that has no socket on the card right now, so the hub can open it. */
export function tabOfHidden(portName, shape, node) {
  if (portName.includes(".") || portName.startsWith("slot:")) return "addons";
  const port = shape.inputs.find((p) => p.name === portName);
  if (port && optionalSockets([port]).length) return "optional";
  return "main";
}
