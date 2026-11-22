/* The starter page drafted from a graph's modules. Pure: no DOM, network or clock. See model.js for the document shape. */

import { resolve } from "./layout.js";
import { findTarget, graphRef, nodeRef } from "./doc.js";
import { defaultSize } from "./card_kinds.js";

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
  const { fieldRefs, displayRefs } = draftRefs(modules, maxFields);
  return placeRefs(fieldRefs, displayRefs, modules, columns, { next: 1, fields: true });
}

/** What a draft offers: the settable fields worth showing and every live view, in the order the graph's nodes come. */
function draftRefs(modules, maxFields) {
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
  for (const name of ["memory", "history", "queue"]) {
    if ((modules.graph_displays || []).some((d) => d.name === name)) displayRefs.push(graphRef("display", name));
  }
  return { fieldRefs, displayRefs };
}

/** Fields stacked on the left (when `fields`), views on the right (the whole width without fields); `at.next` numbers the ids so tabs share them. */
function placeRefs(fieldRefs, displayRefs, modules, columns, at) {
  const narrow = columns < 8;
  const leftW = narrow ? columns : Math.max(3, Math.round(columns / 3));
  const rightW = narrow || !at.fields ? columns : columns - leftW;
  const rightX = narrow || !at.fields ? 0 : leftW;
  const blocks = [];
  const add = (ref, x, y, w, h) => {
    blocks.push({ id: `b${at.next++}`, ref, x, y, w, h, collapsed: false, title: null, options: {} });
    return h;
  };
  let yLeft = 0;
  for (const ref of fieldRefs) yLeft += add(ref, 0, yLeft, leftW, defaultSize(ref, modules, columns)[1]);
  let yRight = narrow ? yLeft : 0;
  for (const ref of displayRefs) {
    const [w, h] = defaultSize(ref, modules, columns);
    yRight += add(ref, rightX, yRight, Math.min(w, rightW), h);
  }
  return resolve(blocks, columns);
}

const WATCHED_KINDS = new Set(["chart", "bars", "scalar"]);
const RUN_LIST_KINDS = new Set(["history", "queue"]);

/**
 * The page a new app starts with: the same cards as `autoFill`, sorted into tabs so a training app is not one long heap --
 * Run (inputs, buttons, status, results), Charts (everything plotted), Runs (history and queue). A tab with nothing for it is left out,
 * and a graph that offers only one is a single tab as before.
 */
export function autoPage(modules, columns = 12, { maxFields = 16 } = {}) {
  const { fieldRefs, displayRefs } = draftRefs(modules, maxFields);
  const kindOf = (ref) => {
    const { descriptor } = findTarget(ref, modules);
    return ref.kind === "action" ? "action" : (descriptor && descriptor.kind) || "";
  };
  const groups = { run: [], charts: [], runs: [] };
  for (const ref of displayRefs) {
    const kind = kindOf(ref);
    groups[WATCHED_KINDS.has(kind) ? "charts" : RUN_LIST_KINDS.has(kind) ? "runs" : "run"].push(ref);
  }
  const at = { next: 1, fields: true };
  const tabs = [{ id: "run", title: "Run", blocks: placeRefs(fieldRefs, groups.run, modules, columns, at) }];
  at.fields = false;
  if (groups.charts.length) tabs.push({ id: "charts", title: "Charts", blocks: placeRefs([], groups.charts, modules, columns, at) });
  if (groups.runs.length) tabs.push({ id: "runs", title: "Runs", blocks: placeRefs([], groups.runs, modules, columns, at) });
  if (tabs.length === 1) tabs[0] = { id: "main", title: "Main", blocks: tabs[0].blocks };
  return tabs;
}
