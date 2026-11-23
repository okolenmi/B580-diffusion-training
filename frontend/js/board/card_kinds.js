/* What kind of card a field makes: tones, layout, default size. Pure: no DOM, network or clock. See model.js for the document shape. */

import { findTarget } from "./doc.js";

const JSON_TYPES = new Set(["list", "dict", "tuple"]);

/** A value that needs room to be typed in: a text area, or a list/dict as JSON. */
export function isMultiline(descriptor) {
  return !!descriptor && ((descriptor.ui && descriptor.ui.widget === "textarea") || JSON_TYPES.has(descriptor.type));
}

/** The colours a card's title tab can have (`options.tone`); the look of each is in board.css. */
export const TONES = ["blue", "teal", "green", "amber", "red", "violet", "gray"];

/**
 * How a field card lays out its name and control. `inline` is one row, "name: [value]"; `stacked` is the title tab on top and the
 * value in the box below. A one-line value inside a box (`nested`) is a row, as on a form; on the page it has a tab of its own.
 * Text areas and lists are always stacked. A card's own `options.layout` wins.
 */
export function fieldLayout(block, descriptor, nested = false) {
  const own = block.options && block.options.layout;
  if (own === "inline" || own === "stacked") return own;
  if (!descriptor || isMultiline(descriptor)) return "stacked";
  return nested ? "inline" : "stacked";
}

/** Suggested [w, h] for a new block, in grid units, clamped to `columns`. The sizes below are written for the coarse grid (12 page
    columns, 6 inside a box); on the fine grid (`grid` 2) they are scaled to the container's columns and rows. */
export function defaultSize(ref, modules, columns, nested = false, grid = 1) {
  const { descriptor } = findTarget(ref, modules);
  let w = 4, h = 2;
  if (ref.kind === "display") {
    const size = descriptor && Array.isArray(descriptor.size) ? descriptor.size : [6, 4];
    [w, h] = size;
  } else if (ref.kind === "action") {
    [w, h] = [3, 2];
  } else if (descriptor) {
    if (nested && fieldLayout({ ref }, descriptor, true) === "inline") h = 1; // a form row
    else if (isMultiline(descriptor)) h = Math.max(2, Math.ceil(((descriptor.ui && descriptor.ui.rows) || 3) * 0.45) + 1); // text lines are shorter than grid rows
  }
  if (grid > 1) {
    w = Math.round(w * columns / (nested ? 6 : 12));
    h *= grid;
  }
  if (columns < 8 * grid) w = columns;
  return [Math.max(1, Math.min(w, columns)), Math.max(1, h)];
}
