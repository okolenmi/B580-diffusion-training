// @ts-check
/* ---------------------------------------------------------------------------
   board/layout.js -- the grid arithmetic of a template page. Pure: no DOM,
   no I/O, nothing but numbers in and numbers out, which is why it is
   unit-tested (frontend/tests/board_layout.test.mjs) instead of eyeballed.

   A block is {id, x, y, w, h, collapsed?, ...}: a rectangle on a grid of
   `columns` columns and unbounded rows. The rules are the usual dashboard
   ones, chosen because people already have the muscle memory for them:

     * blocks never overlap;
     * moving or resizing a block pushes whatever is in its way DOWN;
     * everything then settles upward (vertical compaction), so there are
       no holes unless something is holding the space open;
     * a collapsed block occupies a single row (its header) while it keeps
       its stored `h`, so expanding it again restores what it was.

   Every function returns NEW block objects; callers keep the old array as
   an undo snapshot for free.

   Determinism matters more than cleverness here: the same input must give
   the same layout every time, or two people looking at one saved page see
   two different pages. Ties are broken by (y, x, id).
   --------------------------------------------------------------------------- */

/** @typedef {{id: string, x: number, y: number, w: number, h: number, collapsed?: boolean}} Rect */

/** Rows a block occupies right now (a collapsed block is just its header). */
export function effH(b) {
  return b.collapsed ? 1 : b.h;
}

/** Do two blocks overlap? Touching edges do not count. */
export function collides(a, b) {
  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + effH(b) && b.y < a.y + effH(a);
}

/** Total order used everywhere a deterministic sequence is needed. */
function byPosition(a, b) {
  return a.y - b.y || a.x - b.x || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
}

/** Clamp one block onto the grid: integer, w in 1..columns, x so it fits,
    y and h at least 0 / 1. Never throws -- a saved document from a wider
    grid must still open on a narrower one. */
export function clamp(b, columns) {
  const w = Math.min(Math.max(1, Math.round(b.w) || 1), columns);
  const x = Math.min(Math.max(0, Math.round(b.x) || 0), columns - w);
  const y = Math.max(0, Math.round(b.y) || 0);
  const h = Math.max(1, Math.round(b.h) || 1);
  return { ...b, x, y, w, h };
}

/**
 * Settle a layout.
 *
 * `pinnedId` (optional) is the block the user is holding: it keeps its
 * requested position and everything else arranges itself around it. Without
 * one, blocks are placed in reading order.
 *
 * @param {Rect[]} blocks
 * @param {number} columns
 * @param {string|null} [pinnedId]
 * @returns {Rect[]} in the input's order
 */
export function resolve(blocks, columns, pinnedId = null) {
  const clamped = blocks.map((b) => clamp(b, columns));
  /** @type {Map<string, Rect>} */
  const out = new Map();
  /** @type {Rect[]} */
  const placed = [];

  const place = (b) => {
    // The lowest y at which b clears everything already placed. Starting
    // from 0 is what makes this compaction; each collision jumps straight
    // to the bottom of the offender instead of stepping row by row.
    let y = 0;
    for (;;) {
      const probe = { ...b, y };
      const hit = placed.find((p) => collides(probe, p));
      if (!hit) break;
      y = hit.y + effH(hit);
    }
    const settled = { ...b, y };
    placed.push(settled);
    out.set(b.id, settled);
  };

  const pinned = pinnedId === null ? null : clamped.find((b) => b.id === pinnedId);
  if (pinned) {
    // The held block goes down exactly where requested (not compacted):
    // the user is pointing at a spot, and sliding it up would make the
    // drop land somewhere they did not choose.
    placed.push(pinned);
    out.set(pinned.id, pinned);
  }
  for (const b of clamped.filter((b) => b !== pinned).sort(byPosition)) place(b);
  return blocks.map((b) => /** @type {Rect} */ (out.get(b.id)));
}

/** Move a block to (x, y); the rest make way. */
export function moveTo(blocks, id, x, y, columns) {
  const next = blocks.map((b) => (b.id === id ? { ...b, x, y } : b));
  return resolve(next, columns, id);
}

/** Resize a block (w, h in grid units); the rest make way. */
export function resizeTo(blocks, id, w, h, columns) {
  const next = blocks.map((b) => (b.id === id ? { ...b, w, h } : b));
  return resolve(next, columns, id);
}

/** Collapse / expand a block, then let the rest settle. */
export function setCollapsed(blocks, id, collapsed, columns) {
  return resolve(blocks.map((b) => (b.id === id ? { ...b, collapsed } : b)), columns);
}

/**
 * First place a new w x h block fits, scanning rows top to bottom and
 * columns left to right -- so a new field drops into a gap if there is
 * one, and at the bottom if there is not.
 */
export function findFreeSlot(blocks, w, h, columns) {
  const width = Math.min(Math.max(1, w), columns);
  const bottom = blocks.reduce((m, b) => Math.max(m, b.y + effH(b)), 0);
  for (let y = 0; y <= bottom; y++) {
    for (let x = 0; x + width <= columns; x++) {
      const probe = { id: "", x, y, w: width, h, collapsed: false };
      if (!blocks.some((b) => collides(probe, b))) return { x, y };
    }
  }
  return { x: 0, y: bottom };
}

/** Append `block` (its x/y are ignored) at the first free slot. */
export function addAt(blocks, block, columns) {
  const w = Math.min(Math.max(1, block.w), columns);
  const { x, y } = findFreeSlot(blocks, w, block.h, columns);
  return resolve([...blocks, { ...block, w, x, y }], columns);
}

/** Reading order -- what a one-column (phone) layout stacks in. */
export function flowOrder(blocks) {
  return [...blocks].sort(byPosition);
}

/** Rows the whole layout spans (for sizing the container). */
export function totalRows(blocks) {
  return blocks.reduce((m, b) => Math.max(m, b.y + effH(b)), 0);
}

/**
 * Which grid cell is a pixel offset in? Used by drag and resize: the
 * pointer moved (dx, dy) px from where it started and a cell is
 * (colW + gap) x (rowH + gap) px.
 */
export function cellDelta(dx, dy, colW, rowH, gap) {
  return {
    dCols: Math.round(dx / (colW + gap)),
    dRows: Math.round(dy / (rowH + gap)),
  };
}

/** Pixel width of one column for a container `width` px wide. */
export function columnWidth(width, columns, gap) {
  return Math.max(1, (width - gap * (columns - 1)) / columns);
}

/**
 * Keyboard moves: arrows move by one cell, shift+arrows resize. Returns
 * the new layout, or the same array when the key means nothing here.
 */
export function keyNudge(blocks, id, key, shift, columns) {
  const b = blocks.find((x) => x.id === id);
  if (!b) return blocks;
  const step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[key];
  if (!step) return blocks;
  if (shift) return resizeTo(blocks, id, b.w + step[0], Math.max(1, b.h + step[1]), columns);
  return moveTo(blocks, id, b.x + step[0], Math.max(0, b.y + step[1]), columns);
}
