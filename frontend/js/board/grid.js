/* ---------------------------------------------------------------------------
   board/grid.js -- the size of a grid cell, and the one-time move of an older page to the finer grid. Pure.

   A container carries `grid`: absent or 1 is the coarse grid pages started with (48 px rows, ten columns' worth of steps), 2 the
   fine one (twice the columns, rows of 20 px, so a card moves in half the steps). Everything stored (x, y, w, h, columns) is in
   the units of its container's grid; `refineContainer` converts a coarse container, and the boxes inside it, once.
   --------------------------------------------------------------------------- */

/** The most columns a container may have (backend/domain/board.py). */
export const MAX_COLUMNS = 24;

export const FINE = 2;

const METRICS = { 1: { rowPx: 48, gapPx: 10 }, [FINE]: { rowPx: 20, gapPx: 8 } };

/** The row height and the gap of a grid; two fine rows plus their gap are as tall as one coarse row. */
export function gridMetrics(grid) {
  return METRICS[grid] || METRICS[1];
}

/** The container (and every box inside it) on the fine grid: columns doubled while they fit the limit, rows always doubled. Cards keep
    their place and shape. A container already fine is returned as it is. */
export function refineContainer(container) {
  if ((container.grid || 1) === FINE) return container;
  const columns = container.columns * FINE <= MAX_COLUMNS ? container.columns * FINE : container.columns;
  const across = columns / container.columns;
  const refine = (b) => {
    const moved = { ...b, x: b.x * across, w: b.w * across, y: b.y * FINE, h: b.h * FINE };
    return b.container ? { ...moved, container: refineContainer(b.container) } : moved;
  };
  return { ...container, columns, grid: FINE, tabs: container.tabs.map((t) => ({ ...t, blocks: t.blocks.map(refine) })) };
}
