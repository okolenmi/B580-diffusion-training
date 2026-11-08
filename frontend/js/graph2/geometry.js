/* ---------------------------------------------------------------------------
   graph2/geometry.js -- pure geometry: where pins sit on a minimized node,
   how a wire is drawn between two points.
   --------------------------------------------------------------------------- */

/** Vertical positions of `count` pins spread over `height` (never one shared point: every port keeps its own pin). */
export function pinOffsets(count, height, margin = 14) {
  if (count <= 0) return [];
  if (count === 1) return [Math.round(height / 2)];
  const span = Math.max(height - 2 * margin, 0);
  return Array.from({ length: count }, (_, i) => Math.round(margin + (span * i) / (count - 1)));
}

/** Height a minimized node needs so its pins do not crowd: the larger side decides. */
export function pinnedHeight(inputs, outputs, headHeight = 28, pitch = 16) {
  return headHeight + Math.max(inputs, outputs, 1) * pitch;
}

/** A cubic curve from an output (ax, ay) to an input (bx, by); the handles lengthen with the distance so backward wires still bend. */
export function wirePath(ax, ay, bx, by) {
  const reach = Math.max(40, Math.min(180, Math.abs(bx - ax) / 2 + Math.abs(by - ay) / 4));
  return `M ${ax} ${ay} C ${ax + reach} ${ay}, ${bx - reach} ${by}, ${bx} ${by}`;
}

/** An event wire runs top to bottom: from an output strip (bottom of a node) to an input strip (top of another). */
export function eventPath(ax, ay, bx, by) {
  const reach = Math.max(30, Math.min(140, Math.abs(by - ay) / 2 + Math.abs(bx - ax) / 4));
  return `M ${ax} ${ay} C ${ax} ${ay + reach}, ${bx} ${by - reach}, ${bx} ${by}`;
}

/** Several wires into one `multi` port leave the socket a little apart: the i-th of n is offset around the port's own y. */
export function fanOffset(index, count, step = 5) {
  return (index - (count - 1) / 2) * step;
}

/** Converts a client point to the canvas's own coordinates given its pan and zoom. */
export function toCanvas(clientX, clientY, origin, pan, zoom) {
  return { x: (clientX - origin.left - pan.x) / zoom, y: (clientY - origin.top - pan.y) / zoom };
}

/** Pan that keeps the canvas point under the cursor fixed when the zoom changes. */
export function zoomAround(cursor, pan, zoom, nextZoom) {
  const k = nextZoom / zoom;
  return { x: cursor.x - (cursor.x - pan.x) * k, y: cursor.y - (cursor.y - pan.y) * k };
}

export function clampZoom(zoom, min = 0.25, max = 2) {
  return Math.min(max, Math.max(min, zoom));
}
