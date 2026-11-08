/* ---------------------------------------------------------------------------
   graph2/layout.js -- where to put nodes that have no position (a graph made
   by the command line or a starter). Columns are the longest chain of wires
   leading to a node, so data flows left to right; inside a column nodes keep
   the order of what feeds them, so wires stay short. Pure: it returns
   positions and touches nothing.
   --------------------------------------------------------------------------- */

const COLUMN = 340;
const GAP = 40;

/** True when a document carries no placement: more than one node and every one at the origin. */
export function isUnplaced(nodes) {
  return nodes.length > 1 && nodes.every((n) => !n.layout || (!n.layout.x && !n.layout.y));
}

/**
 * @param {string[]} uids
 * @param {Array<[string, string]>} wires  [from, to] pairs of node uids (events included)
 * @param {(uid:string)=>number} heightOf  the node's height, as drawn
 * @returns {Map<string, {x:number, y:number}>}
 */
export function arrange(uids, wires, heightOf = () => 240) {
  const known = new Set(uids);
  const into = new Map(uids.map((u) => [u, []]));
  for (const [from, to] of wires) if (from !== to && known.has(from) && known.has(to)) into.get(to).push(from);

  // A bounded relaxation: a loop made of event wires cannot push a column further than the node count.
  const column = new Map(uids.map((u) => [u, 0]));
  for (let pass = 0; pass < uids.length; pass += 1) {
    let moved = false;
    for (const uid of uids) {
      const want = Math.max(0, ...into.get(uid).map((p) => column.get(p) + 1));
      if (want > column.get(uid) && want < uids.length) { column.set(uid, want); moved = true; }
    }
    if (!moved) break;
  }

  const columns = [];
  for (const uid of uids) (columns[column.get(uid)] ||= []).push(uid);
  const row = new Map();
  const out = new Map();
  columns.forEach((list, x) => {
    if (!list) return;
    const centre = (uid) => {
      const ys = into.get(uid).filter((p) => row.has(p)).map((p) => row.get(p));
      return ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : Infinity;
    };
    const ordered = list.map((uid, i) => ({ uid, key: centre(uid), i })).sort((a, b) => (a.key - b.key) || (a.i - b.i));
    let y = 0;
    for (const { uid } of ordered) {
      out.set(uid, { x: x * COLUMN, y });
      row.set(uid, y + heightOf(uid) / 2);
      y += heightOf(uid) + GAP;
    }
  });
  return out;
}
