/* datasets/packing.js -- "balanced" layout: pictures of very different shapes get about the same AREA, which rows of
   equal height cannot do (there a wide picture is twice the size of a tall one).

   Pictures are dealt into three streams by shape (tall, square, wide), each cut into blocks of a few pictures that keep
   their relative order. A block is packed into a rectangle by a slicing tree -- two parts side by side or one above
   the other, recursively -- so every picture keeps its exact proportions and the block has straight edges; of all the
   trees the one whose pictures come nearest the wanted area wins. Pictures of one shape tile into nearly equal areas
   whatever the tree. Blocks go into columns, each into the shortest one, and are listed by the position of their first
   picture, so the order on screen stays close to the order given. Pure arithmetic (no DOM), so it is testable in node. */

const BLOCK = 6;
const BLOCK_SHAPE = 1.5;
const CACHE_MAX = 4000;
const cache = new Map();

/* All slicing trees over ratios[i..j) in order: {r: width/height of the whole, shares: each picture's part of its
   area, tree: how to cut it}. Joining side by side adds the ratios; stacking adds the reciprocals. */
function trees(ratios, i, j, memo) {
  const key = i * 16 + j;
  if (memo.has(key)) return memo.get(key);
  let out;
  if (j - i === 1) {
    out = [{ r: ratios[i], shares: [1], tree: i }];
  } else {
    out = [];
    for (let cut = i + 1; cut < j; cut++) {
      for (const a of trees(ratios, i, cut, memo)) {
        for (const b of trees(ratios, cut, j, memo)) {
          const side = a.r + b.r;
          out.push({
            r: side,
            shares: [...a.shares.map((s) => (s * a.r) / side), ...b.shares.map((s) => (s * b.r) / side)],
            tree: { across: true, a, b },
          });
          const stack = (a.r * b.r) / side;
          out.push({
            r: stack,
            shares: [...a.shares.map((s) => (s * b.r) / side), ...b.shares.map((s) => (s * a.r) / side)],
            tree: { across: false, a, b },
          });
        }
      }
    }
  }
  memo.set(key, out);
  return out;
}

/** The tree whose pictures come nearest the wanted area when the block is `width` wide. */
function bestTree(ratios, width, area) {
  let best = null;
  let bestCost = Infinity;
  for (const t of trees(ratios, 0, ratios.length, new Map())) {
    let cost = 0;
    for (const s of t.shares) {
      const e = Math.log((s * width * width) / t.r / area);
      cost += e * e;
    }
    if (cost < bestCost) { bestCost = cost; best = t; }
  }
  return best;
}

function place(node, offset, x, y, w, h, out) {
  if (typeof node.tree === "number") {
    out[offset + node.tree] = { x, y, w, h };
    return;
  }
  const { across, a, b } = node.tree;
  if (across) {
    const wa = (w * a.r) / (a.r + b.r);
    place(a, offset, x, y, wa, h, out);
    place(b, offset, x + wa, y, w - wa, h, out);
  } else {
    const ha = (h * (1 / a.r)) / (1 / a.r + 1 / b.r);
    place(a, offset, x, y, w, ha, out);
    place(b, offset, x, y + ha, w, h - ha, out);
  }
}

/** Keep a picture's tree for a block that comes back unchanged (scrolling in more pictures, redraws). */
function blockTree(ratios, width, area) {
  const key = `${ratios.map((r) => r.toFixed(3)).join(",")}|${Math.round(width)}|${Math.round(area)}`;
  if (!cache.has(key)) {
    if (cache.size >= CACHE_MAX) cache.delete(cache.keys().next().value);
    cache.set(key, bestTree(ratios, width, area));
  }
  return cache.get(key);
}

const SHAPES = [(r) => r < 0.85, (r) => r < 1.18];
const shapeOf = (r) => SHAPES.findIndex((is) => is(r)) + 1 || 3; // 1 tall, 2 square, 3 wide

const SAMPLE = 120;

/** Deal the pictures into blocks and pack them into `columns` columns. Boxes come out without the gap. */
function fill(ratios, space, area, columns, complete) {
  const colWidth = space / columns;
  const streams = [[], [], []];
  ratios.forEach((r, n) => streams[shapeOf(r) - 1].push(n));
  const blocks = [];
  for (const stream of streams) {
    for (let at = 0; at < stream.length; at += BLOCK) {
      const members = stream.slice(at, at + BLOCK);
      if (members.length === BLOCK || complete) blocks.push(members);
    }
  }
  blocks.sort((a, b) => a[0] - b[0]);

  const bottoms = new Array(columns).fill(0);
  const boxes = new Array(ratios.length).fill(null);
  const order = [];
  for (const members of blocks) {
    // an unfinished block is narrower, or one picture would fill a whole column
    const w = members.length === BLOCK ? colWidth : colWidth * Math.min(1, Math.max(0.4, (members.length / BLOCK) * 1.25));
    const t = blockTree(members.map((n) => ratios[n]), w, area);
    const col = bottoms.indexOf(Math.min(...bottoms));
    const h = w / t.r;
    const local = [];
    place(t, 0, col * colWidth, bottoms[col], w, h, local);
    local.forEach((b, k) => { boxes[members[k]] = b; });
    order.push(...members);
    bottoms[col] += h;
  }
  return { boxes, order, bottoms };
}

/* Blocks are whole pictures in whole trees, so a column width gives areas only at certain steps. Of the nearby column
   counts, take the one whose first pictures come nearest the wanted area. */
function pickColumns(ratios, space, area) {
  const near = Math.max(1, Math.round(space / Math.sqrt(BLOCK * area * BLOCK_SHAPE)));
  const sample = ratios.slice(0, SAMPLE);
  let best = near;
  let bestCost = Infinity;
  for (let k = Math.max(1, near - 1); k <= near + 2; k++) {
    const boxes = fill(sample, space, area, k, true).boxes;
    const cost = boxes.reduce((sum, b) => sum + Math.log((b.w * b.h) / area) ** 2, 0) / boxes.length;
    if (cost < bestCost - 1e-9) { bestCost = cost; best = k; }
  }
  return best;
}

/**
 * @param {number[]} ratios   width / height of each picture, in the order given
 * @param {number} width      the space to fill, in px
 * @param {number} size       the side of an average picture, in px (the tile-size setting)
 * @param {number} gap        space between pictures, in px
 * @param {boolean} complete  false while more pictures are still to come: a stream's last, unfinished block waits
 * @returns {{boxes: ({x: number, y: number, w: number, h: number}|null)[], order: number[], height: number}}
 *   boxes by input index (null = waiting); order = input indices as they read on screen
 */
export function pack(ratios, width, size, gap, complete = true) {
  const space = width + gap; // every picture is inset by gap/2, so the outer edge ends up flush
  const area = size * size;
  const { boxes, order, bottoms } = fill(ratios, space, area, pickColumns(ratios, space, area), complete);
  return {
    boxes: boxes.map((b) => b && { x: b.x, y: b.y, w: Math.max(1, b.w - gap), h: Math.max(1, b.h - gap) }),
    order,
    height: Math.max(0, Math.max(...bottoms) - gap),
  };
}
