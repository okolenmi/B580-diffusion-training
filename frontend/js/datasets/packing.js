/* datasets/packing.js -- "balanced" layout: pictures of very different shapes get about the same AREA, which rows of
   equal height cannot do (there a wide picture is twice the size of a tall one).

   The space is cut into columns of different widths. Every column serves one shape: a narrow column takes tall
   pictures, a wide one takes wide pictures, and a picture is as wide as its column, so its area is the column width
   squared over its ratio -- about the wanted area whenever the column's shape is near the picture's. Column shapes
   come from the quantiles of a sample of the set, and a column's width follows its share of the pictures, so all
   columns grow at the same pace and end at about the same height. Each new picture goes to the column that suits its
   shape unless that one is far ahead of the others. Nothing is reflowed when pictures are added; only a change of
   width or tile size starts over. Empty space is only the ragged bottom. Pure arithmetic (no DOM), testable in node. */

const SAMPLE = 400;
const SHAPE_TOLERANCE = 0.15; // ln-ratio distance that costs as much as ...
const LEVEL_WEIGHT = 1.5;     // ... being one tile-size taller than the shortest column, scaled by this
const MAX_COLUMNS = 12;

/** Column shapes and widths for `count` columns over `space` px, from the ratios of a sample. */
function plan(sample, space, size, count) {
  const sorted = [...sample].sort((a, b) => a - b);
  const total = sorted.length;
  let share = new Array(count).fill(1 / count);
  let shapes = [];
  for (let pass = 0; pass < 12; pass++) {
    let from = 0;
    let acc = 0;
    shapes = share.map((s, j) => {
      acc += s;
      const to = j === count - 1 ? total : Math.max(from + 1, Math.round(acc * total));
      const group = sorted.slice(from, Math.min(total, to));
      from = to;
      return Math.exp(group.reduce((sum, r) => sum + Math.log(r), 0) / (group.length || 1));
    });
    const wide = shapes.map(Math.sqrt);
    const sum = wide.reduce((a, b) => a + b, 0);
    share = wide.map((v) => v / sum);
  }
  const natural = shapes.map((s) => size * Math.sqrt(s));
  const fit = space / natural.reduce((a, b) => a + b, 0);
  return { shapes, widths: natural.map((w) => w * fit), fit };
}

export class ColumnPacker {
  /** @param {number} space width to fill (px, gap included)  @param {number} size side of an average picture (px)
   *  @param {number[]} sample ratios (width / height) the columns are shaped after
   *  @param {object} [ready] a finished column plan, to skip choosing one */
  constructor(space, size, sample, ready = null) {
    this.size = size;
    this.columns = ready ?? ColumnPacker.choose(sample.slice(0, SAMPLE), space, size);
    this.lefts = [];
    let x = 0;
    for (const w of this.columns.widths) { this.lefts.push(x); x += w; }
    this.bottoms = new Array(this.columns.widths.length).fill(0);
  }

  /* The column count whose sample comes out nearest the wanted area: widths are whole columns, so counts differ. */
  static choose(sample, space, size) {
    const area = size * size;
    let best = null;
    for (let count = 1; count <= Math.min(MAX_COLUMNS, Math.max(1, sample.length)); count++) {
      const p = plan(sample, space, size, count);
      if (p.widths.some((w) => w < size * 0.35)) continue;
      const trial = new ColumnPacker(space, size, sample, p);
      const cost = sample.reduce((sum, r) => { const b = trial.place(r); return sum + Math.log((b.w * b.h) / area) ** 2; }, 0) / sample.length
        + 0.5 * trial.unevenness();
      if (!best || cost < best.cost - 1e-9) best = { cost, ...p };
    }
    return best ?? plan(sample, space, size, 1);
  }

  /** The cell for one more picture: {x, y, w, h}, gap not yet taken off. */
  place(ratio) {
    const level = Math.min(...this.bottoms);
    let col = 0;
    let best = Infinity;
    this.columns.shapes.forEach((shape, j) => {
      const off = Math.log(ratio / shape) / SHAPE_TOLERANCE;
      const cost = off * off + (LEVEL_WEIGHT * (this.bottoms[j] - level)) / this.size;
      if (cost < best) { best = cost; col = j; }
    });
    const w = this.columns.widths[col];
    const cell = { x: this.lefts[col], y: this.bottoms[col], w, h: w / ratio };
    this.bottoms[col] += cell.h;
    return cell;
  }

  get height() { return Math.max(0, ...this.bottoms); }

  /** How uneven the column ends are, in tile sizes (0 = level). */
  unevenness() { return (Math.max(...this.bottoms) - Math.min(...this.bottoms)) / (this.size * Math.max(1, this.bottoms.length)); }
}

/** Reading order of finished cells: top to bottom in bands half a tile high, left to right inside a band. */
export function readingOrder(boxes, size) {
  const band = Math.max(1, size / 2);
  return boxes.map((_, n) => n).sort((a, b) => Math.round(boxes[a].y / band) - Math.round(boxes[b].y / band) || boxes[a].x - boxes[b].x || a - b);
}

/** Whole layout at once. Boxes are inset by gap/2 so the outer edge ends flush. */
export function pack(ratios, width, size, gap) {
  const packer = new ColumnPacker(width + gap, size, ratios);
  const cells = ratios.map((r) => packer.place(r));
  const boxes = cells.map((c) => inset(c, gap));
  return { boxes, order: readingOrder(boxes, size), height: Math.max(0, packer.height - gap) };
}

export const inset = (c, gap) => ({ x: c.x, y: c.y, w: Math.max(1, c.w - gap), h: Math.max(1, c.h - gap) });
