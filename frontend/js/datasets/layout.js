/* datasets/layout.js -- justified rows: every picture keeps its own proportions, and each row is scaled so that it
   fills the width exactly (no crop, no ragged edge). Pure arithmetic, so it is testable without a browser. */

/**
 * @param {number[]} ratios  width / height of each picture, in the order shown
 * @param {number} width     the space a row must fill, in px
 * @param {number} target    the row height to aim for, in px
 * @param {number} gap       space between pictures, in px
 * @returns {{start: number, count: number, height: number}[]}
 */
export function justify(ratios, width, target, gap) {
  const heightOf = (from, to) => {
    let sum = 0;
    for (let i = from; i < to; i++) sum += ratios[i];
    return (width - gap * (to - from - 1)) / sum;
  };
  const rows = [];
  let start = 0;
  for (let end = 1; end <= ratios.length; end++) {
    const height = heightOf(start, end);
    if (height > target) continue;
    // closing the row here, or one picture earlier: whichever is nearer the target height
    const shorter = end - start > 1 ? heightOf(start, end - 1) : Infinity;
    const useShorter = shorter !== Infinity && Math.abs(shorter - target) < Math.abs(height - target);
    const stop = useShorter ? end - 1 : end;
    rows.push({ start, count: stop - start, height: useShorter ? shorter : height });
    start = stop;
    end = stop;
  }
  if (start < ratios.length) rows.push({ start, count: ratios.length - start, height: Math.min(target, heightOf(start, ratios.length)) });
  return rows;
}
