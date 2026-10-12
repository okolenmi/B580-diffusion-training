/* ---------------------------------------------------------------------------
   board/view.js -- the window a chart shows over a long run, as a pure model.

   The old Monitor page let you look at "the last 50 / 100 / 250 / 1000 /
   all" records, freeze the picture while a run kept going, and drag a slider
   back through history. The board's chart block lost all of that; this is the
   same logic, ported from monitor.js (applyView / toggleFreeze / relockAt /
   setCount / pan) and made independent of any chart: it works over `xs`, the
   ascending x value of every record the stream holds, and answers with the
   x-range to show.

   State:
     count        records shown: a number, or "all" (no window, no panning)
     live         true = the window follows the newest record
     startIdx     slider position (index of the window's first record)
     lockedRange  the concrete {min, max} while not live, so incoming
                  reports can never move a held view
     dragging /   the pointer holds the slider thumb: its geometry must not
     frozenMax    be rewritten under the finger, and "the far end" is the
                  one the thumb was grabbed with

   No DOM, no chart, no clock.
   --------------------------------------------------------------------------- */

export const COUNTS = [50, 100, 250, 1000, "all"];

export class ViewWindow {
  constructor() {
    this.count = "all";
    this.restart();
  }

  /** A new run began: follow it again. The chosen count is a preference and
      stays; a held range belonged to the old run and does not. */
  restart() {
    this.live = true;
    this.startIdx = 0;
    this.lockedRange = null;
    this.dragging = false;
    this.frozenMax = null;
  }

  /** The window for the records `xs` -- and the slider/readout facts. */
  apply(xs) {
    const total = xs.length;
    const n = this.count === "all" ? total : Math.min(this.count, total);
    let range;
    if (this.live) {
      range = (!total || this.count === "all") ? null
        : { min: xs[total - n], max: xs[total - 1] };
      this.startIdx = Math.max(0, total - n);
    } else {
      range = this.lockedRange;
    }
    const maxIdx = Math.max(0, total - n);
    const latest = total ? xs[total - 1] : null;
    return {
      range,
      n,
      maxIdx,
      startIdx: Math.min(this.startIdx, maxIdx),
      sliderDisabled: this.count === "all",
      frozen: !this.live,
      first: total ? (range ? range.min : xs[0]) : null,
      last: total ? (range ? range.max : latest) : null,
      latest,
    };
  }

  /** Lock the window where it is, or release it back to following. */
  toggleFreeze(xs) {
    if (this.live) {
      this.live = false;
      this.relockAt(xs, xs.length ? xs[xs.length - 1] : null);
    } else {
      this.live = true;
      this.lockedRange = null;
    }
  }

  /** Lock the window so it ENDS at `endX`, keeping the current count. */
  relockAt(xs, endX) {
    if (!xs.length || endX === null || endX === undefined) {
      this.lockedRange = null;
      this.startIdx = 0;
      return;
    }
    let endIdx = xs.length - 1;
    while (endIdx > 0 && xs[endIdx] > endX) endIdx--;
    if (this.count === "all") {
      this.startIdx = 0;
      this.lockedRange = { min: xs[0], max: xs[endIdx] };
      return;
    }
    const startIdx = Math.max(0, endIdx - this.count + 1);
    this.startIdx = startIdx;
    this.lockedRange = { min: xs[startIdx], max: xs[endIdx] };
  }

  /** Resizing a held window keeps its END anchored, so widening 50 -> 100
      while frozen reveals earlier records rather than jumping past the
      freeze point. */
  setCount(value, xs) {
    this.count = value;
    if (!this.live && this.lockedRange) this.relockAt(xs, this.lockedRange.max);
  }

  /** Slider moved to record index `idx`. The far end returns to live;
      anywhere else holds the window there. */
  pan(idx, xs) {
    if (this.count === "all" || !xs.length) return;
    const n = Math.min(this.count, xs.length);
    const maxIdx = Math.max(0, xs.length - n);
    const farEnd = this.dragging && this.frozenMax !== null
      ? Math.min(this.frozenMax, maxIdx)
      : maxIdx;
    const at = Math.max(0, Math.min(idx, maxIdx));
    if (at >= farEnd) {
      this.live = true;
      this.lockedRange = null;
    } else {
      this.live = false;
      this.relockAt(xs, xs[Math.min(at + n - 1, xs.length - 1)]);
    }
  }

  /** The pointer took / released the slider thumb. `maxIdx` is the slider's
      current far end, remembered while held. */
  setDragging(on, maxIdx = 0) {
    on = !!on;
    if (this.dragging === on) return;
    this.dragging = on;
    this.frozenMax = on ? maxIdx : null;
  }
}

/** The records of `rows` (each `{x, ...}`, ascending in x) a window needs
    drawn: from `margin` records before the range -- so a running mean has its
    history at the left edge instead of starting empty -- to the range's end.
    No range = everything. */
export function rowsForWindow(rows, range, margin = 0) {
  if (!range) return rows;
  let first = rows.length;
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].x >= range.min) { first = i; break; }
  }
  let end = first;
  while (end < rows.length && rows[end].x <= range.max) end++;
  return rows.slice(Math.max(0, first - margin), end);
}
