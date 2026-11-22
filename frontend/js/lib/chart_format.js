/* Number formatting, tick placement and averaging for the charts. Pure: no DOM, canvas or clock. */

export function formatNum(v) {
  if (v >= 1000) return v.toFixed(1); // MB/ms magnitudes: 9123.4, not 9123.4568
  if (v >= 1) return v.toFixed(4);
  if (v >= 0.01) return v.toFixed(5);
  return v.toExponential(2);
}

export function formatAxisLabel(v) {
  if (!isFinite(v) || v === 0) return "0";
  const av = Math.abs(v);
  const rounded = Number(av.toPrecision(2));
  const s = (rounded >= 1000 || rounded < 0.0001) ? rounded.toExponential(1) : String(rounded);
  return v < 0 ? "-" + s : s;
}

/* Evenly spaced ticks over [min, max] at a nice step (~6 of them).
   Only used by the linear scale: nice numbers are already clean (50,
   200, 500), so their labels print exactly -- formatAxisLabel's 2-sig
   digit rounding would turn a 8850 tick into "8900" and collide with
   the next one. */
export function linearTicks(min, max) {
  if (!isFinite(min) || !isFinite(max) || !(max > min)) return isFinite(max) ? [max] : [];
  const step = niceNum((max - min) / 6, true);
  if (!(step > 0)) return [min, max];
  const out = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-6; t += step) {
    out.push(Math.round(t * 1e9) / 1e9); // undo float drift so labels stay exact
  }
  return out;
}

export function formatLinearTick(v) {
  if (!isFinite(v)) return "";
  const r = Math.round(v);
  if (Math.abs(v - r) < 1e-9) return String(r);
  return String(Math.round(v * 100) / 100);
}

/* Movement = end average - begin average: signed, 2 significant
   digits. This is the number that says progress (negative, loss
   coming down) or regress at a glance, without reading the squiggle. */
export function formatMovement(v) {
  if (v == null || !isFinite(v)) return "";
  return (v < 0 ? "\u2212" : "+") + Math.abs(v).toPrecision(2);
}

export function niceNum(range, round) {
  if (range <= 0) return 1;
  const exp = Math.floor(Math.log10(range));
  const frac = range / Math.pow(10, exp);
  let nice;
  if (round) nice = (frac < 1.5) ? 1 : (frac < 3) ? 2 : (frac < 7) ? 5 : 10;
  else nice = (frac <= 1) ? 1 : (frac <= 2) ? 2 : (frac <= 5) ? 5 : 10;
  return nice * Math.pow(10, exp);
}

/** One number for a window of values: their mean, median, or exponential average (alpha = 2 / (window + 1)). */
export function averageOf(values, method, window) {
  if (method === "median") {
    const sorted = [...values].sort((a, b) => a - b), mid = sorted.length >> 1;
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }
  if (method === "ema") {
    const alpha = 2 / (window + 1);
    return values.reduce((acc, v) => acc + alpha * (v - acc));
  }
  return values.reduce((a, b) => a + b, 0) / values.length;
}
