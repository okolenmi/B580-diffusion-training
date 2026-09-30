/* ---------------------------------------------------------------------------
   LossChart -- symmetric-log scale (linear center, logarithmic top/bottom
   25% for extreme values), so a loss curve with occasional spikes stays
   readable without the everyday range getting squashed to a flat line.

   options.scale picks the y mapping: "symlog" (default, the paragraph
   above) or "linear" -- a plain proportional mapping from the visible
   data's padded min/max, with nice-step gridlines. Symlog is wrong for
   magnitude data with no spike problem: VRAM usage living in an 8-9 GB
   band would be squeezed into the center 50% of the plot and stretched
   non-proportionally through the log wings above/below it, when what the
   reader needs is "this bump is twice that one". The monitor dashboard
   therefore builds its VRAM/residency and phase-timing charts with
   {scale: "linear"} and only the loss chart on the default.

   Multi-series: options.series = [{key, label, color}] defines the lines
   (default: a single "loss" series, so a canvas constructed without
   options draws exactly what it used to). addPoint takes a values object
   keyed by series key; a series absent from the object (or explicitly
   null) has a *gap* at that step -- no raw dot, line broken -- because a
   per-timestep-bucket loss genuinely has no value for a step whose
   batches never sampled that bucket, and a fabricated flat segment would
   lie about that. point.loss / point.smoothed keep tracking the primary
   (first) series for callers that predate multi-series (the monitor
   dashboard's metric readout reads point.smoothed directly).

   Two interaction features, both generic (nothing about loss specifically):

   - Legend click toggles a series hidden: hidden series are excluded from
     the line/dots/tooltip/range (so one spike can't keep compressing an
     axis for a series the reader turned off) and drawn dimmed with a
     strikethrough in the legend itself, so the state is visible where the
     control is. Stored as `s.hidden` on the series object itself -- the
     series list is the caller's data, and a toggle that mutated a copy
     would silently stop persisting across draws.
   - options.referenceLines (or setReferenceLines()) draws fixed dashed
     horizontal lines with labels -- e.g. a VRAM budget ceiling -- and
     joins their values to the axis bounds so an out-of-span reference
     lands at its true position instead of clamping to the plot edge.
   - setViewRange({min, max} in step units, or null for all data)
     windows the x-axis. Points outside the window are clipped at the
     plot border, and both the y-range (_computeRangeCore) and the
     tooltip only consider in-window points -- the axis follows what's
     on screen, so an out-of-window spike can't keep compressing a
     panned-back view. The monitor dashboard's freeze / visible-count /
     history-slider controls drive this.
   - options.trend draws, per visible series, one straight start-to-
     finish trend line: from the average of the first few present
     values in the window to the average of the last few (gaps
     skipped), and appends the movement -- end avg - begin avg -- to
     that series' legend entry. The first and last raw points are
     noise and say nothing on their own; the averaged straight line
     through the middle of the noise is the honest "is this training
     getting better or worse" statement. Legend color follows the loss
     convention: green when the series moved down, red when it moved
     up. Off by default; the monitor dashboard turns it on for the
     loss chart only (a budget ceiling or a phase time has no
     "progress" direction to report).

   This is an instantiable class, not the page-level singleton
   window.ChartManager the original dashboard tab (chart.js) uses -- same
   scale math (ported, not reinvented; it's genuinely good), rebuilt as a
   class so a page can own more than one, hand it any canvas element, and
   tear it down cleanly. chart.js itself is untouched; the main dashboard
   tab keeps working exactly as it did.
   --------------------------------------------------------------------------- */

class LossChart {
  constructor(canvas, options) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.smoothWindow = (options && options.smoothWindow) || 24;
    this.maxPoints = (options && options.maxPoints) || 1000;
    this.series = (options && options.series && options.series.length)
      ? options.series
      : [{ key: "loss", label: "Loss", color: "#6c8cff" }];
    this.primaryKey = this.series[0].key;
    this.scale = (options && options.scale === "linear") ? "linear" : "symlog";
    this.trend = !!(options && options.trend);
    this.referenceLines = (options && options.referenceLines) || [];
    this.viewRange = null; // {min, max} step window or null = all recorded data
    this._trends = {}; // rebuilt per draw when trend is on: key -> _trendFor result
    this._legendHits = []; // rebuilt per draw: legend label boxes -> series, for click-to-hide

    this.points = []; // {step, values: {key: v|null}, s: {key: smoothed|null}, loss, smoothed}
    this.dpr = window.devicePixelRatio || 1;
    this.margin = { top: 24, right: 20, bottom: 40, left: 80 };
    this.lastMouse = null;
    this.hover = null;
    this._animFrame = null;

    this._onMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      this.lastMouse = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      canvas.style.cursor = this._legendHitAt(this.lastMouse) ? "pointer" : "";
      this._requestDraw();
    };
    this._onMouseLeave = () => { this.lastMouse = null; this.hover = null; canvas.style.cursor = ""; this._requestDraw(); };
    this._onClick = (e) => {
      const rect = canvas.getBoundingClientRect();
      const s = this._legendHitAt({ x: e.clientX - rect.left, y: e.clientY - rect.top });
      if (!s) return;
      s.hidden = !s.hidden;
      this._requestDraw();
    };
    canvas.addEventListener("mousemove", this._onMouseMove);
    canvas.addEventListener("mouseleave", this._onMouseLeave);
    canvas.addEventListener("click", this._onClick);

    this._resizeObserver = new ResizeObserver(() => this._requestDraw());
    this._resizeObserver.observe(canvas);

    this._draw();
  }

  addPoint(step, values) {
    // Legacy/number form: addPoint(step, 0.13) == single primary series.
    if (values === null || typeof values !== "object") values = { [this.primaryKey]: values };
    const point = { step, values: {}, s: {} };
    for (const s of this.series) {
      const v = values[s.key];
      point.values[s.key] = v === undefined ? null : v;
      point.s[s.key] = this._smoothedFor(s.key, point.values[s.key]);
    }
    // Primary-series compat fields (pre-multi-series callers/readouts).
    point.loss = point.values[this.primaryKey];
    point.smoothed = point.s[this.primaryKey];
    this.points.push(point);
    if (this.points.length > this.maxPoints) this.points.shift();
    this._requestDraw();
  }

  /* Running mean over the last smoothWindow *present* values of this
     series (gaps skipped, not counted as 0), null until the window has
     enough -- same fill rule the single-series version used, applied per
     series so one sparse series can't shorten another's smoothing. */
  _smoothedFor(key, value) {
    if (value === null || value === undefined) return null;
    const recent = [];
    for (let i = this.points.length - 1; i >= 0 && recent.length < this.smoothWindow - 1; i--) {
      const prev = this.points[i].values[key];
      if (prev !== null && prev !== undefined) recent.push(prev);
    }
    recent.push(value);
    const filled = this._filledCount(key) + 1;
    if (filled < this.smoothWindow) return null;
    return recent.reduce((a, b) => a + b, 0) / recent.length;
  }

  _filledCount(key) {
    let n = 0;
    for (let i = this.points.length - 1; i >= 0; i--) {
      const v = this.points[i].values[key];
      if (v !== null && v !== undefined) n++;
    }
    return n;
  }

  reset() {
    this.points = [];
    this.hover = null;
    this.lastMouse = null;
    this._requestDraw();
  }

  setReferenceLines(lines) {
    this.referenceLines = lines || [];
    this._requestDraw();
  }

  /* Window the x-axis over a step range. null/invalid restores "all data".
     A no-op when the range didn't change, so a frozen (locked-window) view
     doesn't redraw identical pixels on every incoming report. */
  setViewRange(range) {
    const min = range && isFinite(range.min) ? range.min : null;
    const max = range && isFinite(range.max) ? range.max : null;
    const next = (min != null && max != null) ? { min, max } : null;
    const cur = this.viewRange;
    if (cur && next && cur.min === next.min && cur.max === next.max) return;
    if (!cur && !next) return;
    this.viewRange = next;
    this._requestDraw();
  }

  /* Start-to-finish trend for one series over the visible window
     (viewRange applies, hidden series are the caller's job to skip):
     the average of the first k present values (begin), the average of
     the last k (end), and movement = end - begin. Gaps are skipped,
     never zero-filled -- a bucket that sampled nothing at a step has no
     opinion at that step.

     k = max(3, n/4) capped at 30 present values: short runs still
     average over the noise, long runs don't average the whole history
     into one flat number. Needs >= 6 values -- below that the first
     and last windows would overlap and the "movement" would be an
     artifact of that overlap rather than a direction, so it returns
     null and the chart draws nothing (no line, no legend number)
     instead of inventing one. */
  _trendFor(key) {
    const vr = this.viewRange;
    const vals = [];
    let x0 = null, x1 = null;
    for (const p of this.points) {
      if (vr && (p.step < vr.min || p.step > vr.max)) continue;
      const v = p.values[key];
      if (v != null && isFinite(v)) { vals.push(v); x1 = p.step; if (x0 === null) x0 = p.step; }
    }
    const n = vals.length;
    if (n < 6) return null;
    const k = Math.max(3, Math.min(30, Math.floor(n / 4)));
    let begin = 0, end = 0;
    for (let i = 0; i < k; i++) begin += vals[i];
    for (let i = n - k; i < n; i++) end += vals[i];
    begin /= k; end /= k;
    // x0/x1 bracket every present value, so a bucket whose last sample
    // predates the window's end stops there instead of being extrapolated
    // to the plot border.
    return { begin, end, movement: end - begin, count: n, window: k, x0, x1 };
  }

  _legendHitAt(pt) {
    for (const h of this._legendHits) {
      if (pt.x >= h.x0 && pt.x <= h.x1 && pt.y >= h.y0 && pt.y <= h.y1) return h.series;
    }
    return null;
  }

  destroy() {
    this.canvas.removeEventListener("mousemove", this._onMouseMove);
    this.canvas.removeEventListener("mouseleave", this._onMouseLeave);
    this.canvas.removeEventListener("click", this._onClick);
    this._resizeObserver.disconnect();
    if (this._animFrame) cancelAnimationFrame(this._animFrame);
  }

  _requestDraw() {
    if (this._animFrame) return;
    this._animFrame = requestAnimationFrame(() => { this._draw(); this._animFrame = null; });
  }

  // ---- symmetric-log scale: linear across the middle 50% of plot height,
  // logarithmic across the top/bottom 25% each, so a handful of outlier
  // values don't compress the everyday range into a flat line. ----

  _computeRange() {
    const r = this._computeRangeCore();
    // Reference-line values (e.g. a VRAM budget ceiling) join the axis
    // bounds: a ceiling just above every observed number must land at its
    // true position, not clamp to the plot edge where it would read as
    // "equal to the largest sample".
    for (const rl of this.referenceLines) {
      const v = rl.value;
      if (v == null || !isFinite(v) || v <= 0) continue;
      if (v > r.fullMax) r.fullMax = v * 1.05;
      if (v < r.fullMin) r.fullMin = v * 0.95;
    }
    return r;
  }

  _computeRangeCore() {
    const vr = this.viewRange;
    // Linear scale: padded min/max of the visible values (raw + smoothed,
    // hidden series and out-of-window points excluded, same rules as
    // below). fullMin/fullMax start at the padded bounds because that's
    // what _symMap's linear branch maps from -- and _computeRange() may
    // widen them for a reference line (the VRAM budget ceiling), which
    // must land inside the plot at its true proportional position.
    if (this.scale === "linear") {
      let vMin = Infinity, vMax = -Infinity, n = 0;
      for (const s of this.series) {
        if (s.hidden) continue;
        for (const p of this.points) {
          if (vr && (p.step < vr.min || p.step > vr.max)) continue;
          const rv = p.values[s.key];
          if (rv != null && isFinite(rv)) { n++; if (rv < vMin) vMin = rv; if (rv > vMax) vMax = rv; }
          const sv = p.s[s.key];
          if (sv != null && isFinite(sv)) { n++; if (sv < vMin) vMin = sv; if (sv > vMax) vMax = sv; }
        }
      }
      if (!n) return { linMin: 0, linMax: 1, fullMin: 0, fullMax: 1 };
      let span = vMax - vMin;
      if (span <= 0) span = Math.abs(vMax) * 0.05 || 1; // flat series: a little headroom, not zero
      const lo = vMin - span * 0.1, hi = vMax + span * 0.1;
      return { linMin: lo, linMax: hi, fullMin: lo, fullMax: hi };
    }
    // Range spans *every* visible series (raw + smoothed), so a high-t bucket
    // spike and the total loss share one honest axis instead of one series
    // clipping outside the computed bounds. Hidden (legend-toggled-off)
    // series don't participate, and neither do points outside an active
    // view window: the axis should follow what's on screen.
    const smoothVals = [], lossVals = [];
    for (const s of this.series) {
      if (s.hidden) continue;
      for (const p of this.points) {
        if (vr && (p.step < vr.min || p.step > vr.max)) continue;
        const sv = p.s[s.key];
        if (sv != null && sv > 0) smoothVals.push(sv);
        const rv = p.values[s.key];
        if (rv != null && rv > 0) lossVals.push(rv);
      }
    }

    if (smoothVals.length >= 2) {
      const sMin = Math.min(...smoothVals), sMax = Math.max(...smoothVals);
      let range = sMax - sMin;
      if (range <= 0) range = sMin * 0.2;
      let linMin = sMin - range * 0.1;
      let linMax = sMax + range * 0.1;
      if (linMin <= 0) linMin = sMin * 0.5;
      let fullMin = linMin * 0.5, fullMax = linMax * 2.0;
      if (lossVals.length > 0) {
        const lMax = Math.max(...lossVals), lMin = Math.min(...lossVals);
        if (lMax > fullMax) fullMax = lMax * 1.1;
        if (lMin < fullMin) fullMin = lMin * 0.5;
      }
      return { linMin, linMax, fullMin, fullMax };
    }
    if (smoothVals.length === 1) {
      const v = smoothVals[0];
      return { linMin: v * 0.9, linMax: v * 1.1, fullMin: v * 0.2, fullMax: v * 5 };
    }
    if (lossVals.length > 0) {
      const fullMax = Math.max(...lossVals) * 1.2;
      return { linMin: fullMax * 0.01, linMax: fullMax * 0.99, fullMin: 0.0001, fullMax };
    }
    return { linMin: 0.1, linMax: 0.9, fullMin: 0.01, fullMax: 1 };
  }

  _symMap(value, range, plotT, plotH, plotB) {
    // Linear: one proportional mapping over the whole plot (full bounds,
    // so a reference line that widened the range positions correctly).
    if (this.scale === "linear") {
      const lo = range.fullMin, hi = range.fullMax;
      if (!(hi > lo) || !isFinite(value)) return plotT + plotH / 2;
      const frac = Math.max(0, Math.min(1, (value - lo) / (hi - lo)));
      return plotT + plotH * (1 - frac);
    }
    let { linMin, linMax, fullMin, fullMax } = range;
    if (fullMax <= linMax) fullMax = linMax * 2;
    if (fullMin >= linMin || fullMin <= 0) fullMin = linMin * 0.5;
    if (linMax <= linMin) linMax = linMin * 1.01;

    if (value >= linMax) {
      const topH = plotH * 0.25;
      const t = Math.max(0, Math.min(1, Math.log(value / linMax) / Math.log(fullMax / linMax)));
      return plotT + topH * (1 - t);
    }
    if (value <= linMin) {
      const botH = plotH * 0.25;
      const t = Math.max(0, Math.min(1, Math.log(value / linMin) / Math.log(fullMin / linMin)));
      return (plotB - botH) + botH * t;
    }
    const centerH = plotH * 0.5;
    const frac = Math.max(0, Math.min(1, (value - linMin) / (linMax - linMin)));
    return plotT + plotH * 0.25 + centerH * (1 - frac);
  }

  static _formatNum(v) {
    if (v >= 1000) return v.toFixed(1); // MB/ms magnitudes: 9123.4, not 9123.4568
    if (v >= 1) return v.toFixed(4);
    if (v >= 0.01) return v.toFixed(5);
    return v.toExponential(2);
  }

  static _formatAxisLabel(v) {
    if (!isFinite(v) || v === 0) return "0";
    const av = Math.abs(v);
    const rounded = Number(av.toPrecision(2));
    const s = (rounded >= 1000 || rounded < 0.0001) ? rounded.toExponential(1) : String(rounded);
    return v < 0 ? "-" + s : s;
  }

  /* Evenly spaced ticks over [min, max] at a nice step (~6 of them).
     Only used by the linear scale: nice numbers are already clean (50,
     200, 500), so their labels print exactly -- _formatAxisLabel's 2-sig
     digit rounding would turn a 8850 tick into "8900" and collide with
     the next one. */
  static _linearTicks(min, max) {
    if (!isFinite(min) || !isFinite(max) || !(max > min)) return isFinite(max) ? [max] : [];
    const step = LossChart._niceNum((max - min) / 6, true);
    if (!(step > 0)) return [min, max];
    const out = [];
    for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-6; t += step) {
      out.push(Math.round(t * 1e9) / 1e9); // undo float drift so labels stay exact
    }
    return out;
  }

  static _formatLinearTick(v) {
    if (!isFinite(v)) return "";
    const r = Math.round(v);
    if (Math.abs(v - r) < 1e-9) return String(r);
    return String(Math.round(v * 100) / 100);
  }

  /* Movement = end average - begin average: signed, 2 significant
     digits. This is the number that says progress (negative, loss
     coming down) or regress at a glance, without reading the squiggle. */
  static _formatMovement(v) {
    if (v == null || !isFinite(v)) return "";
    return (v < 0 ? "\u2212" : "+") + Math.abs(v).toPrecision(2);
  }

  static _niceNum(range, round) {
    if (range <= 0) return 1;
    const exp = Math.floor(Math.log10(range));
    const frac = range / Math.pow(10, exp);
    let nice;
    if (round) nice = (frac < 1.5) ? 1 : (frac < 3) ? 2 : (frac < 7) ? 5 : 10;
    else nice = (frac <= 1) ? 1 : (frac <= 2) ? 2 : (frac <= 5) ? 5 : 10;
    return nice * Math.pow(10, exp);
  }

  _draw() {
    const { canvas, ctx } = this;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    if (W <= 0 || H <= 0) return;

    this.dpr = window.devicePixelRatio || 1;
    canvas.width = W * this.dpr;
    canvas.height = H * this.dpr;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const m = this.margin;
    const plotL = m.left, plotR = W - m.right, plotT = m.top, plotB = H - m.bottom;
    const plotW = plotR - plotL, plotH = plotB - plotT;

    if (this.points.length === 0) {
      ctx.fillStyle = "#888899";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for data\u2026", W / 2, H / 2);
      return;
    }

    const range = this._computeRange();
    const vr = this.viewRange;
    const xMin = vr ? vr.min : this.points[0].step;
    const xMax = vr ? vr.max : this.points[this.points.length - 1].step;
    const xPos = (step) => xMax === xMin ? plotL + plotW / 2 : plotL + ((step - xMin) / (xMax - xMin)) * plotW;
    const yPos = (v) => this._symMap(v, range, plotT, plotH, plotB);

    // grid + y labels: linear gets nice-step gridlines across the whole
    // range (a proportional axis wants proportional, evenly spaced rules);
    // symlog keeps its four boundary ticks (the scale's landmarks).
    ctx.strokeStyle = "#2a2d3a";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#888899";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const linear = this.scale === "linear";
    const ticks = linear
      ? LossChart._linearTicks(range.fullMin, range.fullMax)
      : [range.fullMax, range.linMax, range.linMin, range.fullMin]
          .filter((v, i, arr) => isFinite(v) && arr.findIndex(o => Math.abs(o - v) <= Math.abs(v) * 1e-6 + 1e-12) === i);
    for (const tick of ticks) {
      const py = yPos(tick);
      if (py >= plotT - 0.5 && py <= plotB + 0.5) {
        ctx.beginPath(); ctx.moveTo(plotL, py); ctx.lineTo(plotR, py); ctx.stroke();
        ctx.fillText(linear ? LossChart._formatLinearTick(tick) : LossChart._formatAxisLabel(tick), plotL - 6, py);
      }
    }

    // x axis
    ctx.beginPath(); ctx.moveTo(plotL, plotB); ctx.lineTo(plotR, plotB); ctx.stroke();
    let xStep = LossChart._niceNum((xMax - xMin) / 8, true);
    if (xStep <= 0) xStep = 1;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let xs = Math.ceil(xMin / xStep) * xStep; xs <= xMax + xStep * 0.01; xs += xStep) {
      const px = xPos(xs);
      if (px >= plotL && px <= plotR) ctx.fillText(Math.round(xs).toString(), px, plotB + 6);
    }

    // Clip all data drawing to the plot area: with a view window active,
    // out-of-window points map left/right of the plot and an unclipped dot
    // would bleed over the axis labels, while a line segment crossing the
    // window edge gets cut cleanly at the border instead of running to the
    // original point. Legend, axes and tooltip stay outside the clip.
    ctx.save();
    ctx.beginPath();
    ctx.rect(plotL, plotT, plotW, plotH);
    ctx.clip();

    // raw loss dots -- one pass per series, in the series' own color; a
    // series with no value at this step simply has no dot (a gap, not a 0).
    ctx.save();
    ctx.globalAlpha = 0.6;
    for (const s of this.series) {
      if (s.hidden) continue;
      ctx.fillStyle = s.color;
      for (const p of this.points) {
        const v = p.values[s.key];
        if (v == null) continue;
        const dx = xPos(p.step), dy = yPos(v);
        if (dy >= plotT - 3 && dy <= plotB + 3) {
          ctx.beginPath(); ctx.arc(dx, dy, 1.5, 0, Math.PI * 2); ctx.fill();
        }
      }
    }
    ctx.restore();

    // smoothed line -- per series, its own color; null smoothed breaks the
    // line (started=false) so gaps stay gaps.
    ctx.lineWidth = 2;
    for (const s of this.series) {
      if (s.hidden) continue;
      ctx.strokeStyle = s.color;
      ctx.beginPath();
      let started = false;
      for (const p of this.points) {
        const sv = p.s[s.key];
        if (sv == null || sv <= 0) { started = false; continue; }
        const ax = xPos(p.step), ay = yPos(sv);
        if (!started) { ctx.moveTo(ax, ay); started = true; } else { ctx.lineTo(ax, ay); }
      }
      ctx.stroke();
    }

    // trend lines (options.trend): one straight begin-avg -> end-avg line
    // per visible series, dashed in the series' own color so it reads as
    // "the direction of this line", not another measurement. Recomputed
    // each draw (view window, hidden toggles) and reused by the legend
    // below, which prints the movement. Still inside the clip: a windowed
    // view maps the line's endpoints to that window's own first/last
    // present steps.
    if (this.trend) {
      this._trends = {};
      for (const s of this.series) {
        if (s.hidden) continue;
        this._trends[s.key] = this._trendFor(s.key);
      }
      ctx.save();
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.9;
      ctx.setLineDash([7, 5]);
      for (const s of this.series) {
        const tr = this._trends[s.key];
        if (!tr) continue;
        ctx.strokeStyle = s.color;
        ctx.beginPath();
        ctx.moveTo(xPos(tr.x0), yPos(tr.begin));
        ctx.lineTo(xPos(tr.x1), yPos(tr.end));
        ctx.stroke();
      }
      ctx.restore();
    }
    ctx.restore();

    // reference lines (a fixed ceiling/target, e.g. vram_budget_mb): a dashed
    // horizontal rule with its label riding the right edge. Drawn after the
    // data so the ceiling stays visible over a line that reaches it, and
    // only when it falls inside the computed range (the range joins
    // reference values in _computeRange, so this is a genuine offscreen
    // guard, not a routine skip).
    if (this.referenceLines.length) {
      ctx.save();
      ctx.lineWidth = 1;
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "bottom";
      for (const rl of this.referenceLines) {
        if (rl.value == null || !isFinite(rl.value)) continue;
        const py = yPos(rl.value);
        if (!(py >= plotT && py <= plotB)) continue;
        ctx.strokeStyle = rl.color || "#ff5252";
        ctx.setLineDash([6, 4]);
        ctx.beginPath(); ctx.moveTo(plotL, py); ctx.lineTo(plotR, py); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = rl.color || "#ff5252";
        ctx.fillText(rl.label || "", plotR - 4, py - 2);
      }
      ctx.restore();
    }

    // legend -- one colored line-swatch + label per series, laid out left
    // to right across the top margin (wraps to a second row if the labels
    // don't fit: 4 series fit one row on any reasonable canvas, but a
    // narrower embedded canvas shouldn't silently overlap). Each label's
    // box is recorded so a click toggles that series hidden; a hidden
    // series draws dimmed with a strikethrough so the toggle's state is
    // visible right where the control is.
    ctx.font = "11px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    this._legendHits = [];
    let lx = plotL + 4, ly = plotT - 8, row = 0;
    for (const s of this.series) {
      const tr = this.trend ? this._trends[s.key] : null;
      const deltaTxt = tr ? LossChart._formatMovement(tr.movement) : "";
      const deltaW = deltaTxt ? ctx.measureText(deltaTxt).width + 6 : 0;
      const labelW = ctx.measureText(s.label).width;
      if (lx + 16 + labelW + deltaW > plotR && row === 0) { lx = plotL + 4; ly = plotT + 8; row = 1; }
      ctx.save();
      if (s.hidden) ctx.globalAlpha = 0.4;
      ctx.strokeStyle = s.color; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 12, ly); ctx.stroke();
      ctx.fillStyle = "#888899";
      ctx.fillText(s.label, lx + 16, ly);
      if (s.hidden) {
        ctx.strokeStyle = "#888899"; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(lx + 16, ly); ctx.lineTo(lx + 16 + labelW, ly); ctx.stroke();
      } else if (deltaTxt) {
        // movement: green when the series came down (progress on a loss
        // chart), red when it went up, gray when it didn't move -- flat is
        // not good, it's nothing. The sign is also in the text.
        ctx.fillStyle = tr.movement < 0 ? "#4caf50" : tr.movement > 0 ? "#ff5252" : "#888899";
        ctx.fillText(deltaTxt, lx + 16 + labelW + 6, ly);
      }
      ctx.restore();
      this._legendHits.push({ series: s, x0: lx - 4, x1: lx + 20 + labelW + deltaW, y0: ly - 8, y1: ly + 8 });
      lx += 16 + labelW + deltaW + 14;
    }

    this._drawTooltip(W, H, plotT, plotB, plotL, plotR, xPos, yPos);
  }

  _drawTooltip(W, H, plotT, plotB, plotL, plotR, xPos, yPos) {
    if (!this.lastMouse || this.lastMouse.x < plotL || this.lastMouse.x > plotR) { this.hover = null; return; }
    let best = null, bestDist = Infinity;
    for (const p of this.points) {
      if (this.viewRange && (p.step < this.viewRange.min || p.step > this.viewRange.max)) continue;
      const dist = Math.abs(xPos(p.step) - this.lastMouse.x);
      if (dist < bestDist) { bestDist = dist; best = p; }
    }
    if (!best || bestDist >= 30) { this.hover = null; return; }
    this.hover = best;

    const { ctx } = this;
    const primaryV = best.values[this.primaryKey];
    const px = xPos(best.step);
    const py = primaryV != null ? yPos(primaryV) : (plotT + plotB) / 2;
    // One line per series present at this step, each in its series color:
    // label + raw value (smoothed in parens when available). A bucket
    // series with no value this step is omitted -- the chart's gap, stated
    // in text.
    const lines = [{ text: `Step: ${best.step}`, color: null }];
    for (const s of this.series) {
      if (s.hidden) continue;
      const v = best.values[s.key];
      if (v == null) continue;
      let text = `${s.label}: ${LossChart._formatNum(v)}`;
      const sv = best.s[s.key];
      if (sv != null) text += ` (${LossChart._formatNum(sv)})`;
      lines.push({ text, color: s.color });
    }

    ctx.font = "11px monospace";
    const tw = Math.max(...lines.map(l => ctx.measureText(l.text).width));
    const th = lines.length * 16 + 10;
    let tx = px + 12, ty = py - th / 2;
    if (tx + tw + 16 > W) tx = px - tw - 20;
    if (ty < 0) ty = 4;
    if (ty + th > H) ty = H - th - 4;

    ctx.fillStyle = "rgba(20,20,40,0.92)";
    ctx.strokeStyle = "#555"; ctx.lineWidth = 1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(tx, ty, tw + 16, th, 6); else ctx.rect(tx, ty, tw + 16, th);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = "#ccc"; ctx.textAlign = "left"; ctx.textBaseline = "top";
    lines.forEach((l, i) => {
      const rowY = ty + 6 + i * 16;
      if (l.color) {
        ctx.fillStyle = l.color;
        ctx.fillRect(tx + 8, rowY + 2, 6, 7);
        ctx.fillStyle = "#ccc";
        ctx.fillText(l.text, tx + 18, rowY);
      } else {
        ctx.fillText(l.text, tx + 8, rowY);
      }
    });

    ctx.strokeStyle = "rgba(200,200,200,0.3)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(px, plotT); ctx.lineTo(px, plotB); ctx.stroke(); ctx.setLineDash([]);

    if (primaryV != null && !this.series[0].hidden) {
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = this.series[0].color; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }
}

export { LossChart };
