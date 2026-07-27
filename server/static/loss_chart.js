/* ---------------------------------------------------------------------------
   LossChart -- symmetric-log scale (linear center, logarithmic top/bottom
   25% for extreme values), so a loss curve with occasional spikes stays
   readable without the everyday range getting squashed to a flat line.

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

    this.points = []; // {step, loss, smoothed}
    this.dpr = window.devicePixelRatio || 1;
    this.margin = { top: 24, right: 20, bottom: 40, left: 80 };
    this.lastMouse = null;
    this.hover = null;
    this._animFrame = null;

    this._onMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      this.lastMouse = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      this._requestDraw();
    };
    this._onMouseLeave = () => { this.lastMouse = null; this.hover = null; this._requestDraw(); };
    canvas.addEventListener("mousemove", this._onMouseMove);
    canvas.addEventListener("mouseleave", this._onMouseLeave);

    this._resizeObserver = new ResizeObserver(() => this._requestDraw());
    this._resizeObserver.observe(canvas);

    this._draw();
  }

  addPoint(step, loss) {
    const recent = this.points.slice(-this.smoothWindow).map(p => p.loss);
    recent.push(loss);
    const smoothed = this.points.length + 1 >= this.smoothWindow
      ? recent.reduce((a, b) => a + b, 0) / recent.length
      : null;
    this.points.push({ step, loss, smoothed });
    if (this.points.length > this.maxPoints) this.points.shift();
    this._requestDraw();
  }

  reset() {
    this.points = [];
    this.hover = null;
    this.lastMouse = null;
    this._requestDraw();
  }

  destroy() {
    this.canvas.removeEventListener("mousemove", this._onMouseMove);
    this.canvas.removeEventListener("mouseleave", this._onMouseLeave);
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
    const smoothVals = this.points.map(p => p.smoothed).filter(v => v != null && v > 0);
    const lossVals = this.points.map(p => p.loss).filter(v => v != null && v > 0);

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
    const xMin = this.points[0].step, xMax = this.points[this.points.length - 1].step;
    const xPos = (step) => xMax === xMin ? plotL + plotW / 2 : plotL + ((step - xMin) / (xMax - xMin)) * plotW;
    const yPos = (v) => this._symMap(v, range, plotT, plotH, plotB);

    // grid + y labels
    ctx.strokeStyle = "#2a2d3a";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#888899";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const ticks = [range.fullMax, range.linMax, range.linMin, range.fullMin]
      .filter((v, i, arr) => isFinite(v) && arr.findIndex(o => Math.abs(o - v) <= Math.abs(v) * 1e-6 + 1e-12) === i);
    for (const tick of ticks) {
      const py = yPos(tick);
      if (py >= plotT - 0.5 && py <= plotB + 0.5) {
        ctx.beginPath(); ctx.moveTo(plotL, py); ctx.lineTo(plotR, py); ctx.stroke();
        ctx.fillText(LossChart._formatAxisLabel(tick), plotL - 6, py);
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

    // raw loss dots
    ctx.fillStyle = "rgba(108,140,255,0.6)";
    for (const p of this.points) {
      const dx = xPos(p.step), dy = yPos(p.loss);
      if (dy >= plotT - 3 && dy <= plotB + 3) {
        ctx.beginPath(); ctx.arc(dx, dy, 1.5, 0, Math.PI * 2); ctx.fill();
      }
    }

    // smoothed line
    ctx.strokeStyle = "#4caf50"; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    for (const p of this.points) {
      if (p.smoothed == null || p.smoothed <= 0) { started = false; continue; }
      const ax = xPos(p.step), ay = yPos(p.smoothed);
      if (!started) { ctx.moveTo(ax, ay); started = true; } else { ctx.lineTo(ax, ay); }
    }
    ctx.stroke();

    // legend
    ctx.fillStyle = "#6c8cff"; ctx.beginPath(); ctx.arc(plotL + 6, plotT - 8, 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#888899"; ctx.font = "11px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText("Loss", plotL + 14, plotT - 8);
    ctx.strokeStyle = "#4caf50"; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(plotL, plotT + 8); ctx.lineTo(plotL + 12, plotT + 8); ctx.stroke();
    ctx.fillText("Smoothed", plotL + 16, plotT + 8);

    this._drawTooltip(W, H, plotT, plotB, plotL, plotR, xPos, yPos);
  }

  _drawTooltip(W, H, plotT, plotB, plotL, plotR, xPos, yPos) {
    if (!this.lastMouse || this.lastMouse.x < plotL || this.lastMouse.x > plotR) { this.hover = null; return; }
    let best = null, bestDist = Infinity;
    for (const p of this.points) {
      const dist = Math.abs(xPos(p.step) - this.lastMouse.x);
      if (dist < bestDist) { bestDist = dist; best = p; }
    }
    if (!best || bestDist >= 30) { this.hover = null; return; }
    this.hover = best;

    const { ctx } = this;
    const px = xPos(best.step), py = yPos(best.loss);
    const lines = [`Step: ${best.step}`, `Loss: ${LossChart._formatNum(best.loss)}`];
    if (best.smoothed != null) lines.push(`Smooth: ${LossChart._formatNum(best.smoothed)}`);

    ctx.font = "11px monospace";
    const tw = Math.max(...lines.map(l => ctx.measureText(l).width));
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
    lines.forEach((l, i) => ctx.fillText(l, tx + 8, ty + 6 + i * 16));

    ctx.strokeStyle = "rgba(200,200,200,0.3)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(px, plotT); ctx.lineTo(px, plotB); ctx.stroke(); ctx.setLineDash([]);

    ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#6c8cff"; ctx.lineWidth = 1.5; ctx.stroke();
  }
}
