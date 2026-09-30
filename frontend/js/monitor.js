/* ---------------------------------------------------------------------------
   Monitor page entry (M6) -- ES-module port of the legacy
   server/static/monitor_dashboard.js. The visualization is pinned by
   docs/design/backend/03-migration-strategy.md section 1 (keep what the
   legacy dashboard did): same frame handling, same readouts, same
   window/freeze/CSV controls. What changed:

   - the stream is the new backend: /api/v1/monitor/{id}/stream (frame
     contract byte-identical -- connected / raw step reports / clear /
     run_end);
   - LossChart is imported instead of loaded as a page global;
   - boot runs at module evaluation (type=module is deferred).

   Frame rules that must survive any future edit (they are the contract):

   - `clear` resets EVERYTHING -- a fresh run on the same monitor_id
     must never overlay the old run's line;
   - `run_end` is terminal state, checked before the step-shaped filter;
   - a message without `step` is ignored, never guessed at;
   - absent keys are gaps (charts) and em dashes (readouts) -- never
     fabricated zeros.
   --------------------------------------------------------------------------- */

import { LossChart } from "./lib/loss_chart.js";

// The managed/main trainers' per-t loss diagnostics (nodes/train/loss.py's
// t_bucket_losses keys) plotted alongside the total, one color per series:
// total in blue; t-low (fine detail, late denoising) green; t-mid amber;
// t-high (coarse structure, early denoising) red. Series keys the report
// lacks render as gaps, not zeros -- see LossChart's header.
const LOSS_SERIES = [
  { key: "loss", label: "loss", color: "#6c8cff" },
  { key: "loss_t_low", label: "t low", color: "#4caf50" },
  { key: "loss_t_mid", label: "t mid", color: "#ffb300" },
  { key: "loss_t_high", label: "t high", color: "#ff5252" },
];

// Allocator state + per-resident footprints. reserved is the primary
// series -- the number the budget is enforced against; peak reserved
// rides above it so a within-step spike stays visible.
const VRAM_SERIES = [
  { key: "vram_reserved_mb", label: "reserved", color: "#6c8cff" },
  { key: "vram_peak_reserved_mb", label: "peak reserved", color: "#ff9800" },
  { key: "vram_allocated_mb", label: "allocated", color: "#4caf50" },
  { key: "resident_model_mb", label: "model", color: "#26c6da" },
  { key: "resident_optimizer_mb", label: "optimizer", color: "#ab47bc" },
  { key: "resident_text_encoder_mb", label: "text encoder", color: "#8d6e63" },
];

// Phase-timing series discovered from the first report that carries any
// {label}_ms key. step_total_ms always gets the neutral tone so "sum of
// everything" reads as an envelope, not one more competing phase.
const TIMING_PALETTE = ["#6c8cff", "#4caf50", "#ffb300", "#ff5252",
                        "#ab47bc", "#26c6da", "#ff9800", "#8d6e63"];
const TIMING_TOTAL_COLOR = "#cfd8dc";

// LossChart's default cap of 1000 points would silently drop the oldest
// records -- exactly what the history slider exists to reach. 100k is
// "no truncation in practice" and matches MonitorBus's HISTORY_LIMIT.
const MAX_RECORDS = 100000;

const el = (id) => document.getElementById(id);

class MonitorDashboard {
  constructor(monitorId, els) {
    this.monitorId = monitorId;
    this.els = els;
    // Trend lines: one begin-avg -> end-avg line per series with the
    // movement in the legend. Loss chart only.
    this.chart = new LossChart(els.canvas, { series: LOSS_SERIES, maxPoints: MAX_RECORDS, trend: true });
    // VRAM/residency and phase timing are magnitude data: linear maps
    // them proportionally over the full plot (symlog would squash them).
    this.vramChart = new LossChart(els.vramCanvas, { series: VRAM_SERIES, maxPoints: MAX_RECORDS, scale: "linear" });
    // Built lazily on the first report carrying *_ms keys.
    this.timingChart = null;
    this.firstEventTime = null;
    this.lastEvent = null;
    this.recentRates = []; // {t, step} pairs, last few, for steps/sec
    this.source = null;
    this.conn = "connecting"; // SSE connection state, for the status readout
    this.runEnded = null;     // null while running; {step, cancelled} after run_end
    this.rawHistory = [];     // every step report this session saw -- the CSV export
    this.budget = null;       // vram_budget_mb, once a report states one
    this.bucketLast = {};     // loss_t_* -> {v, step}: latest measured per range
    // One view window shared by all three charts (all plot against
    // step): count = records shown, live = window follows newest,
    // startIdx = slider position, lockedRange = window captured when
    // follow stopped, dragging/frozenMax = pointer holds on the thumb.
    this.view = {
      count: "all", live: true, startIdx: 0, lockedRange: null,
      dragging: false, frozenMax: null,
    };
    this.countDefs = [
      [els.count50, 50], [els.count100, 100], [els.count250, 250],
      [els.count1000, 1000], [els.countAll, "all"],
    ];
    this.stats = { best: null, bestStep: null, peakVram: null };
  }

  connect() {
    this.source = new EventSource(
      `/api/v1/monitor/${encodeURIComponent(this.monitorId)}/stream`
    );
    this.source.onopen = () => this.setStatus("live");
    this.source.onerror = () => this.setStatus("disconnected");
    this.source.onmessage = (ev) => this.handleEvent(ev);
  }

  setStatus(state) {
    this.conn = state;
    this.renderStatus();
  }

  /* Connection state and run state combined: a finished run's stream
     stays open, so run_end is what makes "finished vs still going"
     answerable at all. */
  renderStatus() {
    const dot = this.els.statusDot, text = this.els.statusText;
    if (this.conn === "disconnected") {
      dot.className = "mon-status-dot disconnected";
      text.textContent = "disconnected \u2014 retrying\u2026";
      return;
    }
    if (this.runEnded) {
      const c = this.runEnded.cancelled;
      dot.className = "mon-status-dot " + (c ? "ended" : "live");
      text.textContent = (c ? "cancelled at step " : "finished at step ") + this.runEnded.step;
      return;
    }
    dot.className = "mon-status-dot" + (this.conn === "live" ? " live" : "");
    text.textContent = this.conn === "live" ? "live" : "connecting\u2026";
  }

  handleEvent(ev) {
    let data;
    try { data = JSON.parse(ev.data); } catch (e) { return; }
    if (data.type === "connected") { this.setStatus("live"); return; }
    if (data.type === "clear") {
      // A fresh run just started reporting to this monitor_id -- without
      // the reset, re-running against the same id would overlay the new
      // run's line on the old one with no indication they differ.
      this.chart.reset();
      this.vramChart.reset();
      if (this.timingChart) this.timingChart.reset();
      this.vramChart.setReferenceLines([]);
      this.firstEventTime = null;
      this.lastEvent = null;
      this.recentRates = [];
      this.rawHistory = [];
      this.budget = null;
      this.bucketLast = {};
      this.view.live = true;
      this.view.lockedRange = null;
      this.view.startIdx = 0;
      this.runEnded = null;
      this.stats = { best: null, bestStep: null, peakVram: null };
      this.resetReadouts();
      this.applyView();
      this.renderStatus();
      return;
    }
    if (data.type === "run_end") {
      // Terminal status from the trainer: run over (normal or
      // cancelled), numbers final. Carries a step but is not a step
      // report -- checked before the step-shaped filter below.
      this.runEnded = { step: data.step, cancelled: !!data.cancelled };
      this.renderStatus();
      return;
    }
    if (data.step === undefined) return; // not training-progress-shaped; ignore

    const now = data.t ? data.t * 1000 : Date.now();
    if (this.firstEventTime === null) this.firstEventTime = now;
    this.recentRates.push({ t: now, step: data.step });
    if (this.recentRates.length > 20) this.recentRates.shift();
    this.lastEvent = data;
    this.rawHistory.push(data);

    // Values object (not a bare number): series absent from this report
    // are omitted, which LossChart renders as a gap.
    const values = {};
    for (const s of LOSS_SERIES) if (data[s.key] !== undefined) values[s.key] = data[s.key];
    this.chart.addPoint(data.step, values);
    this.feedVramChart(data);
    this.feedTimingChart(data);
    this.trackStats(data);
    this.updateMetrics(data, now);
    this.applyView();
  }

  feedVramChart(data) {
    const present = VRAM_SERIES.some((s) => data[s.key] !== undefined);
    if (present) {
      this.els.vramCard.classList.remove("mon-hidden");
      const vals = {};
      for (const s of VRAM_SERIES) if (data[s.key] !== undefined) vals[s.key] = data[s.key];
      this.vramChart.addPoint(data.step, vals);
    }
    if (this.budget == null && typeof data.vram_budget_mb === "number") {
      this.budget = data.vram_budget_mb;
      this.vramChart.setReferenceLines([{
        value: this.budget,
        label: "budget " + this.fmtMB(this.budget),
        color: "#ff5252",
      }]);
    }
  }

  feedTimingChart(data) {
    const keys = Object.keys(data).filter(
      (k) => k.endsWith("_ms") && k !== "step_total_ms");
    // No timing in this report -> nothing to show, and crucially no
    // all-gap point around an empty (cleared) chart.
    if (keys.length === 0) return;
    if (this.timingChart === null) this.buildTimingChart(keys);
    else this.extendTimingChart(keys);
    this.els.timingCard.classList.remove("mon-hidden");
    const vals = {};
    for (const s of this.timingChart.series) {
      if (data[s.key] !== undefined) vals[s.key] = data[s.key];
    }
    this.timingChart.addPoint(data.step, vals);
  }

  buildTimingChart(keys) {
    const series = keys.map((k, i) => ({
      key: k,
      label: k.slice(0, -3), // strip "_ms" -- the phase label itself
      color: TIMING_PALETTE[i % TIMING_PALETTE.length],
    }));
    series.push({ key: "step_total_ms", label: "total", color: TIMING_TOTAL_COLOR });
    this.timingChart = new LossChart(this.els.timingCanvas, { series, maxPoints: MAX_RECORDS, scale: "linear" });
  }

  /* A phase key absent from the first timing report still gets plotted
     rather than silently dropped: the failure mode of being wrong
     about "runs never change phase lists mid-run" is a late line
     appearing, not data vanishing. */
  extendTimingChart(keys) {
    for (const k of keys) {
      if (this.timingChart.series.some((s) => s.key === k)) continue;
      const total = this.timingChart.series.length;
      this.timingChart.series.push({
        key: k,
        label: k.slice(0, -3),
        color: TIMING_PALETTE[total % TIMING_PALETTE.length],
      });
    }
  }

  trackStats(data) {
    if (typeof data.loss === "number"
        && (this.stats.best === null || data.loss < this.stats.best)) {
      this.stats.best = data.loss;
      this.stats.bestStep = data.step;
    }
    // Peak: the explicit per-step peak when reported, else the sampled
    // reserved value (all a config without peaks can honestly claim).
    const peak = data.vram_peak_reserved_mb !== undefined
      ? data.vram_peak_reserved_mb
      : data.vram_reserved_mb;
    if (typeof peak === "number"
        && (this.stats.peakVram === null || peak > this.stats.peakVram)) {
      this.stats.peakVram = peak;
    }
  }

  updateMetrics(data, now) {
    const e = this.els;
    e.step.textContent = data.total_steps ? `${data.step} / ${data.total_steps}` : String(data.step);
    e.loss.textContent = this.fmt(data.loss);
    e.smoothed.textContent = this.chart.points.length && this.chart.points[this.chart.points.length - 1].smoothed != null
      ? this.fmt(this.chart.points[this.chart.points.length - 1].smoothed) : "\u2014";
    if (this.stats.best != null) {
      e.best.textContent = this.fmt(this.stats.best);
      e.best.title = `lowest raw loss seen this session, at step ${this.stats.bestStep}`;
    }
    // Per-t bucket rows: each shows the latest *measured* value --
    // this report's when its window sampled that range, else the last
    // one, dimmed and tagged @step so a carried number never claims to
    // be this step's measurement. The chart still draws per-step gaps.
    // A range never measured this run stays em dash + empty bar.
    const rows = [
      { key: "loss_t_low", val: e.lossTlow, bar: e.barTlow, range: "[0,333)" },
      { key: "loss_t_mid", val: e.lossTmid, bar: e.barTmid, range: "[333,666)" },
      { key: "loss_t_high", val: e.lossThigh, bar: e.barThigh, range: "[666,1000)" },
    ];
    const shown = [];
    for (const r of rows) {
      const v = data[r.key];
      if (typeof v === "number") {
        this.bucketLast[r.key] = { v, step: data.step };
        shown.push({ r, v, stale: false, step: data.step });
      } else if (this.bucketLast[r.key]) {
        const last = this.bucketLast[r.key];
        shown.push({ r, v: last.v, stale: true, step: last.step });
      } else {
        shown.push({ r, v: null });
      }
    }
    const bucketMax = Math.max(...shown.filter((s) => s.v != null).map((s) => s.v), 0);
    for (const s of shown) {
      if (s.v == null) {
        s.r.val.textContent = "\u2014";
        s.r.val.title = `never sampled in ${s.r.range} this run`;
        s.r.val.classList.remove("mon-stale");
        s.r.bar.classList.remove("mon-stale");
        this.setBar(s.r.bar, null, bucketMax);
        continue;
      }
      s.r.val.textContent = s.stale ? `${this.fmt(s.v)} @${s.step}` : this.fmt(s.v);
      s.r.val.classList.toggle("mon-stale", s.stale);
      s.r.val.title = s.stale
        ? `last measured at step ${s.step}; this window had no sample in ${s.r.range}`
        : `measured at step ${data.step} in ${s.r.range}`;
      s.r.bar.classList.toggle("mon-stale", s.stale);
      this.setBar(s.r.bar, s.v, bucketMax);
    }

    e.lr.textContent = data.lr !== undefined ? data.lr.toExponential(2) : "\u2014";
    // Grad norm row appears only once a report carries it -- a
    // permanently em-dashed row would claim "measured, zero known".
    if (e.gradRow) {
      if (typeof data.grad_norm === "number") {
        e.gradRow.classList.remove("mon-hidden");
        e.gradNorm.textContent = data.grad_norm.toFixed(3);
      } else if (!e.gradRow.classList.contains("mon-hidden")) {
        e.gradNorm.textContent = "\u2014";
      }
    }
    if (e.runInfo && data.optimizer) e.runInfo.textContent = data.optimizer;

    if (data.vram_reserved_mb !== undefined) {
      e.memorySection.classList.remove("mon-hidden");
      e.vram.textContent = this.fmtMB(data.vram_reserved_mb);
      const bits = [];
      if (data.vram_allocated_mb !== undefined) {
        bits.push("alloc " + this.fmtMB(data.vram_allocated_mb));
      }
      if (this.budget != null) {
        const pct = (data.vram_reserved_mb / this.budget) * 100;
        bits.push("budget " + this.fmtMB(this.budget) + " \u00b7 " + pct.toFixed(0) + "%");
        // Meter fill = pressure band: accent while comfortable, amber
        // approaching it, red at/over it. Width caps at 100% but the
        // sub-line always prints the true percentage.
        e.vramFill.style.width = Math.min(100, pct) + "%";
        e.vramFill.style.background = pct >= 100 ? "var(--red)" : pct >= 90 ? "#ffb300" : "var(--accent)";
      } else {
        bits.push("budget \u2014");
        e.vramFill.style.width = "0%";
      }
      e.vramSub.textContent = bits.join(" \u00b7 ");
    }
    if (this.stats.peakVram != null) {
      e.vramPeak.textContent = this.fmtMB(this.stats.peakVram);
    }

    if (data.total_steps) {
      const pct = Math.min(100, (data.step / data.total_steps) * 100);
      e.progressFill.style.width = pct + "%";
      e.progressPct.textContent = Math.round(pct) + "%";
    }

    const elapsedS = (now - this.firstEventTime) / 1000;
    e.elapsed.textContent = this.fmtDuration(elapsedS);

    if (this.recentRates.length >= 2) {
      const first = this.recentRates[0], last = this.recentRates[this.recentRates.length - 1];
      const dt = (last.t - first.t) / 1000, dstep = last.step - first.step;
      const rate = dt > 0 ? dstep / dt : 0;
      e.rate.textContent = rate > 0 ? rate.toFixed(2) : "\u2014";
      if (rate > 0 && data.total_steps) {
        const remaining = (data.total_steps - data.step) / rate;
        e.eta.textContent = this.fmtDuration(remaining);
      }
    }
  }

  /* Session CSV: every step report this page received, columns = union
     of keys in first-seen order, missing keys empty -- never imputed. */
  exportCsv() {
    if (this.rawHistory.length === 0) return;
    const cols = [];
    for (const r of this.rawHistory) {
      for (const k of Object.keys(r)) if (!cols.includes(k)) cols.push(k);
    }
    const esc = (v) => {
      if (v === undefined || v === null) return "";
      const s = String(v);
      return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    const lines = [cols.join(",")];
    for (const r of this.rawHistory) lines.push(cols.map((c) => esc(r[c])).join(","));
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const lastStep = this.rawHistory[this.rawHistory.length - 1].step;
    a.href = url;
    a.download = `monitor_${this.monitorId}_step${lastStep}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  /* One bucket bar: width relative to this report's largest bucket, a
     2%-sliver floor so a small value still reads as present, empty for
     an absent one (no bar, not a zero bar). */
  setBar(el, v, max) {
    if (!el) return;
    if (typeof v !== "number" || !(max > 0)) { el.style.width = "0%"; return; }
    el.style.width = Math.max(2, Math.min(100, (v / max) * 100)) + "%";
  }

  /* Everything a previous run put on screen, back to "no data yet" --
     called on clear(). Config-dependent blocks (memory, grad norm, the
     secondary chart cards) hide again: the next run may not produce
     their keys at all. */
  resetReadouts() {
    const e = this.els;
    for (const node of [e.barTlow, e.barTmid, e.barThigh, e.vramFill]) {
      if (node) node.style.width = "0%";
    }
    e.vramCard.classList.add("mon-hidden");
    e.timingCard.classList.add("mon-hidden");
    e.memorySection.classList.add("mon-hidden");
    e.gradRow.classList.add("mon-hidden");
    e.step.textContent = "\u2014";
    e.loss.textContent = "\u2014";
    e.smoothed.textContent = "\u2014";
    e.best.textContent = "\u2014";
    for (const r of [
      { val: e.lossTlow, bar: e.barTlow },
      { val: e.lossTmid, bar: e.barTmid },
      { val: e.lossThigh, bar: e.barThigh },
    ]) {
      r.val.textContent = "\u2014";
      r.val.title = "";
      r.val.classList.remove("mon-stale");
      r.bar.classList.remove("mon-stale");
    }
    e.lr.textContent = "\u2014";
    e.gradNorm.textContent = "\u2014";
    e.vram.textContent = "\u2014";
    e.vramPeak.textContent = "\u2014";
    e.vramSub.textContent = "\u2014";
    e.rate.textContent = "\u2014";
    e.elapsed.textContent = "\u2014";
    e.eta.textContent = "\u2014";
    e.runInfo.textContent = "";
    e.progressFill.style.width = "0%";
    e.progressPct.textContent = "0%";
  }

  /* ---- graph view: one window shared by all three charts ---- */

  /* Recompute the window and push it to every chart, the slider and
     the readout. Live + "all" = no window; live + count = last N;
     frozen/panned = the locked range captured when follow stopped, so
     incoming reports can never move a held view. */
  applyView() {
    const e = this.els, pts = this.chart.points;
    const n = this.view.count === "all" ? pts.length : Math.min(this.view.count, pts.length);
    let range;
    if (this.view.live) {
      range = (!pts.length || this.view.count === "all")
        ? null
        : { min: pts[pts.length - n].step, max: pts[pts.length - 1].step };
      this.view.startIdx = Math.max(0, pts.length - n);
    } else {
      range = this.view.lockedRange;
    }
    this.chart.setViewRange(range);
    this.vramChart.setViewRange(range);
    if (this.timingChart) this.timingChart.setViewRange(range);

    const maxIdx = Math.max(0, pts.length - n);
    // While the pointer holds the thumb, leave the slider alone:
    // rewriting max/value under the finger fights the browser's drag
    // tracking and teleports the thumb. Resync on release.
    if (!this.view.dragging) {
      e.slider.max = String(maxIdx);
      e.slider.disabled = this.view.count === "all";
      e.slider.value = String(Math.min(this.view.startIdx, maxIdx));
    }

    if (!pts.length) {
      e.windowReadout.textContent = "steps \u2014";
    } else {
      const latest = pts[pts.length - 1].step;
      const a = range ? range.min : pts[0].step;
      const b = range ? range.max : latest;
      // "/ latest" only when the window ends short of the newest
      // record, so a held view visibly reports newer data exists.
      e.windowReadout.textContent = `steps ${a}\u2013${b}` + (b < latest ? ` / ${latest}` : "");
    }
    const frozen = !this.view.live;
    e.freeze.textContent = frozen ? "Resume" : "Freeze";
    e.freeze.classList.toggle("mon-freeze-on", frozen);
  }

  /* Pointer holds the slider thumb: while held, applyView stops
     rewriting the slider geometry. On release, applyView resyncs. */
  setDragging(on) {
    on = !!on;
    if (this.view.dragging === on) return;
    this.view.dragging = on;
    if (on) {
      this.view.frozenMax = this.els.slider.max;
    } else {
      this.view.frozenMax = null;
      this.applyView();
    }
  }

  /* Freeze/resume: lock the window exactly where it is, or release it
     back to following the newest records. */
  toggleFreeze() {
    if (this.view.live) {
      this.view.live = false;
      const pts = this.chart.points;
      this.relockAt(pts.length ? pts[pts.length - 1].step : null);
    } else {
      this.view.live = true;
      this.view.lockedRange = null;
    }
    this.applyView();
  }

  /* Lock the window so it ENDS at endStep: keeps the current count,
     positions the slider index, stores the concrete step range that
     follow will no longer touch. */
  relockAt(endStep) {
    const pts = this.chart.points;
    if (!pts.length || endStep == null) {
      this.view.lockedRange = null;
      this.view.startIdx = 0;
      return;
    }
    let endIdx = pts.length - 1;
    while (endIdx > 0 && pts[endIdx].step > endStep) endIdx--;
    if (this.view.count === "all") {
      this.view.startIdx = 0;
      this.view.lockedRange = { min: pts[0].step, max: pts[endIdx].step };
      return;
    }
    const startIdx = Math.max(0, endIdx - this.view.count + 1);
    this.view.startIdx = startIdx;
    this.view.lockedRange = { min: pts[startIdx].step, max: pts[endIdx].step };
  }

  setCount(v) {
    this.view.count = v;
    // Resizing a held window keeps its END anchored, so widening
    // 50 -> 100 while frozen reveals earlier records instead of
    // jumping past the freeze point.
    if (!this.view.live && this.view.lockedRange) this.relockAt(this.view.lockedRange.max);
    for (const [node, val] of this.countDefs) node.classList.toggle("active", val === v);
    this.applyView();
  }

  /* Slider: index into recorded records. Dragging to the far end
     returns to live; anywhere else locks the window there. While held,
     the far end is the frozen max the thumb was grabbed with. */
  pan(idx) {
    const pts = this.chart.points;
    if (this.view.count === "all" || !pts.length) return;
    const n = Math.min(this.view.count, pts.length);
    const maxIdx = Math.max(0, pts.length - n);
    const frozen = this.view.dragging && this.view.frozenMax != null
      ? Math.min(parseInt(this.view.frozenMax, 10) || 0, maxIdx)
      : maxIdx;
    idx = Math.max(0, Math.min(idx, maxIdx));
    if (idx >= frozen) {
      this.view.live = true;
      this.view.lockedRange = null;
    } else {
      this.view.live = false;
      this.relockAt(pts[Math.min(idx + n - 1, pts.length - 1)].step);
    }
    this.applyView();
  }

  fmt(v) {
    if (v === undefined || v === null) return "\u2014";
    return v >= 1 ? v.toFixed(4) : v >= 0.001 ? v.toFixed(5) : v.toExponential(2);
  }

  fmtMB(v) {
    if (v === undefined || v === null) return "\u2014";
    return v >= 1024 ? (v / 1024).toFixed(1) + " GB" : Math.round(v) + " MB";
  }

  fmtDuration(seconds) {
    if (!isFinite(seconds) || seconds < 0) return "\u2014";
    const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = Math.floor(seconds % 60);
    return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`;
  }
}

function idFromUrl() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  return parts[parts.length - 1] || "";
}

function boot() {
  const monitorId = idFromUrl();
  const idEl = el("mon-id");
  idEl.textContent = "id: " + monitorId;
  idEl.addEventListener("click", () => {
    navigator.clipboard.writeText(monitorId).then(() => {
      const original = idEl.textContent;
      idEl.textContent = "copied!";
      setTimeout(() => { idEl.textContent = original; }, 800);
    }).catch(() => {});
  });

  const els = {
    canvas: el("mon-loss-chart"),
    vramCanvas: el("mon-vram-chart"),
    timingCanvas: el("mon-timing-chart"),
    vramCard: el("mon-vram-card"),
    timingCard: el("mon-timing-card"),
    statusDot: el("mon-status-dot"),
    statusText: el("mon-status-text"),
    step: el("m-step"),
    loss: el("m-loss"),
    smoothed: el("m-smoothed"),
    best: el("m-best"),
    lossTlow: el("m-loss-t-low"),
    lossTmid: el("m-loss-t-mid"),
    lossThigh: el("m-loss-t-high"),
    barTlow: el("m-bar-t-low"),
    barTmid: el("m-bar-t-mid"),
    barThigh: el("m-bar-t-high"),
    vram: el("m-vram"),
    vramPeak: el("m-vram-peak"),
    vramFill: el("m-vram-fill"),
    vramSub: el("m-vram-sub"),
    memorySection: el("m-memory"),
    gradRow: el("m-grad-row"),
    gradNorm: el("m-grad-norm"),
    lr: el("m-lr"),
    rate: el("m-rate"),
    elapsed: el("m-elapsed"),
    eta: el("m-eta"),
    progressFill: el("m-progress-fill"),
    progressPct: el("m-progress-pct"),
    runInfo: el("m-runinfo"),
    freeze: el("mon-freeze"),
    slider: el("mon-slider"),
    windowReadout: el("mon-window"),
    count50: el("mon-count-50"),
    count100: el("mon-count-100"),
    count250: el("mon-count-250"),
    count1000: el("mon-count-1000"),
    countAll: el("mon-count-all"),
  };

  const dashboard = new MonitorDashboard(monitorId, els);
  els.freeze.addEventListener("click", () => dashboard.toggleFreeze());
  for (const [node, v] of dashboard.countDefs) {
    node.addEventListener("click", () => dashboard.setCount(v));
  }
  els.slider.addEventListener("input", () => dashboard.pan(parseInt(els.slider.value, 10)));
  // Hold/release on the thumb -- pointerup/pointercancel on document
  // too, since the release can land outside the input.
  els.slider.addEventListener("pointerdown", () => dashboard.setDragging(true));
  const release = () => dashboard.setDragging(false);
  els.slider.addEventListener("pointerup", release);
  els.slider.addEventListener("pointercancel", release);
  els.slider.addEventListener("change", release);
  document.addEventListener("pointerup", release);
  document.addEventListener("pointercancel", release);
  dashboard.applyView(); // initial control state on empty charts
  el("mon-export-csv").addEventListener("click", () => dashboard.exportCsv());
  dashboard.connect();
}

boot();
