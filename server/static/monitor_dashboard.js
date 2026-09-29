(function () {
  "use strict";

  // The managed/main trainers' per-t loss diagnostics (nodes/train/loss.py's
  // t_bucket_losses keys) plotted alongside the total, one color per series:
  // total in blue; t-low (fine detail, late denoising) green; t-mid amber;
  // t-high (coarse structure, early denoising) red -- red last because that's
  // the end whose divergence reads first as "destructive output at strength
  // 1.0". Series keys whose report lacks the key (a window whose batches never
  // sampled that bucket) render as gaps, not zeros -- see LossChart's header.
  const LOSS_SERIES = [
    { key: "loss", label: "loss", color: "#6c8cff" },
    { key: "loss_t_low", label: "t low", color: "#4caf50" },
    { key: "loss_t_mid", label: "t mid", color: "#ffb300" },
    { key: "loss_t_high", label: "t high", color: "#ff5252" },
  ];

  // Allocator state + per-resident footprints, same `vram_{k}` /
  // `resident_{name}_mb` key names on both trainer routes (managed reports
  // them every step on an XPU run; the main route reports them when timing
  // is collected). reserved is the primary series -- it's the number the
  // budget is actually enforced against; peak reserved rides above it so a
  // within-step spike is visible even when the sampled value looks calm.
  const VRAM_SERIES = [
    { key: "vram_reserved_mb", label: "reserved", color: "#6c8cff" },
    { key: "vram_peak_reserved_mb", label: "peak reserved", color: "#ff9800" },
    { key: "vram_allocated_mb", label: "allocated", color: "#4caf50" },
    { key: "resident_model_mb", label: "model", color: "#26c6da" },
    { key: "resident_optimizer_mb", label: "optimizer", color: "#ab47bc" },
    { key: "resident_text_encoder_mb", label: "text encoder", color: "#8d6e63" },
  ];

  // Phase-timing series aren't known up front: which {label}_ms keys exist
  // depends on the trainer route (and its phase list), discovered from the
  // first report that carries any. Palette-assigned in discovery order;
  // step_total_ms always gets the neutral tone so "sum of everything"
  // reads as an envelope, not one more competing phase.
  const TIMING_PALETTE = ["#6c8cff", "#4caf50", "#ffb300", "#ff5252",
                          "#ab47bc", "#26c6da", "#ff9800", "#8d6e63"];
  const TIMING_TOTAL_COLOR = "#cfd8dc";

  // LossChart's default cap of 1000 points would silently drop the oldest
  // records -- exactly what the history slider exists to reach. Runs are a
  // few thousand steps at most; 100k is "no truncation in practice".
  const MAX_RECORDS = 100000;

  class MonitorDashboard {
    constructor(monitorId, els) {
      this.monitorId = monitorId;
      this.els = els;
      this.chart = new LossChart(els.canvas, { series: LOSS_SERIES, maxPoints: MAX_RECORDS });
      this.vramChart = new LossChart(els.vramCanvas, { series: VRAM_SERIES, maxPoints: MAX_RECORDS });
      // Built lazily on the first report carrying *_ms keys -- a run that
      // never collects timing (env/profile off) simply never shows the card.
      this.timingChart = null;
      this.firstEventTime = null;
      this.lastEvent = null;
      this.recentRates = []; // {t, step} pairs, last few, for a steps/sec estimate
      this.source = null;
      this.conn = "connecting"; // SSE connection state, for the status readout
      this.runEnded = null;     // null while running; {step, cancelled} after run_end
      this.rawHistory = [];     // every step report this session saw -- the CSV export
      this.budget = null;       // vram_budget_mb, once a report states one
      this.bucketLast = {};     // loss_t_* -> {v, step}: latest measured value per range (rail carry)
      // Graph-view controls, one window shared by all three charts (they all
      // plot against step): `count` = records shown ("all" | n), `live` =
      // window follows the newest records, `startIdx` = slider position,
      // `lockedRange` = concrete step window captured when follow stopped.
      this.view = { count: "all", live: true, startIdx: 0, lockedRange: null };
      this.countDefs = [
        [els.count50, 50], [els.count100, 100], [els.count250, 250],
        [els.count1000, 1000], [els.countAll, "all"],
      ];
      this.stats = { best: null, bestStep: null, peakVram: null };
    }

    connect() {
      this.source = new EventSource(`/api/nodegraph/monitor/${encodeURIComponent(this.monitorId)}/stream`);
      this.source.onopen = () => this.setStatus("live");
      this.source.onerror = () => this.setStatus("disconnected");
      this.source.onmessage = (ev) => this.handleEvent(ev);
    }

    setStatus(state) {
      this.conn = state;
      this.renderStatus();
    }

    /* Connection state and run state combined into one readout. A finished
       run's SSE stream stays open (and the browser reconnects through the
       occasional blip), so "live" alone would keep saying live after the
       run is over -- run_end is what makes "finished vs still going"
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
        // A fresh run just started reporting to this monitor_id -- see
        // MonitorBus.clear()'s own docstring (monitor_bus.py) for why this exists:
        // without it, re-running training against the same monitor_id overlaid the
        // new run's line on the old one with no indication they were different runs.
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
        // Terminal status from the trainer (both routes, both exits): the
        // run is over -- normal or cancelled -- and these numbers are final.
        // Checked before the step-shaped filter below: run_end carries a
        // step, but it is not a step report.
        this.runEnded = { step: data.step, cancelled: !!data.cancelled };
        this.renderStatus();
        return;
      }
      if (data.step === undefined) return; // not a training-progress-shaped event; ignore rather than guess

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
      // Report has no timing at all -> nothing to show, and crucially no
      // all-gap point: after a clear() the previous run's chart object
      // still exists, and re-unhiding the card around an empty point would
      // present "no timing data" as if this run had some.
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
        label: k.slice(0, -3), // strip the "_ms" suffix -- the phase label itself
        color: TIMING_PALETTE[i % TIMING_PALETTE.length],
      }));
      series.push({ key: "step_total_ms", label: "total", color: TIMING_TOTAL_COLOR });
      this.timingChart = new LossChart(this.els.timingCanvas, { series, maxPoints: MAX_RECORDS });
    }

    /* A phase key that wasn't in the first timing report still has to be
       plotted rather than silently dropped (LossChart only stores keys it
       knows). Real runs don't change phase lists mid-run; this exists so
       the failure mode of being wrong about that is "a late line appears",
       not "data vanished". */
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
      // Peak: the explicit per-step peak when the trainer reports one,
      // otherwise fall back to the sampled reserved value (which is all a
      // config without peaks can honestly claim).
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
      // Per-t bucket readouts: with batch 2 a bucket often gets no sample in
      // a window, so an em dash every few steps made the rail read like data
      // loss. Each row shows the latest *measured* value for its range: this
      // report's own when its window sampled that range, otherwise the last
      // one -- dimmed and tagged @step, so a carried number never claims to
      // be this step's measurement. The chart still draws per-step gaps: the
      // chart answers "what happened at step N", the rail "where does each
      // range stand now". A range never measured this run stays em dash +
      // empty bar; bars compare the three displayed magnitudes (same target,
      // same scale, so they compare honestly).
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
      // Grad norm row appears only once a report actually carries it (clip
      // runs only under grad_clip_max_norm>0) -- before that, a permanently
      // em-dashed row would claim "measured, zero known" instead of
      // "never measured".
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
          // Meter fill = pressure band against the stated budget: accent
          // while comfortable, amber approaching it, red at/over it. The
          // width caps at 100% (a bar can't overflow its track) but the
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

    /* Session CSV: every step report this page received, columns = union of
       keys in first-seen order (reports differ by route/config, and a fixed
       column list would drop whichever keys this run happens to have). One
       row per report, missing keys empty -- never imputed. */
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
       2%-sliver floor so a genuinely-present but small value still reads
       as present, empty for an absent one (no bar, not a zero bar). */
    setBar(el, v, max) {
      if (!el) return;
      if (typeof v !== "number" || !(max > 0)) { el.style.width = "0%"; return; }
      el.style.width = Math.max(2, Math.min(100, (v / max) * 100)) + "%";
    }

    /* Everything a previous run put on screen, back to "no data yet" --
       called on MonitorBus clear(). Config-dependent blocks (memory
       section, grad-norm row, the two secondary chart cards) also go
       hidden again: the next run may not produce their keys at all, and
       stale numbers from the old run would read as the new run's. */
    resetReadouts() {
      const e = this.els;
      for (const el of [e.barTlow, e.barTmid, e.barThigh, e.vramFill]) {
        if (el) el.style.width = "0%";
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

    /* Recompute the window from state and push it to every chart, the
       slider and the readout. Live + "all" = no window (everything
       recorded); live + count = last N records; frozen/panned = the
       locked step range captured when follow stopped, so incoming
       reports can never move a held view. */
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
      e.slider.max = String(maxIdx);
      e.slider.disabled = this.view.count === "all";
      e.slider.value = String(Math.min(this.view.startIdx, maxIdx));

      if (!pts.length) {
        e.windowReadout.textContent = "steps \u2014";
      } else {
        const latest = pts[pts.length - 1].step;
        const a = range ? range.min : pts[0].step;
        const b = range ? range.max : latest;
        // "/ latest" only when the window ends short of the newest record,
        // so a held view visibly reports that newer data exists.
        e.windowReadout.textContent = `steps ${a}\u2013${b}` + (b < latest ? ` / ${latest}` : "");
      }
      const frozen = !this.view.live;
      e.freeze.textContent = frozen ? "Resume" : "Freeze";
      e.freeze.classList.toggle("mon-freeze-on", frozen);
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

    /* Lock the window so it ENDS at endStep: keeps the current count
       ("all" = everything up to endStep), positions the slider index and
       stores the concrete step range that follow will no longer touch. */
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
      // Resizing a held window keeps its END anchored, so switching
      // 50 -> 100 while frozen reveals earlier records instead of
      // jumping the view forward past the freeze point.
      if (!this.view.live && this.view.lockedRange) this.relockAt(this.view.lockedRange.max);
      for (const [el, val] of this.countDefs) el.classList.toggle("active", val === v);
      this.applyView();
    }

    /* Slider: index into recorded records. Dragging to the far end returns
       to live; anywhere else locks the window at that position. */
    pan(idx) {
      const pts = this.chart.points;
      if (this.view.count === "all" || !pts.length) return;
      const n = Math.min(this.view.count, pts.length);
      const maxIdx = Math.max(0, pts.length - n);
      idx = Math.max(0, Math.min(idx, maxIdx));
      if (idx >= maxIdx) {
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
    const idEl = document.getElementById("mon-id");
    idEl.textContent = "id: " + monitorId;
    idEl.addEventListener("click", () => {
      navigator.clipboard.writeText(monitorId).then(() => {
        const original = idEl.textContent;
        idEl.textContent = "copied!";
        setTimeout(() => { idEl.textContent = original; }, 800);
      }).catch(() => {});
    });

    const els = {
      canvas: document.getElementById("mon-loss-chart"),
      vramCanvas: document.getElementById("mon-vram-chart"),
      timingCanvas: document.getElementById("mon-timing-chart"),
      vramCard: document.getElementById("mon-vram-card"),
      timingCard: document.getElementById("mon-timing-card"),
      statusDot: document.getElementById("mon-status-dot"),
      statusText: document.getElementById("mon-status-text"),
      step: document.getElementById("m-step"),
      loss: document.getElementById("m-loss"),
      smoothed: document.getElementById("m-smoothed"),
      best: document.getElementById("m-best"),
      lossTlow: document.getElementById("m-loss-t-low"),
      lossTmid: document.getElementById("m-loss-t-mid"),
      lossThigh: document.getElementById("m-loss-t-high"),
      barTlow: document.getElementById("m-bar-t-low"),
      barTmid: document.getElementById("m-bar-t-mid"),
      barThigh: document.getElementById("m-bar-t-high"),
      vram: document.getElementById("m-vram"),
      vramPeak: document.getElementById("m-vram-peak"),
      vramFill: document.getElementById("m-vram-fill"),
      vramSub: document.getElementById("m-vram-sub"),
      memorySection: document.getElementById("m-memory"),
      gradRow: document.getElementById("m-grad-row"),
      gradNorm: document.getElementById("m-grad-norm"),
      lr: document.getElementById("m-lr"),
      rate: document.getElementById("m-rate"),
      elapsed: document.getElementById("m-elapsed"),
      eta: document.getElementById("m-eta"),
      progressFill: document.getElementById("m-progress-fill"),
      progressPct: document.getElementById("m-progress-pct"),
      runInfo: document.getElementById("m-runinfo"),
      freeze: document.getElementById("mon-freeze"),
      slider: document.getElementById("mon-slider"),
      windowReadout: document.getElementById("mon-window"),
      count50: document.getElementById("mon-count-50"),
      count100: document.getElementById("mon-count-100"),
      count250: document.getElementById("mon-count-250"),
      count1000: document.getElementById("mon-count-1000"),
      countAll: document.getElementById("mon-count-all"),
    };

    const dashboard = new MonitorDashboard(monitorId, els);
    els.freeze.addEventListener("click", () => dashboard.toggleFreeze());
    for (const [el, v] of dashboard.countDefs) {
      el.addEventListener("click", () => dashboard.setCount(v));
    }
    els.slider.addEventListener("input", () => dashboard.pan(parseInt(els.slider.value, 10)));
    dashboard.applyView(); // initial control state (empty charts: readout "steps —", slider disabled under All)
    document.getElementById("mon-export-csv")
      .addEventListener("click", () => dashboard.exportCsv());
    dashboard.connect();
  }

  boot();
})();
