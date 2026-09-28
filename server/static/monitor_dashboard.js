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

  class MonitorDashboard {
    constructor(monitorId, els) {
      this.monitorId = monitorId;
      this.els = els;
      this.chart = new LossChart(els.canvas, { series: LOSS_SERIES });
      this.firstEventTime = null;
      this.lastEvent = null;
      this.recentRates = []; // {t, step} pairs, last few, for a steps/sec estimate
      this.source = null;
    }

    connect() {
      this.source = new EventSource(`/api/nodegraph/monitor/${encodeURIComponent(this.monitorId)}/stream`);
      this.source.onopen = () => this.setStatus("live");
      this.source.onerror = () => this.setStatus("disconnected");
      this.source.onmessage = (ev) => this.handleEvent(ev);
    }

    setStatus(state) {
      this.els.statusDot.className = "mon-status-dot" + (state === "live" ? " live" : state === "disconnected" ? " disconnected" : "");
      this.els.statusText.textContent = state === "live" ? "live" : state === "disconnected" ? "disconnected \u2014 retrying\u2026" : "connecting\u2026";
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
        this.firstEventTime = null;
        this.lastEvent = null;
        this.recentRates = [];
        return;
      }
      if (data.step === undefined) return; // not a training-progress-shaped event; ignore rather than guess

      const now = data.t ? data.t * 1000 : Date.now();
      if (this.firstEventTime === null) this.firstEventTime = now;
      this.recentRates.push({ t: now, step: data.step });
      if (this.recentRates.length > 20) this.recentRates.shift();
      this.lastEvent = data;

      // Values object (not a bare number): series absent from this report
      // are omitted, which LossChart renders as a gap.
      const values = {};
      for (const s of LOSS_SERIES) if (data[s.key] !== undefined) values[s.key] = data[s.key];
      this.chart.addPoint(data.step, values);
      this.updateMetrics(data, now);
    }

    updateMetrics(data, now) {
      const e = this.els;
      e.step.textContent = data.total_steps ? `${data.step} / ${data.total_steps}` : String(data.step);
      e.loss.textContent = this.fmt(data.loss);
      e.smoothed.textContent = this.chart.points.length && this.chart.points[this.chart.points.length - 1].smoothed != null
        ? this.fmt(this.chart.points[this.chart.points.length - 1].smoothed) : "\u2014";
      // Per-t bucket readouts: latest value or em dash when this report had
      // no samples for that bucket (same rule as the chart's gaps).
      if (e.lossTlow) e.lossTlow.textContent = this.fmt(data.loss_t_low);
      if (e.lossTmid) e.lossTmid.textContent = this.fmt(data.loss_t_mid);
      if (e.lossThigh) e.lossThigh.textContent = this.fmt(data.loss_t_high);
      e.lr.textContent = data.lr !== undefined ? data.lr.toExponential(2) : "\u2014";

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

    fmt(v) {
      if (v === undefined || v === null) return "\u2014";
      return v >= 1 ? v.toFixed(4) : v >= 0.001 ? v.toFixed(5) : v.toExponential(2);
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
      statusDot: document.getElementById("mon-status-dot"),
      statusText: document.getElementById("mon-status-text"),
      step: document.getElementById("m-step"),
      loss: document.getElementById("m-loss"),
      smoothed: document.getElementById("m-smoothed"),
      lossTlow: document.getElementById("m-loss-t-low"),
      lossTmid: document.getElementById("m-loss-t-mid"),
      lossThigh: document.getElementById("m-loss-t-high"),
      lr: document.getElementById("m-lr"),
      rate: document.getElementById("m-rate"),
      elapsed: document.getElementById("m-elapsed"),
      eta: document.getElementById("m-eta"),
      progressFill: document.getElementById("m-progress-fill"),
      progressPct: document.getElementById("m-progress-pct"),
    };

    const dashboard = new MonitorDashboard(monitorId, els);
    dashboard.connect();
  }

  boot();
})();
