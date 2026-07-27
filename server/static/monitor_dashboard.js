(function () {
  "use strict";

  class MonitorDashboard {
    constructor(monitorId, els) {
      this.monitorId = monitorId;
      this.els = els;
      this.chart = new LossChart(els.canvas);
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
      if (data.step === undefined) return; // not a training-progress-shaped event; ignore rather than guess

      const now = data.t ? data.t * 1000 : Date.now();
      if (this.firstEventTime === null) this.firstEventTime = now;
      this.recentRates.push({ t: now, step: data.step });
      if (this.recentRates.length > 20) this.recentRates.shift();
      this.lastEvent = data;

      this.chart.addPoint(data.step, data.loss);
      this.updateMetrics(data, now);
    }

    updateMetrics(data, now) {
      const e = this.els;
      e.step.textContent = data.total_steps ? `${data.step} / ${data.total_steps}` : String(data.step);
      e.loss.textContent = this.fmt(data.loss);
      e.smoothed.textContent = this.chart.points.length && this.chart.points[this.chart.points.length - 1].smoothed != null
        ? this.fmt(this.chart.points[this.chart.points.length - 1].smoothed) : "\u2014";
      e.lr.textContent = data.lr !== undefined ? data.lr.toExponential(2) : "\u2014";

      if (data.total_steps) {
        const pct = Math.min(100, (data.step / data.total_steps) * 100);
        e.progressFill.style.width = pct + "%";
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
      lr: document.getElementById("m-lr"),
      rate: document.getElementById("m-rate"),
      elapsed: document.getElementById("m-elapsed"),
      eta: document.getElementById("m-eta"),
      progressFill: document.getElementById("m-progress-fill"),
    };

    const dashboard = new MonitorDashboard(monitorId, els);
    dashboard.connect();
  }

  boot();
})();
