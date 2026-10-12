/* ---------------------------------------------------------------------------
   board/displays.js -- how each display *kind* is drawn.

   The backend says what a live view *is* (nodes/ui.py: kind chart | scalar |
   bars | text | progress, plus the graph's own `status`); this file is the
   only place that knows what each looks like. A new module of an existing
   kind needs no change here; a new kind is one function below, and an
   unknown kind degrades to a labelled placeholder rather than a blank hole.

   Every factory returns {el, update(), destroy()}. `update()` re-derives the
   picture from the shared `StreamState` (board/series.js) and is cheap to
   call -- the board throttles it. Nothing here keeps data of its own, so a
   block that is collapsed, removed and re-added shows the same thing.
   --------------------------------------------------------------------------- */

import { LossChart } from "../lib/loss_chart.js";
import {
  barsOf, chartRows, decimate, formatDuration, formatValue, progressOf,
  referenceLines, rowsToCsv, scalarOf, seriesFor, textOf,
} from "./series.js";
import { COUNTS, ViewWindow, rowsForWindow } from "./view.js";

/** Points a chart is asked to draw. More than the plot has pixels for is
    wasted work; decimate() keeps the extremes so it is never a lie. */
const MAX_DRAWN_ROWS = 800;

const div = (cls, text) => {
  const d = document.createElement("div");
  if (cls) d.className = cls;
  if (text !== undefined) d.textContent = text;
  return d;
};

/* ---- chart --------------------------------------------------------------- */

function chartDisplay({ module, state }) {
  const wrap = div("bd-chart-wrap");

  // ---- controls: the old Monitor page's window / freeze / export ----
  const bar = div("bd-chart-bar");
  const freeze = document.createElement("button");
  freeze.type = "button";
  freeze.className = "bd-btn bd-freeze";
  freeze.textContent = "Freeze";
  freeze.title = "Hold the picture where it is while the run continues";
  const counts = div("bd-counts");
  const countButtons = COUNTS.map((value) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "bd-btn bd-count";
    b.textContent = value === "all" ? "All" : String(value);
    b.title = value === "all"
      ? "Every record (no panning)" : `Window of the last ${value} records`;
    b.addEventListener("click", () => { view.setCount(value, xs); update(); });
    counts.append(b);
    return [b, value];
  });
  const readout = div("bd-window", "—");
  const csv = document.createElement("button");
  csv.type = "button";
  csv.className = "bd-btn bd-csv";
  csv.textContent = "CSV";
  csv.title = "Download every record this view holds as CSV";
  bar.append(freeze, counts, readout, csv);

  const root = div("bd-chart");
  const canvas = document.createElement("canvas");
  canvas.className = "bd-canvas";
  root.append(canvas);

  const slider = document.createElement("input");
  slider.type = "range";
  slider.className = "bd-slider";
  slider.min = "0"; slider.max = "0"; slider.value = "0"; slider.step = "1";
  slider.title = "Scroll back through the run (needs a window size other than All)";
  wrap.append(bar, root, slider);

  const view = new ViewWindow();
  let xs = [];
  let chart = null;
  let signature = "";
  let lastCount = 0;
  const hidden = new Map(); // series key -> hidden, survives a rebuild
  const trend = module.scale !== "linear"; // the loss-like chart: is it getting better?
  const smoothWindow = module.scale === "linear" ? 3 : 24;

  const build = (list) => {
    if (chart) {
      for (const s of chart.series) hidden.set(s.key, !!s.hidden);
      chart.destroy();
    }
    chart = new LossChart(canvas, {
      series: list.map((s) => ({
        key: s.key, label: s.label, color: s.color, hidden: hidden.get(s.key) || false,
      })),
      maxPoints: MAX_DRAWN_ROWS + 8,
      scale: module.scale === "linear" ? "linear" : "symlog",
      smoothWindow,
      trend,
      xLabel: module.x === "t" ? "t (s)" : "Step",
    });
    signature = list.map((s) => `${s.key}:${s.color}`).join("|");
  };

  const syncControls = (v) => {
    for (const [b, value] of countButtons) b.classList.toggle("active", value === view.count);
    freeze.textContent = v.frozen ? "Resume" : "Freeze";
    freeze.classList.toggle("is-on", v.frozen);
    slider.disabled = v.sliderDisabled;
    if (!view.dragging) {
      slider.max = String(v.maxIdx);
      slider.value = String(v.startIdx);
    }
    if (v.first === null) readout.textContent = "—";
    else {
      const unit = module.x === "t" ? "s" : "";
      readout.textContent = `${module.x === "t" ? "t " : "steps "}${formatValue(v.first)}–${formatValue(v.last)}${unit}`
        + (v.last < v.latest ? ` / ${formatValue(v.latest)}` : "");
    }
  };

  function update() {
    const list = seriesFor(module, state);
    if (!list.length) {
      // Nothing to draw -- which is also what a `clear` leaves behind. Return
      // without touching the canvas and the PREVIOUS run's curve stays up until
      // the new run's first report (minutes, while the model loads).
      if (chart) chart.reset();
      xs = [];
      lastCount = 0;
      view.restart();
      syncControls(view.apply(xs));
      return;
    }
    if (!chart || signature !== list.map((s) => `${s.key}:${s.color}`).join("|")) build(list);
    const keys = list.map((s) => s.key);
    const all = chartRows(state, module, keys);
    if (all.length < lastCount) view.restart(); // fewer records than before: a new run
    lastCount = all.length;
    xs = all.map((r) => r.x);
    const v = view.apply(xs);
    const shown = rowsForWindow(all, v.range, smoothWindow * 3);
    const rows = decimate(shown, MAX_DRAWN_ROWS, keys);
    chart.reset();
    for (const r of rows) chart.addPoint(r.x, r.values);
    chart.setViewRange(v.range);
    chart.setReferenceLines(referenceLines(module, state));
    syncControls(v);
  }

  freeze.addEventListener("click", () => { view.toggleFreeze(xs); update(); });
  slider.addEventListener("pointerdown", () => {
    view.setDragging(true, parseInt(slider.max, 10) || 0);
  });
  const release = () => { view.setDragging(false); update(); };
  slider.addEventListener("pointerup", release);
  slider.addEventListener("pointercancel", release);
  slider.addEventListener("change", release);
  slider.addEventListener("input", () => {
    view.pan(parseInt(slider.value, 10) || 0, xs);
    update();
  });
  csv.addEventListener("click", () => {
    if (!state.rows.length) return;
    const blob = new Blob([rowsToCsv(state.rows)], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${module.name || "chart"}-${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  return {
    el: wrap,
    update,
    destroy() { if (chart) chart.destroy(); },
  };
}

/* ---- scalar -------------------------------------------------------------- */

function scalarDisplay({ module, state }) {
  const root = div("bd-scalar");
  const value = div("bd-scalar-value", "—");
  const unit = div("bd-scalar-unit");
  const sub = div("bd-scalar-sub");
  root.append(value, unit, sub);
  return {
    el: root,
    update() {
      const r = scalarOf(module, state);
      if (!r) { value.textContent = "—"; unit.textContent = ""; sub.textContent = "waiting for data"; return; }
      value.textContent = formatValue(r.value);
      unit.textContent = r.series.unit || "";
      const ref = referenceLines(module, state)[0];
      const against = ref ? ` · of ${formatValue(ref.value)}${r.series.unit ? " " + r.series.unit : ""}` : "";
      sub.textContent = r.stale
        ? `last seen at ${module.x === "t" ? formatDuration(r.x.t) : "step " + r.x.step}${against}`
        : `${r.series.label}${against}`;
      root.classList.toggle("is-stale", r.stale);
    },
    destroy() {},
  };
}

/* ---- bars ---------------------------------------------------------------- */

function barsDisplay({ module, state }) {
  const root = div("bd-bars");
  return {
    el: root,
    update() {
      const items = barsOf(module, state);
      root.replaceChildren();
      if (!items.length) { root.append(div("bd-empty", "waiting for data")); return; }
      const max = Math.max(...items.map((i) => i.value), 1e-9);
      for (const it of items) {
        const row = div("bd-bar-row");
        const label = div("bd-bar-label", it.label);
        const track = div("bd-bar-track");
        const fill = div("bd-bar-fill");
        fill.style.width = `${Math.max(0, (it.value / max) * 100)}%`;
        fill.style.background = it.color;
        track.append(fill);
        const val = div("bd-bar-value", `${formatValue(it.value)}${it.unit ? " " + it.unit : ""}`);
        row.append(label, track, val);
        root.append(row);
      }
    },
    destroy() {},
  };
}

/* ---- text ---------------------------------------------------------------- */

function textDisplay({ module, state }) {
  const root = div("bd-text", "—");
  return {
    el: root,
    update() {
      const t = textOf(module, state);
      root.textContent = t === null ? "—" : t;
    },
    destroy() {},
  };
}

/* ---- progress ------------------------------------------------------------ */

function progressDisplay({ module, state }) {
  const root = div("bd-progress");
  const head = div("bd-progress-head");
  const left = div("bd-progress-count", "—");
  const right = div("bd-progress-eta");
  head.append(left, right);
  const track = div("bd-progress-track");
  const fill = div("bd-progress-fill");
  track.append(fill);
  root.append(head, track);
  return {
    el: root,
    update() {
      const p = progressOf(module, state);
      if (!p) { left.textContent = "waiting for data"; right.textContent = ""; fill.style.width = "0%"; return; }
      const pct = p.fraction === null ? null : Math.round(p.fraction * 1000) / 10;
      left.textContent = p.total
        ? `${formatValue(p.value)} / ${formatValue(p.total)}${pct === null ? "" : ` · ${pct}%`}`
        : formatValue(p.value);
      const bits = [];
      if (p.rate) bits.push(`${p.rate >= 10 ? p.rate.toFixed(0) : p.rate.toFixed(2)}/s`);
      if (p.done) bits.push(state.runEnd.cancelled ? "cancelled" : "finished");
      else if (p.eta !== null) bits.push(`ETA ${formatDuration(p.eta)}`);
      right.textContent = bits.join(" · ");
      fill.style.width = `${p.fraction === null ? 0 : p.fraction * 100}%`;
      root.classList.toggle("is-done", p.done);
    },
    destroy() {},
  };
}

/* ---- graph run status ---------------------------------------------------- */

/* The run states the backend actually has (domain/value_objects.GraphStatus):
   queued, running, finished, error, stopped. */
const STATUS_CLASS = {
  queued: "status-idle", running: "status-running", finished: "status-completed",
  error: "status-failed", stopped: "status-cancelled",
};

/** The run's state, from `ctx.execution()` -- polled by the page, since a
    run's lifecycle is persisted state rather than a monitor stream:
    `{execution_id, status, started_at, finished_at, error, done, total}`. */
function statusDisplay({ ctx }) {
  const root = div("bd-status");
  const pill = document.createElement("span");
  pill.className = "status-badge status-idle";
  pill.textContent = "no run yet";
  const line = div("bd-status-line");
  const err = div("bd-status-error");
  err.hidden = true;
  const top = div("bd-status-top");
  top.append(pill);
  root.append(top, line, err);
  return {
    el: root,
    update() {
      const e = ctx && ctx.execution ? ctx.execution() : null;
      if (!e) {
        pill.className = "status-badge status-idle";
        pill.textContent = "no run yet";
        line.textContent = "Press Run to start.";
        err.hidden = true;
        return;
      }
      pill.className = `status-badge ${STATUS_CLASS[e.status] || "status-idle"}`;
      pill.textContent = e.status;
      const bits = [`#${e.execution_id}`];
      if (e.started_at) {
        const end = e.finished_at ? Date.parse(e.finished_at) : Date.now();
        bits.push(formatDuration((end - Date.parse(e.started_at)) / 1000));
      }
      if (e.total) bits.push(`${e.done || 0} / ${e.total} nodes finished`);
      line.textContent = bits.join(" · ");
      err.hidden = !e.error;
      err.textContent = e.error || "";
    },
    destroy() {},
  };
}

/* ---- dispatch ------------------------------------------------------------ */

const KINDS = {
  chart: chartDisplay, scalar: scalarDisplay, bars: barsDisplay,
  text: textDisplay, progress: progressDisplay, status: statusDisplay,
};

export const KNOWN_KINDS = Object.keys(KINDS);

/** A display for `module` (a descriptor from /templates/modules), reading
    `state`. An unknown kind gets a placeholder that says so. */
export function createDisplay({ module, state, ctx }) {
  const make = KINDS[module.kind];
  if (!make) {
    const el = div("bd-empty", `This page does not know how to draw a "${module.kind}" view.`);
    return { el, update() {}, destroy() {} };
  }
  return make({ module, state, ctx });
}
