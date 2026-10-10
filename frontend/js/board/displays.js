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
  referenceLines, scalarOf, seriesFor, textOf,
} from "./series.js";

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
  const root = div("bd-chart");
  const canvas = document.createElement("canvas");
  canvas.className = "bd-canvas";
  root.append(canvas);
  let chart = null;
  let signature = "";
  const hidden = new Map(); // series key -> hidden, survives a rebuild

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
      smoothWindow: module.scale === "linear" ? 3 : 24,
      xLabel: module.x === "t" ? "t (s)" : "Step",
    });
    signature = list.map((s) => `${s.key}:${s.color}`).join("|");
  };

  return {
    el: root,
    update() {
      const list = seriesFor(module, state);
      if (!list.length) return;
      if (!chart || signature !== list.map((s) => `${s.key}:${s.color}`).join("|")) build(list);
      const keys = list.map((s) => s.key);
      const rows = decimate(chartRows(state, module, keys), MAX_DRAWN_ROWS, keys);
      chart.reset();
      for (const r of rows) chart.addPoint(r.x, r.values);
      chart.setReferenceLines(referenceLines(module, state));
    },
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
