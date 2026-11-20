/* ---------------------------------------------------------------------------
   board/runs.js -- the two cards that are about *runs* rather than a node: the history of the runs this app started
   (with "Run again") and the runs waiting for memory (each can be cancelled).

   Both read the page's `ctx`, which polls the executions API (the lifecycle is persisted state, not a monitor stream):
     ctx.runs()          -> {history: [summary], planned: [summary]} or null before the first answer
     ctx.runAgain(run)   -> put the run's values back on the page
     ctx.cancelRun(id)   -> stop a waiting run
   A summary is the executions API's: {execution_id, status, name, started_at, finished_at, created_at, origin}.
   --------------------------------------------------------------------------- */

import { formatDuration } from "./series.js";

const STATUS_CLASS = {
  planned: "status-idle", queued: "status-idle", running: "status-running", finished: "status-completed",
  error: "status-failed", stopped: "status-cancelled",
};
const SHOWN = 30;
const ACTIVE = new Set(["planned", "queued", "running"]);

const h = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

/* ---- pure ------------------------------------------------------------------ */

const shorten = (text, n = 28) => (text.length > n ? `${text.slice(0, n - 1)}…` : text);

function showValue(v) {
  if (v === null || v === undefined) return "default";
  if (typeof v === "string") return shorten(v);
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4)));
  return shorten(JSON.stringify(v));
}

/** `{nodes: {uid: {port: v}}, memory: {key: v}}` as "steps 30, cfg 5, seed 7 +2": the first few, so a row says what was different. */
export function describeValues(values, { max = 4 } = {}) {
  const pairs = [];
  for (const ports of Object.values((values && values.nodes) || {})) {
    for (const [port, v] of Object.entries(ports || {})) pairs.push([port.replace(/_/g, " "), v]);
  }
  for (const [key, v] of Object.entries((values && values.memory) || {})) pairs.push([key.replace(/_/g, " "), v]);
  if (!pairs.length) return "";
  const shown = pairs.slice(0, max).map(([k, v]) => `${k} ${showValue(v)}`);
  return pairs.length > max ? `${shown.join(", ")} +${pairs.length - max}` : shown.join(", ");
}

/** Seconds a run took (or has taken), null while it has not started. */
export function runSeconds(run, now = Date.now()) {
  if (!run.started_at) return null;
  const end = run.finished_at ? Date.parse(run.finished_at) : now;
  return Math.max(0, (end - Date.parse(run.started_at)) / 1000);
}

/** "14:03" today, "Nov 19 14:03" otherwise. */
export function whenText(iso, now = new Date()) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return d.toDateString() === now.toDateString() ? time : `${d.toLocaleDateString([], { month: "short", day: "numeric" })} ${time}`;
}

const label = (run) => run.name || `#${run.execution_id}`;
const hasValues = (run) => !!(run.origin && run.origin.values && (run.origin.values.nodes || run.origin.values.memory));

/* ---- history --------------------------------------------------------------- */

export class HistoryCard {
  constructor({ ctx }) {
    this.ctx = ctx || {};
    this.el = h("div", "bd-runs");
    this.shown = null;
    this.clocks = new Map(); // execution id -> span with the running time of a live run
  }

  update() {
    if (!this.ctx.runs) { this._only(); return; }
    const data = this.ctx.runs();
    const runs = ((data && data.history) || []).slice(0, SHOWN);
    const signature = JSON.stringify(runs.map((r) => [r.execution_id, r.status, r.name, r.finished_at, !!this.ctx.canRunAgain && this.ctx.canRunAgain()]));
    if (signature !== this.shown) { this.shown = signature; this._draw(runs, data === null); }
    for (const [id, span] of this.clocks) {
      const run = runs.find((r) => r.execution_id === id);
      if (run) span.textContent = formatDuration(runSeconds(run));
    }
  }

  _only() {
    if (this.shown === "only") return;
    this.shown = "only";
    this.el.replaceChildren(h("div", "bd-empty", "Runs are listed on an app's page"));
  }

  _draw(runs, loading) {
    this.el.replaceChildren();
    this.clocks.clear();
    if (!runs.length) { this.el.append(h("div", "bd-empty", loading ? "Loading…" : "No runs of this app yet")); return; }
    const canAgain = this.ctx.canRunAgain ? this.ctx.canRunAgain() : true;
    for (const run of runs) {
      const row = h("div", "bd-run");
      const pill = h("span", `status-badge ${STATUS_CLASS[run.status] || "status-idle"}`, run.status);
      const main = h("div", "bd-run-main");
      const head = h("div", "bd-run-head");
      head.append(h("span", "bd-run-id", label(run)), h("span", "bd-run-when", whenText(run.started_at || run.created_at)));
      const seconds = runSeconds(run);
      if (seconds !== null) {
        const clock = h("span", "bd-run-time", formatDuration(seconds));
        head.append(clock);
        if (ACTIVE.has(run.status)) this.clocks.set(run.execution_id, clock);
      }
      main.append(head);
      const values = describeValues(run.origin && run.origin.values);
      const failed = run.status === "error" && run.error;
      const detail = failed ? run.error : values;
      if (detail) {
        const line = h("div", failed ? "bd-run-values is-error" : "bd-run-values", detail);
        line.title = failed ? run.error : JSON.stringify(run.origin.values, null, 1);
        main.append(line);
      }
      const again = h("button", "btn btn-secondary bd-run-again", "Run again");
      again.type = "button";
      again.disabled = !hasValues(run) || !canAgain;
      again.title = !hasValues(run) ? "This run recorded no values" : canAgain ? "Put this run's values back on the page" : "Not while editing the layout";
      again.addEventListener("click", () => { if (this.ctx.runAgain) this.ctx.runAgain(run); });
      row.append(pill, main, again);
      this.el.append(row);
    }
  }

  destroy() { this.clocks.clear(); }
}

/* ---- queue ----------------------------------------------------------------- */

export class QueueCard {
  constructor({ ctx }) {
    this.ctx = ctx || {};
    this.el = h("div", "bd-runs");
    this.shown = null;
    this.busy = new Set();
  }

  update() {
    if (!this.ctx.runs) {
      if (this.shown !== "only") { this.shown = "only"; this.el.replaceChildren(h("div", "bd-empty", "Waiting runs are listed on an app's page")); }
      return;
    }
    const planned = (this.ctx.runs() || {}).planned || [];
    const signature = JSON.stringify(planned.map((r) => [r.execution_id, r.name, [...this.busy].includes(r.execution_id)]));
    if (signature === this.shown) return;
    this.shown = signature;
    this.el.replaceChildren();
    if (!planned.length) { this.el.append(h("div", "bd-empty", "Nothing is waiting")); return; }
    planned.forEach((run, index) => {
      const row = h("div", "bd-run");
      const main = h("div", "bd-run-main");
      const head = h("div", "bd-run-head");
      head.append(h("span", "bd-run-id", `${index + 1}. ${label(run)}`), h("span", "bd-run-when", whenText(run.created_at)));
      main.append(head);
      const what = [run.origin && run.origin.app, describeValues(run.origin && run.origin.values)].filter(Boolean).join(" · ");
      if (what) main.append(h("div", "bd-run-values", what));
      const cancel = h("button", "btn btn-secondary bd-run-again", "Cancel");
      cancel.type = "button";
      cancel.disabled = this.busy.has(run.execution_id);
      cancel.addEventListener("click", async () => {
        this.busy.add(run.execution_id);
        this.shown = null;
        this.update();
        try { if (this.ctx.cancelRun) await this.ctx.cancelRun(run.execution_id); } finally { this.busy.delete(run.execution_id); this.shown = null; this.update(); }
      });
      row.append(main, cancel);
      this.el.append(row);
    });
  }

  destroy() {}
}
