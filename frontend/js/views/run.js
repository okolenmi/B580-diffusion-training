/* ---------------------------------------------------------------------------
   run.js -- Run detail page entry (M8b).

   Reads the run id from the URL (/run/{id}), renders the full RunOut
   as a details grid, and tails the log (last 500 lines -- the API
   cap). While the run is active a 5s interval refreshes the log and a
   1s ticker keeps the elapsed time honest; terminal events over SSE
   trigger one final reload.

   404 (run never existed or was wiped) renders as an honest state,
   not an exception.
   --------------------------------------------------------------------------- */

import { api, ApiError } from "../api.js";
import { subscribeEvents, startSafetyPoll } from "../lib/events.js";
import { fmtDuration, fmtTime } from "../lib/format.js";

const el = (id) => document.getElementById(id);

function log(message, kind = "info") {
  const out = el("console-output");
  const line = document.createElement("div");
  line.className = `console-line ${kind}`;
  line.textContent = message;
  out.appendChild(line);
  while (out.children.length > 60) out.removeChild(out.firstChild);
  out.scrollTop = out.scrollHeight;
}

function logError(err) {
  if (err instanceof ApiError) log(`${err.code}: ${err.message}`, "error");
  else log(String(err && err.message ? err.message : err), "error");
}

/* ---- state ---- */

const runId = Number(location.pathname.split("/").filter(Boolean).pop());
let run = null;          // RunOut
let logTimer = null;     // 5s log refresh while running
let elapsedTimer = null; // 1s duration ticker while running

const STATUS_CLASS = {
  created: "status-idle",
  running: "status-running",
  completed: "status-completed",
  failed: "status-failed",
  cancelled: "status-cancelled",
};

/* ---- rendering ---- */

const dash = (v) => (v === null || v === undefined ? "—" : String(v));

function addRow(dl, label, value, cls) {
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value;
  if (cls) dd.className = cls;
  dl.append(dt, dd);
}

function render() {
  const badge = el("status-badge");
  badge.className = `status-badge ${STATUS_CLASS[run.status] || "status-idle"}`;
  badge.textContent = run.status;
  el("run-title").textContent = `Run #${run.id}`;
  document.title = `Run #${run.id}`;
  el("run-sub").textContent = run.config_path;

  const dl = el("run-details");
  dl.replaceChildren();
  addRow(dl, "Status", run.status);
  addRow(dl, "Mode", dash(run.mode), "mono");
  addRow(dl, "Phase", dash(run.phase));
  addRow(dl, "Config", run.config_path, "mono");
  addRow(dl, "Steps",
    run.total_steps ? `${run.done_steps} / ${run.total_steps}` : `${run.done_steps}`,
    "mono");
  addRow(dl, "Avg loss", dash(run.avg_loss), "mono");
  addRow(dl, "Current loss", dash(run.current_loss), "mono");
  addRow(dl, "Cache",
    run.cache_total ? `${run.cache_done} / ${run.cache_total}` : "—", "mono");
  addRow(dl, "PID", dash(run.pid), "mono");
  addRow(dl, "Exit code", dash(run.exit_code), "mono");
  addRow(dl, "Error", dash(run.error), run.error ? "error" : "mono");
  addRow(dl, "Log file", dash(run.log_path), "mono");
  addRow(dl, "Created", fmtTime(run.created_at), "mono");
  addRow(dl, "Started", fmtTime(run.started_at), "mono");
  addRow(dl, "Finished", fmtTime(run.finished_at), "mono");
  addRow(dl, "Duration", durationText(), "mono");

  el("run-state").hidden = true;
  el("run-grid").hidden = false;
}

function durationText() {
  if (!run) return "—";
  if (!run.started_at) return "—";
  const end = run.finished_at ? Date.parse(run.finished_at) : Date.now();
  return fmtDuration(end - Date.parse(run.started_at));
}

function patchDuration() {
  // dt/dd are appended in pairs, so index alignment holds for the
  // unfiltered NodeList (a .mono-filtered list would skew it)
  const dts = [...el("run-details").querySelectorAll("dt")];
  const dds = el("run-details").querySelectorAll("dd");
  const idx = dts.findIndex((dt) => dt.textContent === "Duration");
  if (idx >= 0 && dds[idx]) dds[idx].textContent = durationText();
}

/* ---- data ---- */

async function loadRun() {
  try {
    run = await api(`/runs/${runId}`);
    render();
    syncTimers();
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      el("run-state").textContent =
        `Run #${runId} was not found -- it may have been wiped from the history.`;
    } else {
      logError(err);
      el("run-state").textContent =
        err instanceof ApiError ? `${err.code}: ${err.message}` : String(err);
    }
  }
}

async function loadLog() {
  if (!run) return;
  try {
    const res = await api(`/runs/${runId}/log?lines=500`);
    el("log-body").textContent = res.log || "";
  } catch (err) {
    logError(err);
  }
}

function syncTimers() {
  const active = run && run.status === "running";
  if (active && logTimer === null) {
    logTimer = setInterval(loadLog, 5000);
  } else if (!active && logTimer !== null) {
    clearInterval(logTimer);
    logTimer = null;
  }
  if (active && elapsedTimer === null) {
    elapsedTimer = setInterval(patchDuration, 1000);
  } else if (!active && elapsedTimer !== null) {
    clearInterval(elapsedTimer);
    elapsedTimer = null;
  }
}

/* ---- live updates ---- */

function handleEvent(e) {
  if (e.type === "run_progressed" && e.run_id === runId && run) {
    run.done_steps = e.step;
    run.total_steps = e.total_steps;
    run.current_loss = e.loss;
    run.avg_loss = e.avg_loss;
    run.cache_done = e.cache_done;
    run.cache_total = e.cache_total;
    // Carry the marker: the server sends a diverged loss as `null` plus
    // `nonfinite: {loss: "nan"}`. Without copying it across, render()
    // formats the null and the page shows an em dash for the exact run
    // a user opens this page to investigate (docs 08 N-06).
    if (e.nonfinite) run.nonfinite = e.nonfinite;
    else delete run.nonfinite;
    render();
    syncTimers();
    return;
  }
  if (["run_completed", "run_failed", "run_cancelled"].includes(e.type)
      && e.run_id === runId) {
    log(`Run #${runId} reached terminal state -- reloading.`, "info");
    loadRun().then(loadLog);
  }
}

/* ---- boot ---- */

async function boot() {
  if (!Number.isInteger(runId) || runId < 1) {
    el("run-state").textContent = "Invalid run id in the URL.";
    return;
  }
  el("btn-refresh-log").addEventListener("click", loadLog);

  await loadRun();
  await loadLog();
  // onResync runs on every open, first included: /events has no replay,
  // so anything published before this page subscribed is missing, and
  // without this a missed run_completed leaves the page on "running".
  subscribeEvents({
    onEvent: handleEvent,
    onResync: () => loadRun().then(loadLog),
    onNotice: (m) => log(m, "error"),
    onError: (m) => log(m, "warn"),
  });
  // And a slow poll for the case where nothing happens at all: no error,
  // no reconnect, just a frame that never arrived.
  startSafetyPoll(() => loadRun(), 30000);
}

boot().catch(logError);
