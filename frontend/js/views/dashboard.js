/* ---------------------------------------------------------------------------
   dashboard.js -- training controls against the new backend (M6).

   Data flow: REST for state that is authoritative in the DB (active run,
   history, logs, start options), SSE /events for live progress -- polling
   is only the fallback refresh after actions, plus a 30s safety poll and a
   refetch on every (re)open, because /events has no replay (docs 07 F-09).

   The page is state-driven: the hero shows exactly one of two faces
   (live run / start form), so idle never shows empty metrics and
   running never shows a start form that would 409.

   A diverged loss (NaN/Inf) arrives as null plus a `nonfinite` marker and
   is rendered loud, never as an em dash (docs 07 F-03). An unparsable
   frame is counted in the console, never dropped quietly.

   Everything network-shaped goes through api.js, so the error envelope
   surfaces in exactly one place (the console log below).
   --------------------------------------------------------------------------- */

import { api, sse, ApiError } from "../api.js";
import { renderMeasured } from "../lib/value.js";

const el = (id) => document.getElementById(id);

/* ---- system console: capped, auto-scrolled, one entry per event ---- */

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
  if (err instanceof ApiError) {
    log(`${err.code}: ${err.message}`, "error");
  } else {
    log(String(err && err.message ? err.message : err), "error");
  }
}

/* ---- state ---- */

let activeRun = null;     // RunOut | null
let historyRuns = [];     // newest first
let shownLogRun = null;   // run id whose log is on screen (or null)
let elapsedTimer = null;  // 1s ticker, only while a run is active
let safetyTimer = null;   // slow resync, so a missed event cannot stick (F-09)
let badFrames = 0;        // frames we could not parse -- counted, shown

/* /events has no replay: a frame sent while the tab was asleep, while
   the socket was reconnecting, or before this page subscribed (the
   startup reconcile publishes first) is simply never delivered. So the
   DB stays the source of truth and every (re)open refetches it, with a
   slow poll underneath for a socket that never notices it is stale.
   Without this the hero can sit on "running" for a finished run until
   a manual reload (docs 07 F-09). */
const SAFETY_POLL_MS = 30000;

function resync(reason) {
  refreshActive();
  refreshHistory();
  if (reason) log(reason, "info");
}

function startSafetyPoll() {
  if (safetyTimer !== null) return;
  safetyTimer = setInterval(() => resync(), SAFETY_POLL_MS);
}

/* ---- active run + hero state ---- */

async function refreshActive() {
  try {
    activeRun = await api("/runs/active");
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) activeRun = null;
    else logError(err);
  }
  renderActive();
}

function renderActive() {
  const badge = el("status-badge");
  const running = activeRun !== null;

  // One hero face at a time (html [hidden] handles visibility).
  el("hero-run").hidden = !running;
  el("hero-idle").hidden = running;
  el("btn-stop").hidden = !running;
  el("btn-kill").hidden = !running;

  if (!running) {
    stopTicker();
    badge.className = "status-badge status-idle";
    badge.textContent = "Idle";
    el("run-id").textContent = "#—";
    el("run-phase").textContent = "";
    el("run-elapsed").textContent = "—";
    el("run-meta").textContent = "";
    el("progress-fill").style.width = "0%";
    el("progress-text").textContent = "—";
    el("cache-progress-wrap").classList.remove("active");
    for (const id of ["metric-loss", "metric-avg", "metric-lr"]) {
      renderMeasured(el(id), null, "current_loss", fmtNum);
    }
    return;
  }

  const run = activeRun;
  badge.className = "status-badge status-running";
  badge.textContent = run.phase ? `Running · ${run.phase}` : "Running";
  el("run-id").textContent = `#${run.id}`;
  el("run-phase").textContent = run.phase || "";
  el("run-meta").textContent = [run.config_path, run.mode]
    .filter(Boolean).join(" · ");
  startTicker();
  renderProgress(run);
  renderMeasured(el("metric-loss"), run, "current_loss", fmtNum);
  renderMeasured(el("metric-avg"), run, "avg_loss", fmtNum);
  renderMeasured(el("metric-lr"), run, "lr", () => "—"); // lr arrives via run_progressed
}

function renderProgress(run) {
  const total = run.total_steps || 0;
  const pct = total > 0 ? Math.min(100, (run.done_steps / total) * 100) : 0;
  el("progress-fill").style.width = pct + "%";
  el("progress-text").textContent = total
    ? `${run.done_steps} / ${run.total_steps} · ${Math.round(pct)}%`
    : `${run.done_steps} steps`;

  const wrap = el("cache-progress-wrap");
  if (run.cache_total) {
    wrap.classList.add("active");
    const cachePct = Math.min(100, (run.cache_done / run.cache_total) * 100);
    el("cache-fill").style.width = cachePct + "%";
    el("cache-text").textContent = `cache ${run.cache_done}/${run.cache_total}`;
  } else {
    wrap.classList.remove("active");
  }
}

/* ---- elapsed ticker (started_at -> "1h 04m", tabular, updates each s) ---- */

function startTicker() {
  stopTicker();
  const tick = () => {
    if (!activeRun) return stopTicker();
    el("run-elapsed").textContent = activeRun.started_at
      ? fmtDuration(Date.now() - Date.parse(activeRun.started_at)) : "—";
  };
  tick();
  elapsedTimer = setInterval(tick, 1000);
}

function stopTicker() {
  if (elapsedTimer !== null) {
    clearInterval(elapsedTimer);
    elapsedTimer = null;
  }
}

/* ---- history table (selection drives the log pane) ---- */

async function refreshHistory() {
  try {
    const res = await api("/runs?limit=20");
    historyRuns = res.runs;
    renderHistory();
  } catch (err) {
    logError(err);
  }
}

const STATUS_CLASS = {
  created: "status-idle",
  running: "status-running",
  completed: "status-completed",
  failed: "status-failed",
  cancelled: "status-cancelled",
};

function renderHistory() {
  const list = el("history-list");
  list.replaceChildren();
  el("btn-wipe").disabled = false;

  if (historyRuns.length === 0) {
    el("btn-wipe").disabled = true;
    const tr = document.createElement("tr");
    tr.className = "empty-row";
    const td = document.createElement("td");
    td.className = "empty";
    td.colSpan = 6;
    td.textContent = "No runs yet.";
    tr.appendChild(td);
    list.appendChild(tr);
    renderLastRun();
    return;
  }

  for (const run of historyRuns) {
    const tr = document.createElement("tr");
    tr.dataset.runId = String(run.id);
    if (run.id === shownLogRun) tr.classList.add("selected");
    tr.addEventListener("click", () => showLog(run.id));

    const id = document.createElement("td");
    id.className = "col-id";
    id.textContent = `#${run.id}`;

    const status = document.createElement("td");
    const chip = document.createElement("span");
    chip.className = `status-badge ${STATUS_CLASS[run.status] || "status-idle"}`;
    chip.textContent = run.status;
    status.appendChild(chip);

    const cfg = document.createElement("td");
    cfg.className = "col-config";
    cfg.title = run.config_path;
    cfg.textContent = run.config_path;

    const steps = document.createElement("td");
    steps.className = "col-num";
    steps.textContent = run.total_steps
      ? `${run.done_steps}/${run.total_steps}` : `${run.done_steps}`;

    const loss = document.createElement("td");
    loss.className = "col-num";
    // A finished run whose loss diverged keeps the marker: an em dash
    // in the history would hide exactly the run worth looking at.
    renderMeasured(loss, run, "avg_loss", (v) => (v != null ? fmtNum(v) : "—"));

    const when = document.createElement("td");
    when.className = "col-dim";
    const ts = Date.parse(run.started_at || run.created_at);
    when.textContent = fmtRel(ts);
    when.title = new Date(ts).toLocaleString() +
      (run.finished_at && run.started_at
        ? ` · ${fmtDuration(Date.parse(run.finished_at) - Date.parse(run.started_at))}`
        : "");

    tr.append(id, status, cfg, steps, loss, when);
    list.appendChild(tr);
  }
  renderLastRun();
}

/* Idle hero context: what happened most recently, honest and clickable. */
function renderLastRun() {
  const box = el("last-run");
  const last = historyRuns[0];
  if (!last) {
    box.hidden = true;
    box.replaceChildren();
    return;
  }
  box.hidden = false;
  box.replaceChildren(
    Object.assign(document.createElement("span"), {
      textContent: `Last: #${last.id} ${last.status} · ` +
        (last.total_steps ? `${last.done_steps}/${last.total_steps} steps` : `${last.done_steps} steps`) +
        ` · ${fmtRel(Date.parse(last.started_at || last.created_at))}`,
    }),
    Object.assign(document.createElement("button"), {
      className: "link-btn",
      textContent: "view log",
      type: "button",
      onclick: () => showLog(last.id),
    })
  );
}

function markSelectedRow() {
  for (const tr of el("history-list").querySelectorAll("tr")) {
    tr.classList.toggle("selected",
      shownLogRun !== null && tr.dataset.runId === String(shownLogRun));
  }
}

/* ---- log tail ---- */

async function showLog(runId) {
  try {
    const res = await api(`/runs/${runId}/log?lines=200`);
    shownLogRun = runId;
    el("log-title").textContent = `#${runId}`;
    const openLink = el("log-open");
    openLink.href = `/run/${runId}`;
    openLink.title = `Open run #${runId} detail: full log + all fields`;
    openLink.hidden = false;
    el("log-body").textContent = res.log || "(empty)";
    el("log-body").scrollTop = 0;
    markSelectedRow();
  } catch (err) {
    logError(err);
  }
}

/* ---- start form (inline errors -- the console keeps the record too) ---- */

function setStartError(message) {
  const box = el("start-error");
  box.textContent = message;
  box.hidden = !message;
}

async function startRun(ev) {
  if (ev) ev.preventDefault();
  const path = el("cfg-path").value.trim();
  if (!path) {
    setStartError("Config path is required.");
    el("cfg-path").focus();
    return;
  }
  const body = {
    config_path: path,
    start_from: el("start-from").value || "teacher",
    reset_optimizer: el("reset-optimizer").checked,
  };
  try {
    setStartError("");
    const run = await api("/runs", { method: "POST", body });
    log(`Run #${run.id} launched (${run.config_path}).`, "success");
    await refreshActive();
    await refreshHistory();
  } catch (err) {
    logError(err); // run_already_active / config_invalid / launch failure
    setStartError(err instanceof ApiError
      ? `${err.code}: ${err.message}`
      : String(err && err.message ? err.message : err));
  }
}

async function stopRun(force) {
  if (!activeRun) return;
  if (force && !confirm("Force-kill the training process? The checkpoint may be incomplete.")) return;
  try {
    await api(`/runs/${activeRun.id}/stop`, { method: "POST", body: { force } });
    log(force ? `Kill signal sent to run #${activeRun.id}.`
              : `Stop requested for run #${activeRun.id} (saving checkpoint…).`, "warn");
  } catch (err) {
    logError(err); // run_not_running
  }
  await refreshActive();
  await refreshHistory();
}

async function wipeHistory() {
  if (!confirm("Delete ALL run history? Log files on disk stay untouched.")) return;
  try {
    const res = await api("/runs", { method: "DELETE" });
    log(`Deleted ${res.deleted} run(s).`, "warn");
    await refreshHistory();
    await refreshActive();
  } catch (err) {
    logError(err);
  }
}

/* ---- start options (continue-from picker) ---- */

/* The API requires an explicit config path (empty -> 422
   invalid_query, same as GET /config); until one is entered the
   picker honestly says so instead of pretending options exist. */
async function loadStartOptions() {
  const path = el("cfg-path").value.trim();
  const select = el("start-from");
  if (!path) {
    // selected:true matters -- a lone disabled option without it leaves
    // selectedIndex at -1 and the select renders blank.
    select.replaceChildren(
      Object.assign(document.createElement("option"), {
        value: "", label: "— enter a config path —", disabled: true, selected: true,
      })
    );
    return;
  }
  try {
    const res = await api(`/config/start-options?path=${encodeURIComponent(path)}`);
    select.replaceChildren();
    for (const [value, opt] of Object.entries(res.start_from || {})) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = opt.label || value;
      option.disabled = !opt.available;
      select.appendChild(option);
    }
    if (!res.has_unfinished_run && select.querySelector('option[value="teacher"]')) {
      select.value = "teacher";
    }
  } catch (err) {
    logError(err); // invalid_query / config_not_found / config_invalid
  }
}

/* ---- live progress over the domain-event stream ---- */

/* A frame we cannot parse is data we cannot show. It is counted and
   said out loud, never swallowed: a silent drop looks exactly like a
   run that stopped reporting (docs 07 F-03, review rule 5). */
function noteBadFrame() {
  badFrames += 1;
  log(`Unreadable event frame dropped (${badFrames} so far).`, "error");
}

function handleEvent(raw) {
  let e;
  try { e = JSON.parse(raw.data); } catch { noteBadFrame(); return; }
  switch (e.type) {
    case "run_started":
      log(`Run #${e.run_id} started.`, "success");
      resync();
      break;
    case "run_progressed": {
      if (!activeRun || e.run_id !== activeRun.id) break;
      // Patch the hero in place -- DB state is refreshed on transitions
      // and on every resync, so a missed frame self-heals.
      activeRun.done_steps = e.step;
      activeRun.total_steps = e.total_steps;
      activeRun.current_loss = e.loss;
      activeRun.avg_loss = e.avg_loss;
      activeRun.cache_done = e.cache_done;
      activeRun.cache_total = e.cache_total;
      if (e.nonfinite) activeRun.nonfinite = e.nonfinite;
      else delete activeRun.nonfinite;
      renderProgress(activeRun);
      renderMeasured(el("metric-loss"), e, "loss", fmtNum);
      renderMeasured(el("metric-avg"), e, "avg_loss", fmtNum);
      renderMeasured(el("metric-lr"), e, "lr",
        (v) => (v != null ? v.toExponential(2) : "—"));
      break;
    }
    case "run_completed":
      log(`Run #${e.run_id} completed (${e.done_steps} steps).`, "success");
      afterRunEnd(e.run_id);
      break;
    case "run_failed":
      log(`Run #${e.run_id} failed: ${e.error || "unknown error"}`, "error");
      afterRunEnd(e.run_id);
      break;
    case "run_cancelled":
      log(`Run #${e.run_id} cancelled.`, "warn");
      afterRunEnd(e.run_id);
      break;
    case "runs_deleted":
      refreshHistory();
      break;
    default:
      break; // graph/dataset events: M7/M8 views consume them
  }
}

async function afterRunEnd(runId) {
  await refreshActive();
  await refreshHistory();
  // Show the finished run's tail unless the user is inspecting another.
  if (shownLogRun === null || shownLogRun === runId) await showLog(runId);
}

/* ---- monitor hand-off ---- */

function openMonitor() {
  const id = el("monitor-id-input").value.trim();
  if (!id) {
    log("Paste a monitor id first (it is in the monitor page URL).", "warn");
    el("monitor-id-input").focus();
    return;
  }
  window.location.href = `/monitor/${encodeURIComponent(id)}`;
}

/* ---- boot ---- */

async function boot() {
  el("start-form").addEventListener("submit", startRun);
  el("btn-stop").addEventListener("click", () => stopRun(false));
  el("btn-kill").addEventListener("click", () => stopRun(true));
  el("btn-wipe").addEventListener("click", wipeHistory);
  el("btn-refresh-log").addEventListener("click", () => {
    if (shownLogRun !== null) showLog(shownLogRun);
    else log("No run selected -- click a row in the history.", "warn");
  });
  el("btn-open-monitor").addEventListener("click", openMonitor);
  el("monitor-id-input").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") openMonitor();
  });
  el("cfg-path").addEventListener("change", loadStartOptions);
  el("cfg-path").addEventListener("input", () => setStartError(""));

  // Prefill the default config from settings when there is one.
  try {
    const settings = await api("/settings");
    const stored = settings.stored || {};
    if (!el("cfg-path").value && stored.default_config) {
      el("cfg-path").value = stored.default_config;
    }
  } catch (err) {
    logError(err);
  }

  await loadStartOptions();
  await refreshActive();
  await refreshHistory();
  if (activeRun) await showLog(activeRun.id);

  // onOpen fires on every (re)connect, so the refetch is also the
  // resync after a dropped connection or a backgrounded tab.
  sse("/events", {
    onMessage: handleEvent,
    onOpen: () => { startSafetyPoll(); resync(); },
    onError: () => log("Event stream reconnecting…", "warn"),
  });
  log("Connected to /api/v1/events.", "info");
}

/* ---- formatting ---- */

function fmtNum(v) {
  if (v === undefined || v === null) return "—";
  return v >= 1 ? v.toFixed(4) : v >= 0.001 ? v.toFixed(5) : v.toExponential(2);
}

function fmtDuration(ms) {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ${String(m % 60).padStart(2, "0")}m`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
}

function fmtRel(ts) {
  if (!Number.isFinite(ts)) return "—";
  const s = Math.round((Date.now() - ts) / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

boot().catch(logError);
