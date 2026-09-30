/* ---------------------------------------------------------------------------
   dashboard.js -- training controls against the new backend (M6).

   Data flow: REST for state that is authoritative in the DB (active run,
   history, logs, start options), SSE /events for live progress -- polling
   is only the fallback refresh after actions (03-migration §2 rules).

   Everything network-shaped goes through api.js, so the error envelope
   surfaces in exactly one place (the console log below).
   --------------------------------------------------------------------------- */

import { api, sse, ApiError } from "../api.js";

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

let activeRun = null;    // RunOut | null
let historyRuns = [];    // newest first
let shownLogRun = null;  // run id whose log is on screen (or null)

/* ---- active run + controls ---- */

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
  el("btn-start").style.display = running ? "none" : "flex";
  el("btn-stop").hidden = !running;
  el("btn-kill").hidden = !running;

  if (!running) {
    badge.className = "status-badge status-idle";
    badge.textContent = "Idle";
    el("progress-fill").style.width = "0%";
    el("progress-text").textContent = "no active run";
    el("cache-progress-wrap").classList.remove("active");
    for (const id of ["metric-loss", "metric-avg", "metric-lr", "metric-step"]) {
      el(id).textContent = "—";
    }
    return;
  }

  const run = activeRun;
  badge.className = "status-badge status-running";
  badge.textContent = run.phase ? `Running · ${run.phase}` : "Running";
  el("btn-start").style.display = "none";
  el("btn-stop").hidden = false;
  el("btn-kill").hidden = false;
  renderProgress(run);
  el("metric-loss").textContent = fmtNum(run.current_loss);
  el("metric-avg").textContent = fmtNum(run.avg_loss);
  el("metric-lr").textContent = "—"; // lr arrives via run_progressed
  el("metric-step").textContent = run.total_steps
    ? `${run.done_steps} / ${run.total_steps}` : String(run.done_steps);
}

function renderProgress(run) {
  const total = run.total_steps || 0;
  const pct = total > 0 ? Math.min(100, (run.done_steps / total) * 100) : 0;
  el("progress-fill").style.width = pct + "%";
  el("progress-text").textContent = total
    ? `${run.done_steps} / ${run.total_steps} steps`
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

/* ---- history ---- */

async function refreshHistory() {
  try {
    const res = await api("/runs?limit=20");
    historyRuns = res.runs;
    renderHistory();
  } catch (err) {
    logError(err);
  }
}

function renderHistory() {
  const list = el("history-list");
  list.replaceChildren();
  if (historyRuns.length === 0) {
    const empty = document.createElement("div");
    empty.className = "console-line info";
    empty.textContent = "No runs yet.";
    list.appendChild(empty);
    return;
  }
  for (const run of historyRuns) {
    const row = document.createElement("div");
    row.className = "history-item";

    const id = document.createElement("span");
    id.style.fontFamily = "'JetBrains Mono', monospace";
    id.textContent = `#${run.id}`;

    const status = document.createElement("span");
    status.className = `status-badge status-${run.status === "running" ? "running"
      : run.status === "failed" ? "failed" : "idle"}`;
    status.textContent = run.status;

    const cfg = document.createElement("span");
    cfg.style.flex = "1";
    cfg.style.overflow = "hidden";
    cfg.style.textOverflow = "ellipsis";
    cfg.style.whiteSpace = "nowrap";
    cfg.textContent = run.config_path;

    const steps = document.createElement("span");
    steps.style.fontFamily = "'JetBrains Mono', monospace";
    steps.textContent = run.total_steps
      ? `${run.done_steps}/${run.total_steps}` : `${run.done_steps}`;

    const loss = document.createElement("span");
    loss.style.fontFamily = "'JetBrains Mono', monospace";
    loss.textContent = run.avg_loss != null ? `avg ${fmtNum(run.avg_loss)}` : "";

    const btn = document.createElement("button");
    btn.className = "btn btn-secondary btn-small";
    btn.textContent = "Log";
    btn.addEventListener("click", () => showLog(run.id));

    row.append(id, status, cfg, steps, loss, btn);
    list.appendChild(row);
  }
}

/* ---- log tail ---- */

async function showLog(runId) {
  try {
    const res = await api(`/runs/${runId}/log?lines=200`);
    shownLogRun = runId;
    el("log-title").textContent = `#${runId}`;
    el("log-body").textContent = res.log || "(empty)";
    el("log-body").scrollTop = 0;
  } catch (err) {
    logError(err);
  }
}

/* ---- actions ---- */

async function startRun() {
  const path = el("cfg-path").value.trim();
  if (!path) {
    log("Config path is required.", "warn");
    el("cfg-path").focus();
    return;
  }
  const body = {
    config_path: path,
    start_from: el("start-from").value || "teacher",
    reset_optimizer: el("reset-optimizer").checked,
  };
  try {
    const run = await api("/runs", { method: "POST", body });
    log(`Run #${run.id} launched (${run.config_path}).`, "success");
    await refreshActive();
    await refreshHistory();
  } catch (err) {
    logError(err); // run_already_active / config_invalid / launch failure
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
    select.replaceChildren(
      Object.assign(document.createElement("option"), {
        value: "", label: "— enter a config path —", disabled: true,
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

function handleEvent(raw) {
  let e;
  try { e = JSON.parse(raw.data); } catch { return; }
  switch (e.type) {
    case "run_started":
      log(`Run #${e.run_id} started.`, "success");
      refreshActive();
      refreshHistory();
      break;
    case "run_progressed": {
      if (!activeRun || e.run_id !== activeRun.id) break;
      // Patch the card in place -- DB state is refreshed on transitions.
      activeRun.done_steps = e.step;
      activeRun.total_steps = e.total_steps;
      activeRun.current_loss = e.loss;
      activeRun.avg_loss = e.avg_loss;
      activeRun.cache_done = e.cache_done;
      activeRun.cache_total = e.cache_total;
      renderProgress(activeRun);
      el("metric-loss").textContent = fmtNum(e.loss);
      el("metric-avg").textContent = fmtNum(e.avg_loss);
      el("metric-lr").textContent = e.lr != null ? e.lr.toExponential(2) : "—";
      el("metric-step").textContent = e.total_steps
        ? `${e.step} / ${e.total_steps}` : String(e.step);
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
  el("btn-start").addEventListener("click", startRun);
  el("btn-stop").addEventListener("click", () => stopRun(false));
  el("btn-kill").addEventListener("click", () => stopRun(true));
  el("btn-wipe").addEventListener("click", wipeHistory);
  el("btn-refresh-log").addEventListener("click", () => {
    if (shownLogRun !== null) showLog(shownLogRun);
    else log("No run selected -- click Log on a history row.", "warn");
  });
  el("btn-open-monitor").addEventListener("click", openMonitor);
  el("monitor-id-input").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") openMonitor();
  });
  el("cfg-path").addEventListener("change", loadStartOptions);

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

  sse("/events", { onMessage: handleEvent });
  log("Connected to /api/v1/events.", "info");
}

function fmtNum(v) {
  if (v === undefined || v === null) return "—";
  return v >= 1 ? v.toFixed(4) : v >= 0.001 ? v.toFixed(5) : v.toExponential(2);
}

boot().catch(logError);
