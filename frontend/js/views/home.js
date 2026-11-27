/* ---------------------------------------------------------------------------
   views/home.js -- the start page: how the machine stands, what is left to set up, and the latest runs.

   Every source is asked separately and may fail on its own; a tile whose answer is missing says so instead of the page going blank.
   The PyTorch check imports torch in the chosen environment (seconds), so it runs once, after the page is already showing.
   --------------------------------------------------------------------------- */

import { api } from "../api.js";
import { h, mount } from "../lib/dom.js";
import { checkEnvironment } from "../lib/environment_check.js";
import { errText } from "../lib/errors.js";
import { fmtDuration, fmtRel } from "../lib/format.js";
import { checklist, memoryView, nodeHealth, remaining } from "../lib/home_model.js";
import { logError } from "../lib/log.js";

const root = document.getElementById("home-root");

const data = {
  health: null, settings: null, catalog: null, runs: null, datasets: null, checkpoints: null, tokenizers: null,
  env: /** @type {Awaited<ReturnType<typeof checkEnvironment>>|null} */ (null),
};

const BADGE = { finished: "completed", error: "failed", stopped: "cancelled", running: "running", queued: "idle", planned: "idle" };
const gb = (mb) => `${(mb / 1000).toFixed(1)} GB`;
const quiet = (promise) => promise.catch((err) => { logError(err); return null; });

/* ---- tiles --------------------------------------------------------------- */

function tile(label, state, value, sub, link) {
  return h("div", { class: "tile" },
    h("div", { class: "tile-label" }, h("span", { class: `dot dot-${state}`, "aria-hidden": "true" }), label),
    h("div", { class: "tile-value" }, value),
    sub ? h("div", { class: "tile-sub" }, sub) : null,
    link ? h("div", { class: "tile-sub" }, h("a", { href: link[0] }, link[1])) : null);
}

function serverTile() {
  const health = data.health;
  if (!health) return tile("Server", "bad", "Not answering", "The page could not read the server's health.");
  const since = health.started_at ? `up ${fmtRel(Date.parse(health.started_at)).replace(" ago", "")}` : "";
  return tile("Server", "ok", `Running ${health.version}`, `Python ${health.python || "?"}${since ? `, ${since}` : ""}`);
}

function cardTile() {
  const env = data.env;
  if (!env) return tile("Graphics card", "busy", "Checking…", "PyTorch is being asked in the chosen environment.");
  if (env.present) {
    return tile("Graphics card", "ok", env.name, `${Math.round(env.totalMemoryMb).toLocaleString()} MB, seen by PyTorch`);
  }
  return tile("Graphics card", "warn", "Not seen by PyTorch", env.reason, ["/settings#environment", "Open the environment check"]);
}

function environmentTile() {
  const env = data.env;
  if (!env) return tile("PyTorch", "busy", "Checking…", null);
  if (!env.folder) return tile("PyTorch", "warn", "No environment chosen", env.reason, ["/settings#environment", "Choose one"]);
  if (!env.torchVersion) return tile("PyTorch", "bad", "Not found", env.reason, ["/settings#environment", "Open the environment check"]);
  if (env.serverInEnvironment === false) {
    return tile("PyTorch", "warn", env.torchVersion, "The server does not run in this environment, so its nodes cannot load it. Restart the server with the launcher.",
      ["/settings#environment", "Details"]);
  }
  return tile("PyTorch", "ok", env.torchVersion, env.folder);
}

function nodesTile() {
  const nodes = nodeHealth(data.catalog);
  if (nodes.state === "unknown") return tile("Node library", "warn", "Unknown", "The catalog could not be read.");
  if (nodes.state === "ok") return tile("Node library", "ok", "All modules loaded", null);
  return tile("Node library", "bad", `${nodes.failed} modules failed to load`, nodes.reason, ["/help#nodes-red", "Why nodes are red"]);
}

/* ---- panels -------------------------------------------------------------- */

function panel(title, desc, ...body) {
  return h("section", { class: "panel" },
    h("div", { class: "panel-head" }, h("div", {}, h("div", { class: "panel-title" }, title), desc ? h("div", { class: "panel-desc" }, desc) : null)),
    h("div", { class: "panel-body" }, ...body));
}

function gettingStarted() {
  const items = checklist({
    environmentReady: data.env ? Boolean(data.env.present) : null,
    checkpoints: data.checkpoints,
    datasets: data.datasets,
    promptFiles: data.tokenizers,
    runs: data.runs ? data.runs.count : null,
  });
  const open = remaining(items);
  if (!open.length) return null;
  const next = open[0];
  const head = h("div", { class: "panel-head" },
    h("div", {}, h("div", { class: "panel-title" }, "Getting started"),
      h("div", { class: "panel-desc" }, "What is left before the first training run.")),
    h("span", { class: "chip chip-accent" }, `${items.length - open.length} of ${items.length} done`));
  return h("section", { class: "panel", id: "getting-started" }, head,
    h("div", { class: "panel-body" },
      h("ol", { class: "todo" }, items.map((item, i) => h("li", { class: `todo-item${item.done ? " is-done" : item === next ? " is-next" : ""}` },
        h("span", { class: "todo-mark", "aria-hidden": "true" }, item.done ? "✓" : String(i + 1)),
        h("span", { class: "todo-title" }, item.title),
        h("span", { class: "todo-detail" }, item.detail),
        item.done ? null : h("a", { class: `btn btn-link ${item === next ? "btn-primary" : "btn-secondary"} btn-small`, href: item.href }, item.action))))));
}

function runRow(run) {
  const label = run.name || `Run #${run.execution_id}`;
  const stamp = Date.parse(run.finished_at || run.started_at || run.created_at);
  const took = run.started_at && run.finished_at ? fmtDuration(Date.parse(run.finished_at) - Date.parse(run.started_at)) : null;
  const row = h("div", { class: "run-row" },
    h("span", { class: "run-name", title: label }, label),
    h("span", { class: "run-when" }, took ? `${took} · ${fmtRel(stamp)}` : fmtRel(stamp)),
    h("span", { class: `status-badge status-${BADGE[run.status] || "idle"}` }, run.status));
  if (run.status !== "error" && run.status !== "stopped") return row;
  // What went wrong stays one click away: the message the run ended with, and the process's own output while the server still keeps it.
  const logLink = h("a", { href: `/api/v1/graphs/executions/${run.execution_id}/log`, target: "_blank", rel: "noopener" }, "Full log");
  return h("details", { class: "run-more" },
    h("summary", {}, row),
    h("pre", { class: "run-error" }, run.error || "Stopped by request."),
    h("div", { class: "run-error-links" }, logLink));
}

function recentRuns() {
  const runs = data.runs ? data.runs.executions : null;
  const active = (runs || []).find((run) => run.status === "running");
  return panel("Recent runs", "Every run is kept with its charts; open the graph editor and press Watch to look inside one.",
    active ? h("div", { class: "callout callout-ok" }, `${active.name || `Run #${active.execution_id}`} is running. `, h("a", { href: "/graph" }, "Open the graph editor")) : null,
    runs === null ? h("div", { class: "empty" }, "The run history could not be read.")
      : runs.length ? h("div", { class: "runs" }, runs.slice(0, 6).map(runRow))
        : h("div", { class: "empty" }, "No runs yet. A run starts from a graph: the editor's ", h("em", {}, "Quick workflow"), " builds a ready one."));
}

function startPanel() {
  return panel("Start", null,
    h("div", { class: "start-links" },
      h("a", { class: "btn btn-primary btn-link", href: "/graph" }, "Open the graph editor"),
      h("a", { class: "btn btn-secondary btn-link", href: "/datasets" }, `Datasets${data.datasets != null ? ` (${data.datasets})` : ""}`),
      h("a", { class: "btn btn-secondary btn-link", href: "/settings" }, "Settings"),
      h("a", { class: "btn btn-secondary btn-link", href: "/help" }, "Help")));
}

function memoryPanel() {
  const view = memoryView(data.health ? data.health.memory : null);
  if (!view.known) {
    return panel("Card memory", null,
      h("div", { class: "empty" }, "Not measured yet. The server learns the card's memory when the first run asks for it."));
  }
  const share = (mb) => `${Math.min(100, Math.max(0, (mb / view.totalMb) * 100)).toFixed(1)}%`;
  return panel("Card memory", `${gb(view.totalMb)} in total.`,
    h("div", { class: "meter", role: "img", "aria-label": `${gb(view.heldMb)} of ${gb(view.totalMb)} held by runs` },
      h("span", { style: `width:${share(view.heldMb)}` })),
    h("div", { class: "kv", style: "margin-top:12px" },
      h("div", { class: "kv-row" }, h("span", { class: "kv-key" }, "Held by runs"), h("span", { class: "kv-value" }, gb(view.heldMb)), h("span", {})),
      h("div", { class: "kv-row" }, h("span", { class: "kv-key" }, "Free for a new run"), h("span", { class: "kv-value" }, view.freeMb == null ? "—" : gb(view.freeMb)), h("span", {})),
      view.holders.map((holder) => h("div", { class: "kv-row" }, h("span", { class: "kv-key" }, holder.owner), h("span", { class: "kv-value" }, gb(holder.mb)), h("span", {})))));
}

function render() {
  mount(root,
    h("div", { class: "tiles" }, serverTile(), cardTile(), environmentTile(), nodesTile()),
    h("div", { class: "home-grid" },
      h("div", { class: "home-col" }, gettingStarted(), recentRuns()),
      h("div", { class: "home-col" }, startPanel(), memoryPanel())));
}

/* ---- loading ------------------------------------------------------------- */

async function loadFast() {
  const [health, catalog, runs, datasets, checkpoints, tokenizers] = await Promise.all([
    quiet(api("/health")), quiet(api("/catalog")), quiet(api("/graphs/executions?limit=20")), quiet(api("/datasets")),
    quiet(api("/catalog/models/checkpoints")), quiet(api("/installer/tokenizers")),
  ]);
  Object.assign(data, {
    health, catalog, runs, tokenizers,
    datasets: datasets ? datasets.datasets.length : null,
    checkpoints: checkpoints ? checkpoints.files.length : null,
  });
}

async function boot() {
  await loadFast();
  data.settings = await quiet(api("/settings"));
  render();
  data.env = await checkEnvironment(data.settings ? data.settings.resolved.venv_python : null);
  render();
  // Runs and memory change while the page is open; the environment does not, and re-importing torch every few seconds would be rude.
  setInterval(async () => {
    const [health, runs] = await Promise.all([quiet(api("/health")), quiet(api("/graphs/executions?limit=20"))]);
    if (health) data.health = health;
    if (runs) data.runs = runs;
    render();
  }, 5000);
}

boot().catch((err) => {
  logError(err);
  mount(root, h("div", { class: "error-box" }, errText(err)));
});
