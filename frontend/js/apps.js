/* ---------------------------------------------------------------------------
   apps.js -- the Apps page (apps.html).

   Two routes, one module:
     /apps           the library of apps, and "new from a graph";
     /apps/<name>    one app. Mode **use** (default) is the interface a person
                     meets: change the exposed values, press Run, watch the live
                     cards. Mode **edit** unlocks the layout and a tray of
                     everything the app's graph offers.

   An app is a frozen snapshot of a graph plus a page of cards over it
   (docs/contracts/apps.md). The graph editor keeps authoring graphs; "Update
   from graph" is the explicit, reviewed step that copies a newer graph in.

   Where each piece of truth lives (so nothing here is a second copy):
     * the document -- `S.app`, saved with PUT. Its graph snapshot is what the
       field cards edit (the board's stand-in "draft graph" writes through to it);
     * what a card can bind to, and whether it still does -- the backend
       (`POST /boards/resolve`), re-asked after edits, never guessed here;
     * run state -- the executions API, polled; live numbers -- the monitor
       streams, one connection per stream (board/live.js).

   Run is refused client-side only for one reason the server would also
   refuse: the page shows a field the *saved* app does not expose. Everything
   else is the server's answer, shown as it came.
   --------------------------------------------------------------------------- */

import { api, ApiError } from "./api.js";
import { Board, settleLayout } from "./board/board.js";
import { LiveStreams } from "./board/live.js";
import {
  autoFill, buildAvailable, layoutKey, refKey, statusByBlock, valuesPayload,
} from "./board/model.js";
import { el, showMessage } from "./lib/dom.js";
import { errText } from "./lib/errors.js";
import { log } from "./lib/log.js";

const ACTIVE = new Set(["queued", "running"]);
const enc = encodeURIComponent;

const url = new URL(location.href);
const route = url.pathname.replace(/\/+$/, "");
const pageName = route.startsWith("/apps/") ? decodeURIComponent(route.slice("/apps/".length)) : null;
const query = url.searchParams;

const live = new LiveStreams();

/* ======================================================================== */
/* LIST VIEW                                                                  */
/* ======================================================================== */

async function showList() {
  el("view-list").hidden = false;
  el("view-page").hidden = true;
  try {
    const [apps, library] = await Promise.all([api("/apps"), api("/graphs/library")]);
    renderNewForm(library.graphs || [], new Set((apps.apps || []).map((t) => t.name)));
    renderCards(apps.apps || []);
  } catch (err) {
    const box = el("list-state");
    box.hidden = false;
    box.textContent = `Could not load apps: ${errText(err)}`;
  }
}

function renderNewForm(graphs, taken) {
  const select = el("new-graph");
  select.replaceChildren();
  for (const g of graphs) {
    const o = document.createElement("option");
    o.value = g.name;
    o.textContent = `${g.name} (${g.node_count} nodes)`;
    select.append(o);
  }
  const create = el("btn-new");
  const none = graphs.length === 0;
  create.disabled = none;
  if (none) {
    el("new-hint").textContent =
      "There is no saved graph yet. Build one in the Graph Editor and save it to the library; an app is built on a saved graph.";
  }

  const go = () => {
    const nm = el("new-name").value.trim();
    if (!nm) return showMessage("new-error", "Give it a name.");
    if (taken.has(nm)) return showMessage("new-error", `"${nm}" already exists — pick another name or open the existing one.`);
    if (!select.value) return showMessage("new-error", "Choose a graph.");
    showMessage("new-error", "");
    location.href = `/apps/${enc(nm)}?new=1&graph=${enc(select.value)}`;
  };
  create.onclick = go;
  el("new-name").onkeydown = (e) => { if (e.key === "Enter") go(); };
}

function renderCards(items) {
  el("list-state").hidden = items.length > 0;
  el("list-state").textContent = "Nothing here yet — create an app from one of your graphs above.";
  el("list-apps").hidden = items.length === 0;
  const grid = el("grid-apps");
  grid.replaceChildren();
  for (const t of items) {
    const card = document.createElement("a");
    card.className = "app-card";
    card.href = `/apps/${enc(t.name)}`;
    const name = document.createElement("div");
    name.className = "app-card-name";
    name.textContent = t.name;
    const desc = document.createElement("div");
    desc.className = "app-card-desc";
    desc.textContent = t.description || "";
    const meta = document.createElement("div");
    meta.className = "app-card-meta";
    meta.textContent = `v${t.version} · ${t.card_count} card${t.card_count === 1 ? "" : "s"} · ${t.node_count} node${t.node_count === 1 ? "" : "s"} · ${new Date(t.updated_at).toLocaleDateString()}`;
    card.append(name, desc, meta);
    grid.append(card);
  }
}

/* ======================================================================== */
/* PAGE VIEW                                                                  */
/* ======================================================================== */

/** Everything the page view holds. One object so it can be inspected in the
    console (`window.__app`) when something looks wrong. */
const S = {
  name: pageName,
  isNew: false,
  description: "",
  version: 0,
  app: null,           // the document: {graph, root, style, source_graph, ...}
  view: null,          // what the board edits: {columns, blocks, graph: draft}
  saved: null,         // {layout, graph, exposed} as last saved; null = never
  mode: "use",
  resolution: null,
  statuses: {},        // card id -> {status, detail}
  execution: null,     // {execution_id, status, started_at, finished_at, error, done, total}
  resolveSeq: 0,
  resolveTimer: 0,
  pollTimer: 0,
  board: null,
};
window.__app = S;

/** The graph as an app stores it: what a run reads, without the editor's layout and views. */
function snapshotGraph(graph) {
  const copy = structuredClone(graph);
  delete copy.views;
  for (const node of copy.nodes || []) { delete node.layout; node.params = node.params || {}; }
  copy.settings = copy.settings || {};
  return copy;
}

/**
 * What the board edits. Phase one of the editor is one page: the root's first tab. The "draft graph" is a stand-in whose
 * parameter objects *are* the snapshot's, so a field card writes straight into what gets saved and run.
 */
function makeView(app) {
  const graph = app.graph;
  const tab = app.root.tabs[0];
  const memory = new Proxy({}, {
    get: (_t, key) => graph.settings[`memory.${String(key)}`],
    set: (_t, key, value) => { graph.settings[`memory.${String(key)}`] = value; return true; },
    deleteProperty: (_t, key) => { delete graph.settings[`memory.${String(key)}`]; return true; },
    has: (_t, key) => `memory.${String(key)}` in graph.settings,
  });
  return {
    columns: app.root.columns,
    blocks: tab.blocks,
    graph: { get nodes() { return graph.nodes.map((n) => ({ id: n.uid, params: n.params })); }, memory },
  };
}

/** The document as it should be PUT: the board's current cards folded back into the page. */
function appDoc() {
  const tab = S.app.root.tabs[0];
  const blocks = S.view.blocks.map((b) => ({ ...b, collapsed: !!b.collapsed, title: b.title || null, options: b.options || {} }));
  return { ...S.app, format: 1, root: { ...S.app.root, columns: S.view.columns, tabs: [{ ...tab, blocks }] } };
}

const exposedKey = () =>
  JSON.stringify(S.view.blocks.filter((b) => b.ref && b.ref.kind === "field").map((b) => refKey(b.ref)).sort());
const graphKey = () => JSON.stringify(S.app.graph);

function markSaved() {
  S.saved = { layout: layoutKey(S.view), graph: graphKey(), exposed: exposedKey() };
}

const layoutDirty = () => !S.saved || S.saved.layout !== layoutKey(S.view);
const valuesDirty = () => !S.saved || S.saved.graph !== graphKey();
const isDirty = () => layoutDirty() || valuesDirty();
const exposureChanged = () => !S.saved || S.saved.exposed !== exposedKey();
const executionActive = () => !!S.execution && ACTIVE.has(S.execution.status);

async function showPage() {
  el("view-list").hidden = true;
  el("view-page").hidden = false;
  el("btn-back").hidden = false;
  el("app-title").textContent = S.name;
  document.title = `${S.name} · Apps`;
  try {
    await loadDoc();
  } catch (err) {
    showMessage("page-error", `Could not open "${S.name}": ${errText(err)}`);
    el("toolbar").hidden = true;
    return;
  }

  S.board = new Board(el("board"), {
    live,
    ctx: { execution: () => S.execution },
    onChange: onBoardChange,
    onNote: (msg, kind) => log(msg, kind || "info"),
  });
  S.board.setDoc(S.view);
  S.board.setModules(S.resolution.modules);
  applyResolution();
  setMode(S.isNew ? "edit" : "use");
  wireToolbar();
  el("cols").value = String(S.view.columns);
  describePage();
  await adoptActiveExecution();
  updateState();
}

async function loadDoc() {
  if (query.get("new")) {
    const graphName = query.get("graph") || "";
    const taken = (await api("/apps")).apps || [];
    if (taken.some((t) => t.name === S.name)) throw new Error(`an app called "${S.name}" already exists`);
    const lib = await api(`/graphs/library/${enc(graphName)}`);
    S.isNew = true;
    S.description = `Built on the graph "${graphName}".`;
    S.app = {
      format: 1, source_graph: graphName, style: { accent: "blue", density: "comfortable" },
      graph: snapshotGraph(lib.graph),
      root: { columns: 12, tabs: [{ id: "main", title: "Main", blocks: [] }] },
    };
    S.view = makeView(S.app);
    S.resolution = await resolveNow();
    S.view.blocks = autoFill(S.resolution.modules, S.view.columns);
    S.resolution = await resolveNow();
    S.saved = null;
  } else {
    const t = await api(`/apps/${enc(S.name)}`);
    S.description = t.description || "";
    S.version = t.version;
    S.app = t.doc;
    S.app.graph = snapshotGraph(S.app.graph);
    S.view = makeView(S.app);
    S.resolution = await resolveNow();
    markSaved();
  }
}

function describePage() {
  const bits = [];
  if (S.version) bits.push(`version ${S.version}`);
  if (S.app.source_graph) bits.push(`built on graph “${S.app.source_graph}”`);
  const n = S.view.blocks.length;
  bits.push(`${n} card${n === 1 ? "" : "s"}`);
  el("app-sub").textContent = bits.join(" · ");
  el("btn-update").hidden = !S.app.source_graph || S.isNew;
}

/* ---- resolving ------------------------------------------------------------ */

function resolveNow() {
  return api("/boards/resolve", { method: "POST", body: { doc: appDoc() } });
}

function scheduleResolve(delay = 300) {
  clearTimeout(S.resolveTimer);
  S.resolveTimer = setTimeout(async () => {
    const seq = ++S.resolveSeq;
    try {
      const res = await resolveNow();
      if (seq !== S.resolveSeq) return; // a newer edit superseded this answer
      S.resolution = res;
      S.board.setModules(res.modules);
      applyResolution();
      renderTray();
    } catch (err) {
      showMessage("page-error", `Could not check the app: ${errText(err)}`);
    }
  }, delay);
}

function applyResolution() {
  const res = S.resolution;
  S.statuses = {};
  for (const b of res.blocks) S.statuses[b.id] = { status: b.status, detail: b.detail };
  S.board.setStatuses(S.statuses);
  renderIssues(res.issues || []);
  renderTray();
  updateState();
}

function renderIssues(issues, heading) {
  const box = el("issues");
  const list = el("issues-list");
  list.replaceChildren();
  box.hidden = issues.length === 0;
  if (box.hidden) return;
  const errors = issues.filter((i) => i.severity === "error").length;
  el("issues-summary").textContent = heading
    || (errors
      ? `${errors} problem${errors === 1 ? "" : "s"} — this app cannot run yet`
      : `${issues.length} note${issues.length === 1 ? "" : "s"} about this graph`);
  box.classList.toggle("has-errors", errors > 0);
  box.open = errors > 0;
  for (const i of issues) {
    const li = document.createElement("li");
    li.className = i.severity === "error" ? "error" : "warn";
    const where = i.node_id ? `${i.node_id}${i.param ? "." + i.param : ""}: ` : "";
    li.textContent = `${where}${i.message}`;
    list.append(li);
  }
}

/* ---- board events ----------------------------------------------------------- */

function onBoardChange(reason) {
  if (reason === "blocks") { scheduleResolve(0); describePage(); }
  if (reason === "value") scheduleResolve(600);
  updateState();
}

/* ---- mode + toolbar --------------------------------------------------------- */

function setMode(mode) {
  S.mode = mode;
  el("mode-use").classList.toggle("active", mode === "use");
  el("mode-edit").classList.toggle("active", mode === "edit");
  el("mode-use").setAttribute("aria-pressed", String(mode === "use"));
  el("mode-edit").setAttribute("aria-pressed", String(mode === "edit"));
  el("tray").hidden = mode !== "edit";
  document.body.classList.toggle("app-editing", mode === "edit");
  S.board.setMode(mode);
  renderTray();
}

function wireToolbar() {
  el("mode-use").onclick = () => setMode("use");
  el("mode-edit").onclick = () => setMode("edit");
  el("btn-save").onclick = save;
  el("btn-run").onclick = run;
  el("btn-stop").onclick = stop;
  el("btn-delete").onclick = remove;
  el("btn-update").onclick = updateFromGraph;
  el("tray-adv").onchange = renderTray;
  el("tray-search").oninput = renderTray;
  el("cols").onchange = () => {
    const n = Math.max(1, Math.min(24, parseInt(el("cols").value, 10) || S.view.columns));
    el("cols").value = String(n);
    if (n === S.view.columns) return;
    S.view.columns = n;
    S.view.blocks = settleLayout(S.view.blocks, n);
    S.board.applyLayout();
    onBoardChange("layout");
  };
  window.addEventListener("beforeunload", (e) => {
    if (S.view && isDirty()) { e.preventDefault(); e.returnValue = ""; }
  });
}

function updateState() {
  if (!S.view) return;
  const save = el("btn-save");
  save.textContent = isDirty() ? "Save •" : "Save";
  save.classList.toggle("is-dirty", isDirty());

  const run = el("btn-run");
  const active = executionActive();
  el("btn-stop").hidden = !active;
  let why = "";
  if (S.isNew) why = "Save this app first.";
  else if (exposureChanged()) why = "Save the app first — it now shows fields the saved app does not.";
  else if (active) why = "A run is already in progress.";
  run.disabled = !!why;
  run.title = why || "Run the graph with the values on this page";

  const bits = [];
  if (S.isNew) bits.push("not saved yet");
  else if (isDirty()) bits.push(layoutDirty() && valuesDirty() ? "unsaved layout and values" : layoutDirty() ? "unsaved layout" : "unsaved values");
  el("bd-state").textContent = bits.join(" · ");
}

/* ---- tray ------------------------------------------------------------------- */

function renderTray() {
  const tray = el("tray-list");
  if (!S.view || !S.resolution || S.mode !== "edit") return;
  const needle = el("tray-search").value.trim().toLowerCase();
  let groups = buildAvailable(S.resolution.modules, S.view.blocks, { showAdvanced: el("tray-adv").checked });
  groups = groups
    .map((g) => ({
      ...g,
      items: needle
        ? g.items.filter((i) => `${i.label} ${g.title} ${i.sub}`.toLowerCase().includes(needle))
        : g.items,
    }))
    .filter((g) => g.items.length);

  // Keep open/closed state of groups across redraws.
  const open = new Map([...tray.querySelectorAll("details")].map((d) => [d.dataset.key, d.open]));
  tray.replaceChildren();
  if (!groups.length) {
    const empty = document.createElement("div");
    empty.className = "bd-tray-empty";
    empty.textContent = needle ? "Nothing matches." : "Nothing more to add.";
    tray.append(empty);
    return;
  }
  for (const g of groups) {
    const d = document.createElement("details");
    d.dataset.key = g.key;
    d.open = open.has(g.key) ? open.get(g.key) : g.scope === "graph" || groups.length <= 3 || !!needle;
    const sum = document.createElement("summary");
    sum.textContent = g.title;
    d.append(sum);
    for (const kind of ["display", "field"]) {
      const items = g.items.filter((i) => i.kind === kind);
      if (!items.length) continue;
      const cap = document.createElement("div");
      cap.className = "bd-tray-cap";
      cap.textContent = kind === "display" ? "Live views" : "Settings";
      d.append(cap);
      for (const it of items) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "bd-tray-item";
        b.disabled = it.placed;
        b.title = it.sub || it.label;
        const label = document.createElement("span");
        label.textContent = it.label;
        const tag = document.createElement("span");
        tag.className = "bd-tray-tag";
        tag.textContent = it.placed ? "added" : it.advanced ? "advanced" : "+";
        b.append(label, tag);
        b.onclick = () => S.board.addRef(it.ref);
        d.append(b);
      }
    }
    tray.append(d);
  }
}

/* ---- save / delete ------------------------------------------------------------ */

async function save() {
  showMessage("page-error", "");
  try {
    const out = await api(`/apps/${enc(S.name)}`, {
      method: "PUT",
      body: { description: S.description, doc: appDoc() },
    });
    S.isNew = false;
    S.version = out.version;
    markSaved();
    history.replaceState(null, "", `/apps/${enc(S.name)}`);
    describePage();
    updateState();
    log(`Saved "${out.name}" (version ${out.version}, ${out.card_count} cards).`, "success");
  } catch (err) {
    const details = err instanceof ApiError && Array.isArray(err.details) ? `\n${err.details.join("\n")}` : "";
    showMessage("page-error", `Could not save: ${errText(err)}${details}`);
  }
}

async function remove() {
  if (!window.confirm(`Delete "${S.name}"? This removes the app, not the graph it was built on.`)) return;
  if (!S.isNew) {
    try { await api(`/apps/${enc(S.name)}`, { method: "DELETE" }); }
    catch (err) { return showMessage("page-error", `Could not delete: ${errText(err)}`); }
  }
  markSaved(); // no unload prompt
  location.href = "/apps";
}

/** Copy the source graph's current state into the app, after showing what that changes. */
async function updateFromGraph() {
  showMessage("page-error", "");
  try {
    const lib = await api(`/graphs/library/${enc(S.app.source_graph)}`);
    const next = snapshotGraph(lib.graph);
    const diff = await api(`/apps/${enc(S.name)}/diff`, { method: "POST", body: { graph: next } });
    const nothing = !diff.added.length && !diff.removed.length && !diff.changed.length && !diff.wiring_changed && !diff.settings_changed;
    if (nothing) return log("The app already matches its graph.", "info");
    const lines = [];
    if (diff.added.length) lines.push(`${diff.added.length} node(s) added`);
    if (diff.removed.length) lines.push(`${diff.removed.length} node(s) removed`);
    if (diff.changed.length) lines.push(`${diff.changed.length} node(s) changed`);
    if (diff.wiring_changed) lines.push("wiring changed");
    if (diff.settings_changed) lines.push("graph settings changed");
    if (diff.orphaned_cards.length) lines.push(`${diff.orphaned_cards.length} card(s) would lose their target`);
    if (!window.confirm(`Update "${S.name}" from graph "${S.app.source_graph}"?\n\n${lines.join("\n")}\n\nValues typed on this page are replaced by the graph's.`)) return;
    S.app.graph = next;
    S.view = makeView({ ...S.app, root: { ...S.app.root, columns: S.view.columns, tabs: [{ ...S.app.root.tabs[0], blocks: S.view.blocks }] } });
    S.board.setDoc(S.view);
    scheduleResolve(0);
    updateState();
  } catch (err) {
    showMessage("page-error", `Could not update from the graph: ${errText(err)}`);
  }
}

/* ---- run / stop --------------------------------------------------------------- */

async function run() {
  showMessage("page-error", "");
  const values = valuesPayload(S.view, statusByBlock(S.resolution));
  try {
    const exec = await api(`/apps/${enc(S.name)}/run`, { method: "POST", body: { values } });
    log(`Run #${exec.execution_id} started from "${S.name}".`, "success");
    S.execution = { ...exec, done: 0, total: 0 };
    S.board.refreshDisplays();
    updateState();
    schedulePoll(400);
  } catch (err) {
    if (err instanceof ApiError && err.code === "graph_invalid") {
      const issues = Array.isArray(err.details) ? err.details : (err.details && err.details.issues) || [];
      renderIssues(issues, `${issues.length} problem${issues.length === 1 ? "" : "s"} — the run was refused`);
      el("issues").open = true;
      showMessage("page-error", "The graph is not valid with these values — see the list below.");
    } else if (err instanceof ApiError && err.code === "graph_execution_active") {
      showMessage("page-error", "Another run is active — stop it first (in the Graph Editor, or here once it shows).");
      adoptActiveExecution();
    } else if (err instanceof ApiError && err.code === "memory_unavailable") {
      showMessage("page-error", `${errText(err)}\n${describeMemory(err.details)}`);
    } else {
      const details = err instanceof ApiError && Array.isArray(err.details) ? `\n${err.details.join("\n")}` : "";
      showMessage("page-error", `${errText(err)}${details}`);
    }
  }
}

function describeMemory(details) {
  if (!details || typeof details !== "object") return "";
  return Object.entries(details)
    .filter(([, v]) => typeof v === "number" || typeof v === "string")
    .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`)
    .join("\n");
}

async function stop() {
  if (!S.execution) return;
  try {
    await api(`/graphs/executions/${S.execution.execution_id}/stop`, { method: "POST" });
    log(`Run #${S.execution.execution_id} stopped.`, "warn");
    schedulePoll(200);
  } catch (err) {
    showMessage("page-error", `Could not stop: ${errText(err)}`);
  }
}

async function adoptActiveExecution() {
  try {
    const list = await api("/graphs/executions?limit=1");
    const latest = (list.executions || [])[0];
    if (latest && ACTIVE.has(latest.status)) {
      S.execution = { ...latest, done: 0, total: 0 };
      schedulePoll(0);
    } else if (latest && (!S.execution || S.execution.execution_id === latest.execution_id)) {
      S.execution = { ...latest, done: 0, total: 0 };
      await pollExecution(false);
    }
    S.board.refreshDisplays();
    updateState();
  } catch (err) {
    log(`Could not read run state: ${errText(err)}`, "warn");
  }
}

function schedulePoll(delay) {
  clearTimeout(S.pollTimer);
  S.pollTimer = setTimeout(() => pollExecution(true), delay);
}

async function pollExecution(again) {
  if (!S.execution) return;
  try {
    const d = await api(`/graphs/executions/${S.execution.execution_id}`);
    S.execution = {
      execution_id: d.execution_id, status: d.status, error: d.error,
      started_at: d.started_at, finished_at: d.finished_at,
      done: (d.results || []).length, total: ((d.graph && d.graph.nodes) || []).length,
    };
  } catch (err) {
    log(`Could not read run state: ${errText(err)}`, "warn");
  }
  S.board.refreshDisplays();
  updateState();
  if (again && executionActive()) schedulePoll(1500);
}

/* ======================================================================== */

if (pageName === null) showList();
else showPage();
