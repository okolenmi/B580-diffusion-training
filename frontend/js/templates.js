/* ---------------------------------------------------------------------------
   templates.js -- the Templates page (templates.html).

   Two routes, one module:
     /templates          the library: pages and monitor windows, and "new
                         from a graph";
     /templates/<name>   one page. Mode **use** (default) is the interface a
                         beginner sees: change the exposed values, press Run,
                         watch the live blocks. Mode **edit** unlocks the
                         layout and a tray of everything the graph offers.

   A *page* (kind "template") edits and runs a graph; a *monitor window*
   (kind "monitor") only watches one. Same document, same board engine.

   Where each piece of truth lives (so nothing here is a second copy):
     * the document and its draft graph -- `S.doc`, saved with PUT;
     * what a block can bind to, and whether it still does -- the backend
       (`POST /templates/resolve`), re-asked after edits, never guessed here;
     * run state -- the executions API, polled; live numbers -- the monitor
       streams, one connection per stream (board/live.js).

   Run is refused client-side only for one reason the server would also
   refuse: the page shows a field the *saved* page does not expose. Everything
   else is the server's answer, shown as it came.
   --------------------------------------------------------------------------- */

import { api, ApiError } from "./api.js";
import { Board } from "./board/board.js";
import { LiveStreams } from "./board/live.js";
import { settleLayout } from "./board/board.js";
import {
  autoFill, buildAvailable, emptyDoc, graphKey, layoutKey, refKey, statusByBlock, toSaveDoc,
  valuesPayload,
} from "./board/model.js";
import { el, showMessage } from "./lib/dom.js";
import { errText } from "./lib/errors.js";
import { log } from "./lib/log.js";

const ACTIVE = new Set(["queued", "running"]);
const enc = encodeURIComponent;

const url = new URL(location.href);
const route = url.pathname.replace(/\/+$/, "");
const pageName = route.startsWith("/templates/") ? decodeURIComponent(route.slice("/templates/".length)) : null;
const query = url.searchParams;
const WINDOW_MODE = query.has("window");
if (WINDOW_MODE) document.body.classList.add("tpl-window");

const live = new LiveStreams();

/* ======================================================================== */
/* LIST VIEW                                                                  */
/* ======================================================================== */

async function showList() {
  el("view-list").hidden = false;
  el("view-page").hidden = true;
  try {
    const [templates, library] = await Promise.all([api("/templates"), api("/graphs/library")]);
    renderNewForm(library.graphs || [], new Set((templates.templates || []).map((t) => t.name)));
    renderCards(templates.templates || []);
  } catch (err) {
    const box = el("list-state");
    box.hidden = false;
    box.textContent = `Could not load templates: ${errText(err)}`;
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
  el("new-hint").textContent = none
    ? "There is no saved graph yet. Build one in the Graph Editor and save it to the library; a page is built on a saved graph."
    : el("new-hint").textContent;
  const kindSel = el("new-kind");
  const hint = {
    template: "A page starts with the settings the graph's author already filled in and the live views its nodes offer. Everything else is one click away in Edit mode.",
    monitor: "A monitor window starts empty: choose which live views of the graph's nodes to show, and arrange them.",
  };
  kindSel.onchange = () => { if (!none) el("new-hint").textContent = hint[kindSel.value]; };

  const go = () => {
    const nm = el("new-name").value.trim();
    if (!nm) return showMessage("new-error", "Give it a name.");
    if (taken.has(nm)) return showMessage("new-error", `"${nm}" already exists — pick another name or open the existing one.`);
    if (!select.value) return showMessage("new-error", "Choose a graph.");
    showMessage("new-error", "");
    location.href = `/templates/${enc(nm)}?new=1&kind=${enc(kindSel.value)}&graph=${enc(select.value)}`;
  };
  create.onclick = go;
  el("new-name").onkeydown = (e) => { if (e.key === "Enter") go(); };
}

function renderCards(items) {
  el("list-state").hidden = items.length > 0;
  el("list-state").textContent = "Nothing here yet — create a page from one of your graphs above.";
  for (const [gridId, sectionId, kind] of [
    ["grid-pages", "list-pages", "template"], ["grid-monitors", "list-monitors", "monitor"],
  ]) {
    const mine = items.filter((t) => t.kind === kind);
    el(sectionId).hidden = mine.length === 0;
    const grid = el(gridId);
    grid.replaceChildren();
    for (const t of mine) {
      const card = document.createElement("a");
      card.className = "tpl-card";
      card.href = `/templates/${enc(t.name)}`;
      const name = document.createElement("div");
      name.className = "tpl-card-name";
      name.textContent = t.name;
      const desc = document.createElement("div");
      desc.className = "tpl-card-desc";
      desc.textContent = t.description || "";
      const meta = document.createElement("div");
      meta.className = "tpl-card-meta";
      meta.textContent = `${t.block_count} block${t.block_count === 1 ? "" : "s"} · ${t.node_count} node${t.node_count === 1 ? "" : "s"} · ${new Date(t.updated_at).toLocaleDateString()}`;
      card.append(name, desc, meta);
      grid.append(card);
    }
  }
}

/* ======================================================================== */
/* PAGE VIEW                                                                  */
/* ======================================================================== */

/** Everything the page view holds. One object so it can be inspected in the
    console (`window.__tpl`) when something looks wrong. */
const S = {
  name: pageName,
  isNew: false,
  description: "",
  doc: null,
  saved: null,         // {layout, graph, exposed} as last saved; null = never
  mode: "use",
  resolution: null,
  statuses: {},        // block id -> {status, detail}
  execution: null,     // {execution_id, status, started_at, finished_at, error, done, total}
  resolveSeq: 0,
  resolveTimer: 0,
  pollTimer: 0,
  board: null,
};
window.__tpl = S;

const snapshotGraph = (g) => ({
  nodes: (g.nodes || []).map((n) => ({ id: n.id, class_name: n.class_name, params: { ...(n.params || {}) } })),
  edges: (g.edges || []).map((e) => ({ ...e })),
  ...(g.memory ? { memory: { ...g.memory } } : {}),
});

const exposedKey = (doc) =>
  JSON.stringify(doc.blocks.filter((b) => b.ref.kind === "field").map((b) => refKey(b.ref)).sort());

function markSaved() {
  S.saved = { layout: layoutKey(S.doc), graph: graphKey(S.doc), exposed: exposedKey(S.doc) };
}

const layoutDirty = () => !S.saved || S.saved.layout !== layoutKey(S.doc);
const valuesDirty = () => !S.saved || S.saved.graph !== graphKey(S.doc);
const isDirty = () => layoutDirty() || valuesDirty();
const exposureChanged = () => !S.saved || S.saved.exposed !== exposedKey(S.doc);
const executionActive = () => !!S.execution && ACTIVE.has(S.execution.status);

async function showPage() {
  el("view-list").hidden = true;
  el("view-page").hidden = false;
  el("btn-back").hidden = false;
  el("tpl-title").textContent = S.name;
  document.title = `${S.name} · Templates`;
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
  S.board.setDoc(S.doc);
  S.board.setModules(S.resolution.modules);
  applyResolution();
  setMode(S.isNew && S.doc.kind === "monitor" ? "edit" : "use");
  wireToolbar();
  el("btn-popout").hidden = S.doc.kind !== "monitor" || S.isNew;
  el("cols").value = String(S.doc.columns);
  describePage();
  await adoptActiveExecution();
  updateState();
}

async function loadDoc() {
  if (query.get("new")) {
    const kind = query.get("kind") === "monitor" ? "monitor" : "template";
    const graphName = query.get("graph") || "";
    const taken = (await api("/templates")).templates || [];
    if (taken.some((t) => t.name === S.name)) {
      throw new Error(`a page called "${S.name}" already exists`);
    }
    const lib = await api(`/graphs/library/${enc(graphName)}`);
    S.isNew = true;
    S.description = `Built on the graph "${graphName}".`;
    S.doc = emptyDoc(kind, snapshotGraph(lib.graph));
    S.doc.source_graph = graphName;
    S.resolution = await resolveNow();
    if (kind === "template") S.doc.blocks = autoFill(S.resolution.modules, S.doc.columns);
    S.resolution = await resolveNow();
    S.saved = null;
  } else {
    const t = await api(`/templates/${enc(S.name)}`);
    S.description = t.description || "";
    S.doc = t.doc;
    S.doc.blocks = S.doc.blocks || [];
    S.resolution = await resolveNow();
    markSaved();
  }
}

function describePage() {
  const d = S.doc;
  const kind = d.kind === "monitor" ? "Monitor window" : "Page";
  const bits = [kind];
  if (d.source_graph) bits.push(`built on graph “${d.source_graph}”`);
  bits.push(`${d.blocks.length} block${d.blocks.length === 1 ? "" : "s"}`);
  el("tpl-sub").textContent = bits.join(" · ");
  el("btn-run").hidden = d.kind === "monitor";
  if (d.kind === "monitor") el("btn-stop").hidden = true;
}

/* ---- resolving ------------------------------------------------------------ */

function resolveNow() {
  return api("/templates/resolve", { method: "POST", body: { doc: toSaveDoc(S.doc) } });
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
      showMessage("page-error", `Could not check the page: ${errText(err)}`);
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
  box.hidden = issues.length === 0 || S.doc.kind === "monitor";
  if (box.hidden) return;
  const errors = issues.filter((i) => i.severity === "error").length;
  el("issues-summary").textContent = heading
    || (errors
      ? `${errors} problem${errors === 1 ? "" : "s"} — this page cannot run yet`
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
  document.body.classList.toggle("tpl-editing", mode === "edit");
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
  el("btn-popout").onclick = () =>
    window.open(`/templates/${enc(S.name)}?window=1`, `tpl-${S.name}`, "popup,width=960,height=640");
  el("tray-adv").onchange = renderTray;
  el("tray-search").oninput = renderTray;
  el("cols").onchange = () => {
    const n = Math.max(1, Math.min(24, parseInt(el("cols").value, 10) || S.doc.columns));
    el("cols").value = String(n);
    if (n === S.doc.columns) return;
    S.doc.columns = n;
    S.doc.blocks = settleLayout(S.doc.blocks, n);
    S.board.applyLayout();
    onBoardChange("layout");
  };
  window.addEventListener("beforeunload", (e) => {
    if (S.doc && isDirty() && !WINDOW_MODE) { e.preventDefault(); e.returnValue = ""; }
  });
}

function updateState() {
  if (!S.doc) return;
  const save = el("btn-save");
  save.textContent = isDirty() ? "Save •" : "Save";
  save.classList.toggle("is-dirty", isDirty());

  const run = el("btn-run");
  const active = executionActive();
  el("btn-stop").hidden = !active || S.doc.kind === "monitor";
  let why = "";
  if (S.doc.kind === "template") {
    if (S.isNew) why = "Save this page first.";
    else if (exposureChanged()) why = "Save the page first — it now shows fields the saved page does not.";
    else if (active) why = "A run is already in progress.";
  }
  run.disabled = !!why;
  run.title = why || "Run the graph with the values on this page";

  const bits = [];
  if (S.isNew) bits.push("not saved yet");
  else if (isDirty()) bits.push(layoutDirty() && valuesDirty() ? "unsaved layout and values" : layoutDirty() ? "unsaved layout" : "unsaved values");
  el("tpl-state").textContent = bits.join(" · ");
}

/* ---- tray ------------------------------------------------------------------- */

function renderTray() {
  const tray = el("tray-list");
  if (!S.doc || !S.resolution || S.mode !== "edit") return;
  const needle = el("tray-search").value.trim().toLowerCase();
  let groups = buildAvailable(S.resolution.modules, S.doc.blocks, { showAdvanced: el("tray-adv").checked });
  if (S.doc.kind === "monitor") {
    groups = groups.map((g) => ({ ...g, items: g.items.filter((i) => i.kind === "display") }));
  }
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
    empty.className = "tpl-tray-empty";
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
      cap.className = "tpl-tray-cap";
      cap.textContent = kind === "display" ? "Live views" : "Settings";
      d.append(cap);
      for (const it of items) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "tpl-tray-item";
        b.disabled = it.placed;
        b.title = it.sub || it.label;
        const label = document.createElement("span");
        label.textContent = it.label;
        const tag = document.createElement("span");
        tag.className = "tpl-tray-tag";
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
    const out = await api(`/templates/${enc(S.name)}`, {
      method: "PUT",
      body: { description: S.description, doc: toSaveDoc(S.doc) },
    });
    S.isNew = false;
    markSaved();
    history.replaceState(null, "", `/templates/${enc(S.name)}`);
    el("btn-popout").hidden = S.doc.kind !== "monitor";
    updateState();
    log(`Saved "${out.name}" (${out.block_count} blocks).`, "success");
  } catch (err) {
    const details = err instanceof ApiError && Array.isArray(err.details) ? `\n${err.details.join("\n")}` : "";
    showMessage("page-error", `Could not save: ${errText(err)}${details}`);
  }
}

async function remove() {
  if (!window.confirm(`Delete "${S.name}"? This removes the page, not the graph it is built on.`)) return;
  if (!S.isNew) {
    try { await api(`/templates/${enc(S.name)}`, { method: "DELETE" }); }
    catch (err) { return showMessage("page-error", `Could not delete: ${errText(err)}`); }
  }
  S.saved = { layout: layoutKey(S.doc), graph: graphKey(S.doc), exposed: exposedKey(S.doc) }; // no unload prompt
  location.href = "/templates";
}

/* ---- run / stop --------------------------------------------------------------- */

async function run() {
  showMessage("page-error", "");
  const values = valuesPayload(S.doc, statusByBlock(S.resolution));
  try {
    const exec = await api(`/templates/${enc(S.name)}/run`, { method: "POST", body: { values } });
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
