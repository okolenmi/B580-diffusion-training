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
import { ContainerView } from "./board/container.js";
import { History } from "./board/history.js";
import { Inspector } from "./board/inspector.js";
import { LiveStreams } from "./board/live.js";
import {
  applyValues, autoPage, buildAvailable, collectIds, containerKey, groupCards, leafCards, locateCard, moveCard, moveTargets, refKey, statusByBlock,
  titleFor, valuesPayload, walkCards,
} from "./board/model.js";
import { el, showMessage } from "./lib/dom.js";
import { errText } from "./lib/errors.js";
import { log } from "./lib/log.js";

const ACTIVE = new Set(["planned", "queued", "running"]);
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
  draft: null,         // stand-in graph the field cards write into: {nodes, memory}
  sel: null,           // selected card id (edit mode)
  picked: new Set(),   // every selected card (Ctrl/Shift-click adds); always holds `sel`
  target: null,        // the Board "add" lands in
  history: new History(),
  saved: null,         // {layout, graph, exposed} as last saved; null = never
  mode: "use",
  resolution: null,
  statuses: {},        // card id -> {status, detail}
  execution: null,     // {execution_id, status, started_at, finished_at, error, done, total}
  runs: null,          // {history: this app's runs, planned: every run waiting for memory}, newest first; null before the first answer
  resolveSeq: 0,
  resolveTimer: 0,
  pollTimer: 0,
  page: null,          // the ContainerView over app.root
  inspector: null,
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

/** The "draft graph" the cards edit: a stand-in whose parameter objects *are* the snapshot's, so a field card writes straight into
    what gets saved and run. */
function makeDraft(app) {
  const graph = app.graph;
  const memory = new Proxy({}, {
    get: (_t, key) => graph.settings[`memory.${String(key)}`],
    set: (_t, key, value) => { graph.settings[`memory.${String(key)}`] = value; return true; },
    deleteProperty: (_t, key) => { delete graph.settings[`memory.${String(key)}`]; return true; },
    has: (_t, key) => `memory.${String(key)}` in graph.settings,
  });
  return { get nodes() { return graph.nodes.map((n) => ({ id: n.uid, params: n.params })); }, memory };
}

/** Every card of the page, at any depth (the page edits `S.app.root` in place, so this is always current). */
const rootBlocks = () => S.app.root.tabs.flatMap((t) => t.blocks);

/** The document as it should be PUT. */
const appDoc = () => ({ ...S.app, format: 1 });

const exposedKey = () =>
  JSON.stringify([...leafCards(rootBlocks())].filter((b) => b.ref.kind === "field").map((b) => refKey(b.ref)).sort());
const graphKey = () => JSON.stringify(S.app.graph);

function markSaved() {
  S.saved = { layout: containerKey(S.app.root), graph: graphKey(), exposed: exposedKey() };
}

const layoutDirty = () => !S.saved || S.saved.layout !== containerKey(S.app.root);
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

  S.page = new ContainerView(el("board"), {
    live,
    ctx: {
      execution: () => S.execution,
      runs: () => S.runs,
      runAgain,
      cancelRun,
      canRunAgain: () => S.mode === "use" && !S.isNew,
    },
    onNote: (msg, kind) => log(msg, kind || "info"),
    ids: () => collectIds(rootBlocks()),
    onSelect,
  }, { onChange: onBoardChange });
  S.inspector = new Inspector(el("inspector"), {
    patch: (id, patch) => { const hit = S.page.find(id); if (hit) hit.board.update(id, patch, "title"); renderInspector(); },
    move: moveSelected,
    remove: (id) => { const hit = S.page.find(id); if (hit) hit.board.removeId(id); },
    group: groupSelected,
    removeMany: () => { for (const id of [...S.picked]) { const hit = S.page.find(id); if (hit) hit.board.removeId(id); } },
    addTab: (id) => { const hit = S.page.find(id); const entry = hit && hit.board.entries.get(id); if (entry && entry.view) entry.view.addTab(); },
  });
  S.page.setModules(S.resolution.modules);
  S.page.setContainer(S.app.root, S.draft);
  S.history.reset(JSON.stringify(S.app.root));
  applyResolution();
  setMode(S.isNew ? "edit" : "use");
  wireToolbar();
  describePage();
  await adoptActiveExecution();
  await refreshRuns();
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
    S.draft = makeDraft(S.app);
    S.resolution = await resolveNow();
    S.app.root.tabs = autoPage(S.resolution.modules, S.app.root.columns);
    S.resolution = await resolveNow();
    S.saved = null;
  } else {
    const t = await api(`/apps/${enc(S.name)}`);
    S.description = t.description || "";
    S.version = t.version;
    S.app = t.doc;
    S.app.graph = snapshotGraph(S.app.graph);
    S.draft = makeDraft(S.app);
    S.resolution = await resolveNow();
    markSaved();
  }
}

function describePage() {
  const bits = [];
  if (S.version) bits.push(`version ${S.version}`);
  if (S.app.source_graph) bits.push(`built on graph “${S.app.source_graph}”`);
  const n = [...walkCards(rootBlocks())].length;
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
      S.page.setModules(res.modules);
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
  S.page.setStatuses(S.statuses);
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
  if (reason === "blocks") {
    scheduleResolve(0);
    describePage();
    S.picked = new Set([...S.picked].filter((id) => S.page.find(id)));
    if (S.sel && !S.picked.has(S.sel)) S.sel = [...S.picked][0] || null;
    renderInspector();
    renderTray();
  }
  if (reason === "value") scheduleResolve(600);
  if (reason !== "value") S.history.record(JSON.stringify(S.app.root), { merge: reason === "text" });
  if (reason === "layout") renderInspector();
  updateState();
}

/* ---- selection, inspector, undo ----------------------------------------------- */

const targetBoard = () => (S.target && S.target.alive ? S.target : S.page.activeBoard);

function onSelect({ board, id, view, extend }) {
  const target = view ? view.activeBoard : board;
  // Cards are picked together only from one tab: a card on another board starts a new selection.
  if (extend && id && S.sel && S.target === target && S.target.alive) {
    if (S.picked.has(id) && S.picked.size > 1) S.picked.delete(id); else S.picked.add(id);
  } else {
    S.picked = new Set(id ? [id] : []);
  }
  S.sel = id && S.picked.has(id) ? id : [...S.picked][0] || null;
  S.target = target;
  markSelection();
  renderInspector();
  el("cols").value = String(targetBoard().view.container.columns);
}

function markSelection() {
  S.page.mark(S.picked.size > 1 ? S.picked : S.sel, targetBoard());
}

function renderInspector() {
  const hit = S.sel && S.mode === "edit" ? S.page.find(S.sel) : null;
  if (!hit) { S.inspector.hide(); return; }
  if (S.picked.size > 1) {
    S.inspector.showMany({ count: S.picked.size, canGroup: groupCards(S.app.root, [...S.picked]) !== S.app.root });
    markSelection();
    return;
  }
  const { block } = hit;
  S.inspector.show({
    block,
    defaultTitle: titleFor({ ...block, title: null }, S.resolution && S.resolution.modules).defaultTitle,
    targets: moveTargets(S.app.root, block.id),
    here: locateCard(S.app.root, block.id),
  });
  markSelection();
}

/** Wrap the picked cards (or the selected one) in a new box, and select the box. */
function groupSelected() {
  const ids = S.picked.size > 1 ? [...S.picked] : [S.sel];
  const root = groupCards(S.app.root, ids);
  if (root === S.app.root) { log("Those cards cannot be boxed together (one tab only, and boxes nest three deep at most).", "warn"); return; }
  const box = locateCard(root, ids[0]).owner;
  applyRoot(root);
  S.picked = new Set([box]);
  S.sel = box;
  markSelection();
  renderInspector();
}

function moveSelected(id, owner, tabId) {
  const root = moveCard(S.app.root, id, owner, tabId);
  if (root === S.app.root) return;
  applyRoot(root);
}

/** Replace the whole page (a move, an undo): rebuild the boards over it and re-check it with the server. */
function applyRoot(root, { record = true } = {}) {
  S.app.root = root;
  S.page.setContainer(root, S.draft);
  if (S.sel && !S.page.find(S.sel)) S.sel = null;
  S.picked = new Set(S.sel ? [S.sel] : []);
  if (record) S.history.record(JSON.stringify(root));
  scheduleResolve(0);
  describePage();
  renderInspector();
  renderTray();
  updateHistoryButtons();
  el("cols").value = String(targetBoard().view.container.columns);
  updateState();
}

function stepHistory(direction) {
  const snap = direction < 0 ? S.history.undo() : S.history.redo();
  if (snap !== null) applyRoot(JSON.parse(snap), { record: false });
}

function updateHistoryButtons() {
  el("btn-undo").disabled = !S.history.canUndo;
  el("btn-redo").disabled = !S.history.canRedo;
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
  S.page.setMode(mode);
  if (mode !== "edit") { S.sel = null; S.picked = new Set(); }
  renderInspector();
  renderTray();
  updateHistoryButtons();
}

function wireToolbar() {
  el("mode-use").onclick = () => setMode("use");
  el("mode-edit").onclick = () => setMode("edit");
  el("btn-save").onclick = save;
  el("btn-run").onclick = () => run("auto");
  el("btn-queue").onclick = () => run("queue");
  el("btn-stop").onclick = stop;
  el("btn-delete").onclick = remove;
  el("btn-update").onclick = updateFromGraph;
  el("btn-undo").onclick = () => stepHistory(-1);
  el("btn-redo").onclick = () => stepHistory(1);
  el("btn-phone").onclick = () => {
    const on = el("board").classList.toggle("bd-phone");
    el("btn-phone").setAttribute("aria-pressed", String(on));
  };
  el("add-note").onclick = () => targetBoard().addNote();
  el("add-heading").onclick = () => targetBoard().addNote("heading");
  el("add-section").onclick = () => targetBoard().addSection();
  document.addEventListener("keydown", (ev) => {
    if (S.mode !== "edit" || !(ev.ctrlKey || ev.metaKey) || ev.target.closest("input, textarea, select")) return;
    const key = ev.key.toLowerCase();
    if (key === "z") { ev.preventDefault(); stepHistory(ev.shiftKey ? 1 : -1); }
    else if (key === "y") { ev.preventDefault(); stepHistory(1); }
  });
  el("tray-adv").onchange = renderTray;
  el("tray-search").oninput = renderTray;
  el("cols").onchange = () => {
    const view = targetBoard().view;
    const n = Math.max(1, Math.min(24, parseInt(el("cols").value, 10) || view.container.columns));
    el("cols").value = String(n);
    view.setColumns(n);
  };
  window.addEventListener("beforeunload", (e) => {
    if (S.app && isDirty()) { e.preventDefault(); e.returnValue = ""; }
  });
}

function updateState() {
  if (!S.app) return;
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
  const queue = el("btn-queue");
  const queueWhy = S.isNew ? "Save this app first." : exposureChanged() ? "Save the app first — it now shows fields the saved app does not." : "";
  queue.disabled = !!queueWhy;
  queue.title = queueWhy || "Run these values after whatever is running or waiting";

  const bits = [];
  if (S.isNew) bits.push("not saved yet");
  else if (isDirty()) bits.push(layoutDirty() && valuesDirty() ? "unsaved layout and values" : layoutDirty() ? "unsaved layout" : "unsaved values");
  el("bd-state").textContent = bits.join(" · ");
}

/* ---- tray ------------------------------------------------------------------- */

function renderTray() {
  const tray = el("tray-list");
  if (!S.app || !S.resolution || S.mode !== "edit") return;
  const needle = el("tray-search").value.trim().toLowerCase();
  let groups = buildAvailable(S.resolution.modules, rootBlocks(), { showAdvanced: el("tray-adv").checked });
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
    if (g.scope === "node" && g.items.some((i) => i.kind === "field" && !i.placed && !i.advanced)) {
      const form = document.createElement("button");
      form.type = "button";
      form.className = "bd-tray-item bd-tray-form";
      form.title = "Add this node's settings as one section of cards";
      form.textContent = "All settings as a section";
      form.onclick = () => targetBoard().addNodeSection(g.key.slice("node:".length));
      d.append(form);
    }
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
        b.onclick = () => targetBoard().addRef(it.ref);
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
    S.draft = makeDraft(S.app);
    S.page.setContainer(S.app.root, S.draft);
    scheduleResolve(0);
    updateState();
  } catch (err) {
    showMessage("page-error", `Could not update from the graph: ${errText(err)}`);
  }
}

/* ---- run / stop --------------------------------------------------------------- */

async function run(mode = "auto") {
  showMessage("page-error", "");
  const values = valuesPayload({ blocks: rootBlocks(), graph: S.draft }, statusByBlock(S.resolution));
  try {
    const exec = await api(`/apps/${enc(S.name)}/run`, { method: "POST", body: { values, mode } });
    const waiting = exec.status === "planned";
    log(waiting ? `Run #${exec.execution_id} is waiting its turn.` : `Run #${exec.execution_id} started from "${S.name}".`, "success");
    if (!(waiting && executionActive())) {
      S.execution = { ...exec, done: 0, total: 0 };
      schedulePoll(400);
    }
    S.page.refreshDisplays();
    updateState();
    refreshRuns();
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
    refreshRuns();
  } catch (err) {
    showMessage("page-error", `Could not stop: ${errText(err)}`);
  }
}

/** A waiting run is cancelled the way a running one is stopped. */
async function cancelRun(id) {
  try {
    await api(`/graphs/executions/${id}/stop`, { method: "POST" });
    log(`Run #${id} cancelled.`, "warn");
  } catch (err) {
    showMessage("page-error", `Could not cancel: ${errText(err)}`);
  }
  await refreshRuns();
}

/** Put a past run's values back on the page; it still has to be run. */
function runAgain(run) {
  const values = run.origin && run.origin.values;
  if (!values) return;
  const changed = applyValues({ blocks: rootBlocks(), graph: S.draft }, values);
  S.page.setContainer(S.app.root, S.draft);
  scheduleResolve(0);
  updateState();
  log(changed ? `Values of run #${run.execution_id} are on the page: ${changed} changed.` : `The page already has the values of run #${run.execution_id}.`, "info");
}

/** The history and queue cards' data, and: when the run this page followed has ended and the next one of this app has started, follow that. */
async function refreshRuns() {
  try {
    const [mine, all] = await Promise.all([
      api(`/graphs/executions?limit=30&app=${enc(S.name)}`),
      api("/graphs/executions?limit=100"),
    ]);
    S.runs = {
      history: mine.executions || [],
      planned: (all.executions || []).filter((r) => r.status === "planned").sort((a, b) => a.execution_id - b.execution_id),
    };
    const next = S.runs.history.find((r) => r.status === "running" || r.status === "queued");
    if (next && !executionActive()) {
      S.execution = { ...next, done: 0, total: 0 };
      schedulePoll(0);
    }
  } catch (err) {
    log(`Could not read the run list: ${errText(err)}`, "warn");
  }
  if (S.page) S.page.refreshDisplays();
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
    S.page.refreshDisplays();
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
  S.page.refreshDisplays();
  updateState();
  refreshRuns();
  if (again && executionActive()) schedulePoll(1500);
}

/* ======================================================================== */

if (pageName === null) showList();
else showPage();
