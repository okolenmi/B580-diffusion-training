/* One app's page (/apps/<name>): use mode is the interface a person meets (change values, Run, watch live cards); edit
   mode unlocks the layout and a tray of everything the app's graph offers.

   An app owns a graph (a copy made when the app was created) plus a page of cards over it (docs/contracts/apps.md). The page edits
   the values and the layout; "Edit graph" opens the app's own graph in the graph editor's app mode.

   This class only coordinates. The document and its saved state are `AppDocument`, the selection `CardSelection`, runs
   `RunTracker`, the tray and toolbar their own classes; what a card can bind to is the backend's answer
   (`POST /boards/resolve`), re-asked after edits. Run is refused client-side only for the one reason the server would also
   refuse: the page shows a field the *saved* app does not expose. */

import { ApiError } from "../api.js";
import { ContainerView } from "../board/container.js";
import { History } from "../board/history.js";
import { Inspector } from "../board/inspector.js";
import { LiveStreams } from "../board/live.js";
import {
  applyValues, autoPage, collectIds, groupCards, locateCard, moveCard, moveTargets, refineContainer, statusByBlock, titleFor, valuesPayload, walkCards,
} from "../board/model.js";
import { el, showMessage } from "../lib/dom.js";
import { errText } from "../lib/errors.js";
import { log } from "../lib/log.js";
import { AppDocument } from "./document.js";
import { ValueKeeper } from "./value_keeper.js";
import { RunTracker } from "./run_tracker.js";
import { CardSelection } from "./selection.js";
import { Toolbar } from "./toolbar.js";
import { Tray } from "./tray.js";

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

function describeMemory(details) {
  if (!details || typeof details !== "object") return "";
  return Object.entries(details)
    .filter(([, v]) => typeof v === "number" || typeof v === "string")
    .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`)
    .join("\n");
}

const apiDetails = (err) => (err instanceof ApiError && Array.isArray(err.details) ? `\n${err.details.join("\n")}` : "");

export class AppPage {
  constructor(name, query) {
    this.query = query;
    this.doc = new AppDocument(name);
    this.selection = new CardSelection();
    this.history = new History();
    this.live = new LiveStreams();
    this.toolbar = new Toolbar();
    this.tray = new Tray({ targetBoard: () => this.targetBoard() });
    this.tracker = new RunTracker({
      name,
      hooks: { displays: () => this.view && this.view.refreshDisplays(), state: () => this.updateState() },
      notify: (msg, kind) => log(msg, kind),
    });
    this.keeper = new ValueKeeper({
      name,
      values: () => valuesPayload({ blocks: this.doc.rootBlocks(), graph: this.doc.draft }, statusByBlock(this.resolution), { keepCleared: true }),
      enabled: () => !!this.doc.app && !this.doc.isNew,
      onError: (err) => log(`The values on the page could not be kept: ${errText(err)}`, "warn"),
    });
    this.mode = "use";
    this.resolution = null;
    this.resolveSeq = 0;
    this.resolveTimer = 0;
    this.view = null;       // the ContainerView over doc.app.root
    this.inspector = null;
    window.__app = this;    // inspectable from the console when something looks wrong
  }

  async show() {
    el("view-list").hidden = true;
    el("view-page").hidden = false;
    el("btn-back").hidden = false;
    el("app-title").textContent = this.doc.name;
    document.title = `${this.doc.name} · Apps`;
    try {
      this.resolution = await this.doc.load(this.query);
    } catch (err) {
      showMessage("page-error", `Could not open "${this.doc.name}": ${errText(err)}`);
      el("toolbar").hidden = true;
      return;
    }
    this.#putBackValues();
    this.#buildView();
    this.view.setModules(this.resolution.modules);
    this.view.setContainer(this.doc.app.root, this.doc.draft);
    this.history.reset(JSON.stringify(this.doc.app.root));
    this.toolbar.showColumns(this.doc.app.root.columns);
    this.applyResolution();
    this.setMode(this.doc.isNew ? "edit" : "use");
    this.#wireToolbar();
    this.describePage();
    await this.tracker.adopt();
    await this.tracker.refresh();
    this.updateState();
  }

  /** What was typed here last time goes over the document's values, for the cards the page has. */
  #putBackValues() {
    const kept = this.doc.values;
    if (!kept || !Object.keys(kept).length) return;
    applyValues({ blocks: this.doc.rootBlocks(), graph: this.doc.draft }, kept);
    this.keeper.known(kept);
    this.scheduleResolve(0);
  }

  #buildView() {
    const tracker = this.tracker;
    this.view = new ContainerView(el("board"), {
      live: this.live,
      ctx: {
        execution: () => tracker.execution,
        runs: () => tracker.runs,
        runAgain: (run) => this.runAgain(run),
        cancelRun: (id) => tracker.cancel(id).catch((err) => showMessage("page-error", `Could not cancel: ${errText(err)}`)),
        canRunAgain: () => this.mode === "use" && !this.doc.isNew,
      },
      onNote: (msg, kind) => log(msg, kind || "info"),
      ids: () => collectIds(this.doc.rootBlocks()),
      onSelect: (event) => this.onSelect(event),
      drag: { allows: (id, board) => this.canDropOn(id, board), drop: (id, board, cell) => this.dropOn(id, board, cell) },
    }, { onChange: (reason) => this.onBoardChange(reason) });
    const each = (ids, fn) => { for (const id of ids) { const hit = this.view.find(id); if (hit) fn(hit, id); } };
    this.inspector = new Inspector(el("inspector"), {
      patch: (id, patch) => { each([id], (hit) => hit.board.update(id, patch, "title")); this.renderInspector(); },
      move: (id, owner, tabId) => this.moveSelected(id, owner, tabId),
      remove: (id) => each([id], (hit) => hit.board.removeId(id)),
      group: () => this.groupSelected(),
      removeMany: () => each([...this.selection.picked], (hit, id) => hit.board.removeId(id)),
      addTab: (id) => each([id], (hit) => { const entry = hit.board.entries.get(id); if (entry && entry.view) entry.view.addTab(); }),
    });
  }

  #wireToolbar() {
    this.toolbar.wire({
      mode: (mode) => this.setMode(mode),
      save: () => this.save(),
      run: (mode) => this.run(mode),
      queue: () => this.run("queue"),
      stop: () => this.tracker.stop().catch((err) => showMessage("page-error", `Could not stop: ${errText(err)}`)),
      remove: () => this.remove(),
      graph: () => this.openGraphEditor(),
      undo: () => this.stepHistory(-1),
      redo: () => this.stepHistory(1),
      note: () => this.targetBoard().addNote(),
      heading: () => this.targetBoard().addNote("heading"),
      section: () => this.targetBoard().addSection(),
      columns: (raw) => this.setColumns(raw),
      fill: () => this.fillPage(),
    });
    this.tray.onChange(() => this.renderTray());
    document.addEventListener("keydown", (ev) => {
      if (this.mode === "edit" && ev.key === "Delete" && !ev.target.closest("input, textarea, select, [contenteditable]")) {
        if (this.removeSelected()) ev.preventDefault();
        return;
      }
      if (this.mode !== "edit" || !(ev.ctrlKey || ev.metaKey) || ev.target.closest("input, textarea, select")) return;
      const key = ev.key.toLowerCase();
      if (key === "z") { ev.preventDefault(); this.stepHistory(ev.shiftKey ? 1 : -1); }
      else if (key === "y") { ev.preventDefault(); this.stepHistory(1); }
    });
    window.addEventListener("beforeunload", (e) => {
      if (this.doc.app && this.doc.isDirty()) { e.preventDefault(); e.returnValue = ""; }
    });
  }

  /** Remove the selected cards (the Delete key); false when none is selected. */
  removeSelected() {
    const ids = this.selection.sel ? this.selection.ids() : [];
    let removed = false;
    for (const id of ids) {
      const hit = this.view.find(id);
      if (hit) { hit.board.removeId(id); removed = true; }
    }
    return removed;
  }

  describePage() {
    const { doc } = this;
    const bits = [];
    if (doc.version) bits.push(`version ${doc.version}`);
    if (doc.app.source_graph) bits.push(`started from graph “${doc.app.source_graph}”`);
    bits.push(`${doc.app.graph.nodes.length} node${doc.app.graph.nodes.length === 1 ? "" : "s"}`);
    const n = [...walkCards(doc.rootBlocks())].length;
    bits.push(`${n} card${n === 1 ? "" : "s"}`);
    el("app-sub").textContent = bits.join(" · ");
  }

  updateState() {
    if (this.doc.app) this.toolbar.render(this.doc, this.tracker.active);
  }

  /* ---- resolving: what a card can bind to is the backend's answer ---------------- */

  scheduleResolve(delay = 300) {
    clearTimeout(this.resolveTimer);
    this.resolveTimer = setTimeout(async () => {
      const seq = ++this.resolveSeq;
      try {
        const res = await this.doc.resolve();
        if (seq !== this.resolveSeq) return; // a newer edit superseded this answer
        this.resolution = res;
        this.view.setModules(res.modules);
        this.applyResolution();
        this.renderTray();
      } catch (err) {
        showMessage("page-error", `Could not check the app: ${errText(err)}`);
      }
    }, delay);
  }

  applyResolution() {
    const statuses = {};
    for (const b of this.resolution.blocks) statuses[b.id] = { status: b.status, detail: b.detail };
    this.view.setStatuses(statuses);
    renderIssues(this.resolution.issues || []);
    this.renderTray();
    this.updateState();
  }

  /* ---- board events, selection, inspector, undo ---------------------------------- */

  onBoardChange(reason) {
    if (reason === "blocks") {
      this.scheduleResolve(0);
      this.describePage();
      this.selection.prune((id) => this.view.find(id));
      this.renderInspector();
      this.renderTray();
    }
    if (reason === "value") { this.scheduleResolve(600); this.keeper.touch(); }
    if (reason !== "value") this.history.record(JSON.stringify(this.doc.app.root), { merge: reason === "text" });
    if (reason === "layout") this.renderInspector();
    this.updateState();
  }

  targetBoard() {
    const { target } = this.selection;
    return target && target.alive ? target : this.view.activeBoard;
  }

  onSelect({ board, id, view, extend }) {
    this.selection.select(id, view ? view.activeBoard : board, extend);
    this.markSelection();
    this.renderInspector();
    this.toolbar.showColumns(this.targetBoard().view.container.columns);
  }

  markSelection() {
    const { selection } = this;
    this.view.mark(selection.many ? selection.picked : selection.sel, this.targetBoard());
  }

  renderInspector() {
    const { selection, doc } = this;
    const hit = selection.sel && this.mode === "edit" ? this.view.find(selection.sel) : null;
    if (!hit) { this.inspector.hide(); return; }
    if (selection.many) {
      this.inspector.showMany({ count: selection.picked.size, canGroup: groupCards(doc.app.root, [...selection.picked]) !== doc.app.root });
      this.markSelection();
      return;
    }
    const { block } = hit;
    this.inspector.show({
      block,
      defaultTitle: titleFor({ ...block, title: null }, this.resolution && this.resolution.modules).defaultTitle,
      targets: moveTargets(doc.app.root, block.id),
      here: locateCard(doc.app.root, block.id),
    });
    this.markSelection();
  }

  /** Wrap the picked cards (or the selected one) in a new box, and select the box. */
  groupSelected() {
    const ids = this.selection.ids();
    const root = groupCards(this.doc.app.root, ids);
    if (root === this.doc.app.root) { log("Those cards cannot be boxed together (one tab only, and boxes nest three deep at most).", "warn"); return; }
    const box = locateCard(root, ids[0]).owner;
    this.applyRoot(root);
    this.selection.only(box);
    this.markSelection();
    this.renderInspector();
  }

  /** The card that holds `board` ("" for the page itself) and the tab it shows: where a card dropped there goes. */
  placeOf(board) {
    const { view } = board;
    if (view === this.view) return { owner: "", tabId: view.activeId };
    for (const b of this.view.boards()) {
      for (const [id, entry] of b.entries) if (entry.view === view) return { owner: id, tabId: view.activeId };
    }
    return null;
  }

  canDropOn(id, board) {
    const place = this.placeOf(board);
    const here = locateCard(this.doc.app.root, id);
    if (!place || !here || (place.owner === here.owner && place.tabId === here.tabId)) return false;
    return moveTargets(this.doc.app.root, id).some((t) => t.owner === place.owner && t.tabId === place.tabId);
  }

  /** A card dragged out of its board and let go over another: it goes there, its corner at the cell it was dropped on. */
  dropOn(id, board, cell) {
    const place = this.placeOf(board);
    if (!place) return;
    const root = moveCard(this.doc.app.root, id, place.owner, place.tabId, cell);
    if (root !== this.doc.app.root) this.applyRoot(root);
  }

  moveSelected(id, owner, tabId) {
    const root = moveCard(this.doc.app.root, id, owner, tabId);
    if (root !== this.doc.app.root) this.applyRoot(root);
  }

  /** Replace the whole page (a move, an undo): rebuild the boards over it and re-check it with the server. */
  applyRoot(root, { record = true } = {}) {
    this.doc.app.root = root;
    this.view.setContainer(root, this.doc.draft);
    this.selection.reset((id) => this.view.find(id));
    if (record) this.history.record(JSON.stringify(root));
    this.scheduleResolve(0);
    this.describePage();
    this.renderInspector();
    this.renderTray();
    this.toolbar.showHistory(this.history);
    this.toolbar.showColumns(this.targetBoard().view.container.columns);
    this.updateState();
  }

  stepHistory(direction) {
    const snap = direction < 0 ? this.history.undo() : this.history.redo();
    if (snap !== null) this.applyRoot(JSON.parse(snap), { record: false });
  }

  setColumns(raw) {
    const view = this.targetBoard().view;
    const n = Math.max(1, Math.min(24, parseInt(raw, 10) || view.container.columns));
    this.toolbar.showColumns(n);
    view.setColumns(n);
  }

  setMode(mode) {
    this.mode = mode;
    this.toolbar.showMode(mode);
    this.view.setMode(mode);
    if (mode !== "edit") { this.selection.clear(); this.view.mark(new Set(), null); }
    this.renderInspector();
    this.renderTray();
    this.toolbar.showHistory(this.history);
  }

  renderTray() {
    if (this.doc.app && this.resolution && this.mode === "edit") this.tray.render(this.resolution.modules, this.doc.rootBlocks());
    el("add-auto").hidden = !(this.doc.app && this.doc.hasNodes && ![...walkCards(this.doc.rootBlocks())].length);
  }

  /* ---- save / delete / graph ------------------------------------------------------ */

  async save() {
    showMessage("page-error", "");
    try {
      const out = await this.doc.save();
      this.keeper.known({});
      history.replaceState(null, "", `/apps/${encodeURIComponent(this.doc.name)}`);
      this.describePage();
      this.updateState();
      log(`Saved "${out.name}" (version ${out.version}, ${out.card_count} cards).`, "success");
    } catch (err) {
      showMessage("page-error", `Could not save: ${errText(err)}${apiDetails(err)}`);
    }
  }

  async remove() {
    if (!window.confirm(`Delete "${this.doc.name}"? This removes the app, not the graph it was built on.`)) return;
    try {
      await this.doc.remove();
    } catch (err) {
      return showMessage("page-error", `Could not delete: ${errText(err)}`);
    }
    location.href = "/apps";
  }

  /** The app's own graph in the graph editor (app mode). What is on the page is saved first: the editor reads the saved app. */
  async openGraphEditor() {
    if (this.doc.isNew || this.doc.needsSave()) {
      await this.save();
      if (this.doc.isNew || this.doc.needsSave()) return; // the save said why it did not work
    }
    location.href = `/graph?app=${encodeURIComponent(this.doc.name)}`;
  }

  /** A page of cards for everything the graph offers, for an app whose page is still empty. */
  fillPage() {
    if (!this.resolution || [...walkCards(this.doc.rootBlocks())].length) return;
    const root = refineContainer({ ...this.doc.app.root, tabs: autoPage(this.resolution.modules, this.doc.app.root.columns) });
    this.applyRoot(root);
  }

  /* ---- running --------------------------------------------------------------------- */

  async run(mode = "auto") {
    showMessage("page-error", "");
    const values = valuesPayload({ blocks: this.doc.rootBlocks(), graph: this.doc.draft }, statusByBlock(this.resolution));
    try {
      await this.tracker.start(values, mode);
    } catch (err) {
      if (err instanceof ApiError && err.code === "graph_invalid") {
        const issues = Array.isArray(err.details) ? err.details : (err.details && err.details.issues) || [];
        renderIssues(issues, `${issues.length} problem${issues.length === 1 ? "" : "s"} — the run was refused`);
        el("issues").open = true;
        showMessage("page-error", "The graph is not valid with these values — see the list below.");
      } else if (err instanceof ApiError && err.code === "graph_execution_active") {
        showMessage("page-error", "Another run is active — stop it first (in the Graph Editor, or here once it shows).");
        this.tracker.adopt();
      } else if (err instanceof ApiError && err.code === "memory_unavailable") {
        showMessage("page-error", `${errText(err)}\n${describeMemory(err.details)}`);
      } else {
        showMessage("page-error", `${errText(err)}${apiDetails(err)}`);
      }
    }
  }

  /** Put a past run's values back on the page; it still has to be run. */
  runAgain(run) {
    const values = run.origin && run.origin.values;
    if (!values) return;
    const changed = applyValues({ blocks: this.doc.rootBlocks(), graph: this.doc.draft }, values);
    this.view.setContainer(this.doc.app.root, this.doc.draft);
    this.scheduleResolve(0);
    this.keeper.touch();
    this.updateState();
    log(changed ? `Values of run #${run.execution_id} are on the page: ${changed} changed.` : `The page already has the values of run #${run.execution_id}.`, "info");
  }
}
