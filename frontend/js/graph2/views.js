/* ---------------------------------------------------------------------------
   graph2/views.js -- views: boards of blocks that live inside the graph document (`views`, graph-json.md).

   A view is `{id, title, mode: "page" | "monitor", columns, blocks}`. A monitor view holds live displays; a page view
   also holds fields, so it is the interface a person uses without meeting the canvas. Blocks point at nodes by uid.

   One truth: a field block edits the graph's own node parameter or setting through the document (the canvas and the
   page show the same value at once), and Run is the editor's Run. The board itself is the one the templates page uses
   (board/board.js); this file only binds it to the open document: a stand-in "draft graph" whose parameter objects
   write through `GraphDoc`, and a resolve call (`POST /templates/resolve`) that says which blocks still have a target.
   --------------------------------------------------------------------------- */

import { api } from "../api.js";
import { Board, settleLayout } from "../board/board.js";
import { buildAvailable, nodeRef } from "../board/model.js";
import { errText } from "../lib/errors.js";
import { h } from "../lib/dom.js";

const SEED_MODULES = 5; // a new monitor view starts with this many displays of each node that has some

export class Views {
  /**
   * @param root where the panel goes  @param doc the open GraphDoc  @param live shared LiveStreams
   * @param hooks {onRun, onStop, onNote, execution}
   */
  constructor(root, doc, live, hooks) {
    this.doc = doc;
    this.hooks = hooks;
    this.currentId = null;
    this.mode = "use";
    this.modules = null;
    this.statuses = {};
    this._seed = null;
    this._seq = 0;
    this._timer = 0;
    this._own = false;
    this._active = false;

    this.pick = h("select", { class: "vw-pick", "aria-label": "View", onchange: () => this.open(this.pick.value) });
    this.useBtn = h("button", { class: "tpl-seg-btn active", type: "button", text: "Use", onclick: () => this.setMode("use") });
    this.editBtn = h("button", { class: "tpl-seg-btn", type: "button", text: "Edit", onclick: () => this.setMode("edit") });
    this.runBtn = h("button", { class: "btn btn-start btn-small", type: "button", text: "▶ Run", onclick: () => hooks.onRun() });
    this.stopBtn = h("button", { class: "btn btn-danger btn-small", type: "button", text: "■ Stop", onclick: () => hooks.onStop() });
    this.state = h("span", { class: "tpl-state" });
    this.bar = h("div", { class: "vw-bar" },
      this.pick,
      h("button", { class: "btn btn-secondary btn-small", type: "button", text: "New monitor", onclick: () => this.create("monitor") }),
      h("button", { class: "btn btn-secondary btn-small", type: "button", text: "New page", onclick: () => this.create("page") }),
      h("button", { class: "btn btn-secondary btn-small", type: "button", text: "Rename", onclick: () => this.rename() }),
      h("button", { class: "btn btn-kill btn-small", type: "button", text: "Delete", onclick: () => this.remove() }),
      h("span", { class: "tpl-seg", role: "group", "aria-label": "Mode" }, this.useBtn, this.editBtn),
      this.runBtn, this.stopBtn, this.state);
    this.issues = h("div", { class: "error-box", role: "alert", hidden: true });
    this.search = h("input", { class: "cfg-input", type: "search", placeholder: "Search…", "aria-label": "Search available blocks", oninput: () => this.renderTray() });
    this.advanced = h("input", { type: "checkbox", onchange: () => this.renderTray() });
    this.trayList = h("div");
    this.columns = h("input", { type: "number", min: "1", max: "24", "aria-label": "Grid columns", onchange: () => this.setColumns() });
    this.tray = h("aside", { class: "tpl-tray", hidden: true, "aria-label": "Add blocks" },
      h("div", { class: "tpl-tray-head" }, h("strong", { text: "Add blocks" }), h("label", { class: "tpl-adv" }, this.advanced, " advanced")),
      this.search, this.trayList, h("div", { class: "tpl-tray-foot" }, "Columns ", this.columns));
    this.boardEl = h("div", { class: "tpl-board" });
    this.empty = h("div", { class: "console-line info", text: "No views yet. A monitor view watches the live displays of your nodes; a page view also edits their values." });
    this.work = h("div", { class: "tpl-work" }, this.boardEl, this.tray);
    root.append(this.bar, this.issues, this.empty, this.work);

    this.board = new Board(this.boardEl, {
      live, ctx: { execution: () => hooks.execution() },
      onChange: (reason) => this.changed(reason), onNote: (msg, kind) => hooks.onNote(msg, kind || "info"),
    });
    for (const type of ["reset", "node:add", "node:remove", "edge:add", "edge:remove", "settings"]) doc.on(type, () => this.resolveLater(400));
    doc.on("node:params", (e) => { if (e.origin === "view") return; this.board.refreshFields(); this.resolveLater(500); });
    doc.on("views", () => { if (!this._own) this.syncList(); });
    doc.on("reset", () => this.syncList());
    this.syncList();
  }

  views() { return this.doc.views; }
  current() { return this.views().find((v) => v.id === this.currentId) || null; }

  /* ---- the list ---- */

  syncList() {
    const views = this.views();
    if (!views.some((v) => v.id === this.currentId)) this.currentId = views[0]?.id ?? null;
    this.pick.replaceChildren(...views.map((v) => h("option", { value: v.id, text: `${v.title} (${v.mode})`, selected: v.id === this.currentId })));
    this.show();
  }

  open(id) {
    this.currentId = id;
    this.show();
  }

  create(mode) {
    const views = [...this.views()];
    let n = views.length + 1;
    while (views.some((v) => v.id === `v${n}`)) n++;
    const view = { id: `v${n}`, title: mode === "page" ? "Page" : "Monitor", mode, columns: 12, blocks: [] };
    this._own = true;
    this.doc.setViews([...views, view]);
    this._own = false;
    this._seed = mode === "monitor" ? view.id : null;
    this.currentId = view.id;
    this.mode = "edit";
    this.syncList();
  }

  rename() {
    const view = this.current();
    const title = view && window.prompt("Name of this view", view.title);
    if (!title?.trim()) return;
    this.write((views) => { views.find((v) => v.id === view.id).title = title.trim(); });
    this.syncList();
  }

  remove() {
    const view = this.current();
    if (!view || !window.confirm(`Delete the view "${view.title}"?`)) return;
    this.write((views) => views.splice(views.findIndex((v) => v.id === view.id), 1));
    this.currentId = null;
    this.syncList();
  }

  /** Change the views through a copy, so the document sees one `views` event. */
  write(edit) {
    const views = JSON.parse(JSON.stringify(this.views()));
    edit(views);
    this._own = true;
    this.doc.setViews(views);
    this._own = false;
  }

  /* ---- showing one ---- */

  show() {
    const view = this.current();
    this.empty.hidden = !!view;
    this.work.hidden = !view;
    for (const b of [this.useBtn, this.editBtn, this.runBtn, this.stopBtn]) b.hidden = !view;
    if (!view) { this.board.setDoc({ kind: "monitor", columns: 12, blocks: [], graph: { nodes: [] } }); return; }
    this.runBtn.hidden = view.mode !== "page";
    this.stopBtn.hidden = view.mode !== "page" || !this._active;
    this.columns.value = String(view.columns);
    this.board.setDoc(this.standIn(view.id));
    this.board.setMode(this.mode);
    this.board.setModules(this.modules);
    this.board.setStatuses(this.statuses);
    this.setMode(this.mode);
    this.resolveLater(0);
  }

  setMode(mode) {
    this.mode = mode;
    this.useBtn.classList.toggle("active", mode === "use");
    this.editBtn.classList.toggle("active", mode === "edit");
    this.tray.hidden = mode !== "edit";
    this.board.setMode(mode);
    this.renderTray();
  }

  /** What the board edits: this view's own blocks, and a draft graph whose values write through to the document. */
  standIn(id) {
    const doc = this.doc;
    const view = () => this.views().find((v) => v.id === id);
    const params = (node) => new Proxy(node.params, {
      set: (_t, name, value) => { doc.setParam(node.uid, String(name), value, "view"); return true; },
      deleteProperty: (_t, name) => { doc.setParam(node.uid, String(name), undefined, "view"); return true; },
    });
    const memory = new Proxy({}, {
      get: (_t, key) => doc.settings[`memory.${String(key)}`],
      set: (_t, key, value) => { doc.setSetting(`memory.${String(key)}`, value); return true; },
      deleteProperty: (_t, key) => { doc.setSetting(`memory.${String(key)}`, null); return true; },
    });
    return {
      get kind() { return view().mode === "page" ? "template" : "monitor"; },
      get columns() { return view().columns; },
      set columns(n) { view().columns = n; },
      get blocks() { return view().blocks; },
      set blocks(list) { view().blocks = list; },
      source_graph: null,
      graph: { get nodes() { return [...doc.nodes.values()].map((n) => ({ id: n.uid, params: params(n) })); }, memory },
    };
  }

  changed(reason) {
    if (reason === "blocks" || reason === "layout") {
      this._own = true;
      this.doc.setViews(this.views());
      this._own = false;
    }
    if (reason === "blocks") this.resolveLater(0);
    if (reason === "value") this.resolveLater(600);
  }

  setColumns() {
    const view = this.current();
    if (!view) return;
    const n = Math.max(1, Math.min(24, parseInt(this.columns.value, 10) || view.columns));
    this.columns.value = String(n);
    if (n === view.columns) return;
    view.columns = n;
    view.blocks = settleLayout(view.blocks, n);
    this.board.applyLayout();
    this.changed("layout");
  }

  /* ---- what the blocks bind to ---- */

  resolveLater(delay = 300) {
    clearTimeout(this._timer);
    this._timer = setTimeout(() => this.resolve(), delay);
  }

  async resolve() {
    const view = this.current();
    if (!view) return;
    const seq = ++this._seq;
    try {
      const res = await api("/templates/resolve", {
        method: "POST",
        body: { doc: { format: 1, kind: view.mode === "page" ? "template" : "monitor", columns: view.columns, source_graph: null, graph: this.doc.runPayload(), blocks: view.blocks } },
      });
      if (seq !== this._seq) return;
      this.modules = res.modules;
      this.statuses = Object.fromEntries((res.blocks || []).map((b) => [b.id, { status: b.status, detail: b.detail }]));
      this.board.setModules(this.modules);
      this.board.setStatuses(this.statuses);
      this.seedIfNew();
      this.renderTray();
      this.issues.hidden = true;
    } catch (err) {
      this.issues.textContent = `Could not check the view: ${errText(err)}`;
      this.issues.hidden = false;
    }
  }

  /** A new monitor view starts with the first displays of every node that has some, so it is useful at once. */
  seedIfNew() {
    if (this._seed !== this.currentId || !this.modules) return;
    this._seed = null;
    for (const node of this.modules.nodes || []) {
      for (const d of (node.displays || []).slice(0, SEED_MODULES)) this.board.addRef(nodeRef(node.id, "display", d.name));
    }
  }

  renderTray() {
    const view = this.current();
    if (!view || this.mode !== "edit" || !this.modules) return;
    const needle = this.search.value.trim().toLowerCase();
    let groups = buildAvailable(this.modules, view.blocks, { showAdvanced: this.advanced.checked });
    if (view.mode === "monitor") groups = groups.map((g) => ({ ...g, items: g.items.filter((i) => i.kind === "display") }));
    groups = groups
      .map((g) => ({ ...g, items: needle ? g.items.filter((i) => `${i.label} ${g.title} ${i.sub}`.toLowerCase().includes(needle)) : g.items }))
      .filter((g) => g.items.length);
    const open = new Map([...this.trayList.querySelectorAll("details")].map((d) => [d.dataset.key, d.open]));
    const note = h("button", { class: "tpl-tray-item", type: "button", title: "Free text on the page, for whoever uses it", onclick: () => this.board.addNote() },
      h("span", { text: "Note" }), h("span", { class: "tpl-tray-tag", text: "+" }));
    this.trayList.replaceChildren(note, ...(groups.length ? groups.map((g) => this.trayGroup(g, open, groups.length, !!needle)) : [h("div", { class: "tpl-tray-empty", text: needle ? "Nothing matches." : "Nothing more to add." })]));
  }

  trayGroup(g, open, count, searching) {
    const details = h("details", { "data-key": g.key });
    details.open = open.has(g.key) ? open.get(g.key) : g.scope === "graph" || count <= 3 || searching;
    details.append(h("summary", { text: g.title }));
    for (const kind of ["display", "field"]) {
      const items = g.items.filter((i) => i.kind === kind);
      if (!items.length) continue;
      details.append(h("div", { class: "tpl-tray-cap", text: kind === "display" ? "Live views" : "Settings" }));
      for (const it of items) {
        details.append(h("button", { class: "tpl-tray-item", type: "button", disabled: it.placed, title: it.sub || it.label, onclick: () => this.board.addRef(it.ref) },
          h("span", { text: it.label }), h("span", { class: "tpl-tray-tag", text: it.placed ? "added" : it.advanced ? "advanced" : "+" })));
      }
    }
    return details;
  }

  /** The editor's run state, so a page's Stop shows only while something runs. */
  running(active) {
    this._active = active;
    const view = this.current();
    this.stopBtn.hidden = !view || view.mode !== "page" || !active;
    this.board.refreshDisplays();
  }

  shown() {
    this.board.refreshDisplays();
  }
}
