/* ---------------------------------------------------------------------------
   board/container.js -- a tab strip over one Board per tab: the App's page
   and every section / tabbed card are the same thing.

   A container is `{columns, tabs: [{id, title, blocks}]}` (backend/domain/board.py). The view edits it *in place*: the active tab's
   Board works on a small document `{columns, blocks, graph}` and every change is written back into the tab before it is reported
   up, so whoever holds the container object always holds the current page.

   Only the active tab has a live Board. Switching tabs disposes the old one (its live-stream subscriptions with it) and builds the
   next, which keeps a page with many charts cheap and means a hidden tab never draws.

   Pure arithmetic (adding / moving / removing tabs, moving cards between tabs) is in model.js; this file only turns it into DOM.
   --------------------------------------------------------------------------- */

import { Board } from "./board.js";
import { addTab, MAX_TABS, moveTab, removeTab, renameTab, withColumns } from "./model.js";

const h = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

function tool(label, title) {
  const b = h("button", "bd-tab-tool", label);
  b.type = "button";
  b.title = title;
  b.setAttribute("aria-label", title);
  return b;
}

export class ContainerView {
  /**
   * @param {HTMLElement} host
   * @param {object} shared  what every board of the page shares: `{live, ctx, onNote, ids, onSelect}`
   * @param {object} [opts]
   * @param {boolean} [opts.nested]  inside a card: width does not decide the phone layout, the page's does
   * @param {(reason: string) => void} [opts.onChange]
   */
  constructor(host, shared, { nested = false, onChange = () => {} } = {}) {
    this.host = host;
    this.shared = shared;
    this.nested = nested;
    this.onChange = onChange;
    this.modules = null;
    this.statuses = {};
    this.mode = "use";
    this.narrow = false;
    this.container = null;
    this.graph = null;
    this.activeId = null;
    this.board = null;
    this.doc = null;

    this.el = h("div", "bd-container");
    this.strip = h("div", "bd-tabs");
    this.strip.setAttribute("role", "tablist");
    this.stage = h("div", "bd-stage");
    this.el.append(this.strip, this.stage);
    host.append(this.el);
  }

  destroy() {
    this._dropBoard();
    this.el.remove();
  }

  /* ---- inputs ----------------------------------------------------------- */

  /** Show `container` (edited in place from now on). `graph` is the draft the field cards write into. */
  setContainer(container, graph) {
    this.container = container;
    this.graph = graph;
    if (!container.tabs.some((t) => t.id === this.activeId)) this.activeId = container.tabs[0].id;
    this._showTab();
  }

  setModules(modules) { this.modules = modules; if (this.board) this.board.setModules(modules); }

  setStatuses(statuses) { this.statuses = statuses || {}; if (this.board) this.board.setStatuses(this.statuses); }

  setMode(mode) {
    this.mode = mode;
    this._renderStrip();
    if (this.board) this.board.setMode(mode);
  }

  setNarrow(narrow) {
    this.narrow = narrow;
    if (this.board) this.board.setNarrow(narrow);
  }

  refreshDisplays() { if (this.board) this.board.refreshDisplays(); }

  refreshFields() { if (this.board) this.board.refreshFields(); }

  /** The active tab's board: where "add" lands unless a card says otherwise. */
  get activeBoard() { return this.board; }

  /** This view's live boards, outermost first. */
  *boards() { if (this.board) yield* this.board.boards(); }

  find(id) { return this.board ? this.board.find(id) : null; }

  mark(selected, targetBoard) { if (this.board) this.board.mark(selected, targetBoard); }

  /** Re-lay every tab for a new column count. */
  setColumns(columns) {
    if (columns === this.container.columns) return;
    Object.assign(this.container, withColumns(this.container, columns));
    this._showTab();
    this.onChange("layout");
  }

  /** A new empty tab, shown. */
  addTab() {
    if (this.container.tabs.length >= MAX_TABS) return;
    const tabs = addTab(this.container.tabs);
    this._setTabs(tabs, tabs[tabs.length - 1].id);
  }

  /** Switch to a tab (in edit mode the tab's board also becomes the place new cards go). */
  select(id) {
    if (!this.container.tabs.some((t) => t.id === id)) return;
    this.activeId = id;
    this._showTab();
    if (this.mode === "edit" && this.shared.onSelect) this.shared.onSelect({ board: this.board, id: null, view: this });
  }

  /* ---- tabs ------------------------------------------------------------- */

  _showTab() {
    this._dropBoard();
    const tab = this.container.tabs.find((t) => t.id === this.activeId);
    this.doc = { columns: this.container.columns, blocks: tab.blocks, graph: this.graph };
    this.board = new Board(this.stage, {
      ...this.shared, nested: this.nested, onChange: (reason) => this._changed(reason),
    });
    this.board.view = this;
    this.board.modules = this.modules;
    this.board.statuses = this.statuses;
    this.board.narrow = this.narrow;
    this.board.setMode(this.mode);
    this.board.setDoc(this.doc);
    this._renderStrip();
  }

  _dropBoard() {
    if (this.board) { this.board.destroy(); this.board = null; }
  }

  _changed(reason) {
    const tab = this.container.tabs.find((t) => t.id === this.activeId);
    if (tab) tab.blocks = this.doc.blocks;
    this.container.columns = this.doc.columns;
    this.onChange(reason);
  }

  _setTabs(tabs, activeId = this.activeId) {
    this.container.tabs = tabs;
    this.activeId = activeId;
    this._showTab();
    this.onChange("blocks");
  }

  _renderStrip() {
    const { container } = this;
    this.strip.replaceChildren();
    if (!container) return;
    const editing = this.mode === "edit";
    this.strip.hidden = container.tabs.length < 2 && !editing;
    for (const tab of container.tabs) {
      const b = h("button", "bd-tab", tab.title);
      b.type = "button";
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", String(tab.id === this.activeId));
      b.classList.toggle("is-active", tab.id === this.activeId);
      b.addEventListener("click", () => this.select(tab.id));
      b.addEventListener("dblclick", () => { if (editing) this._rename(b, tab); });
      this.strip.append(b);
    }
    if (!editing) return;
    const active = this.activeId;
    const add = tool("+", "Add a tab");
    add.disabled = container.tabs.length >= MAX_TABS;
    add.addEventListener("click", () => this.addTab());
    const left = tool("‹", "Move this tab left");
    const right = tool("›", "Move this tab right");
    left.addEventListener("click", () => this._setTabs(moveTab(container.tabs, active, -1)));
    right.addEventListener("click", () => this._setTabs(moveTab(container.tabs, active, 1)));
    const drop = tool("✕", "Remove this tab and its cards");
    drop.disabled = container.tabs.length < 2;
    drop.addEventListener("click", () => {
      const tab = container.tabs.find((t) => t.id === active);
      if (tab.blocks.length && !window.confirm(`Remove the tab "${tab.title}" and the ${tab.blocks.length} card(s) on it?`)) return;
      const tabs = removeTab(container.tabs, active);
      this._setTabs(tabs, tabs[0].id);
    });
    this.strip.append(add, left, right, drop);
  }

  _rename(button, tab) {
    const input = h("input", "bd-rename bd-tab-rename");
    input.type = "text";
    input.maxLength = 120;
    input.value = tab.title;
    input.setAttribute("aria-label", "Tab title");
    button.replaceWith(input);
    input.focus();
    input.select();
    let done = false;
    const finish = (commit) => {
      if (done) return;
      done = true;
      const tabs = commit ? renameTab(this.container.tabs, tab.id, input.value) : this.container.tabs;
      if (tabs !== this.container.tabs) { this.container.tabs = tabs; this.onChange("title"); }
      this._renderStrip();
    };
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); finish(true); }
      else if (ev.key === "Escape") { ev.preventDefault(); finish(false); }
      ev.stopPropagation();
    });
    input.addEventListener("blur", () => finish(true));
  }
}
