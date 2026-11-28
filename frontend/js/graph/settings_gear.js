/* ---------------------------------------------------------------------------
   graph/settings_gear.js -- the graph's settings behind a gear in the corner of the canvas.

   The gear is always in sight; it opens an overlay on the canvas with the settings of this graph (they apply to every
   run of it), and a dot on the gear says how many differ from their defaults. Esc, a click outside or the close
   button put it away. A frozen run's graph shows the same overlay with the controls dead (CSS, `body.is-instance`).
   --------------------------------------------------------------------------- */

import { SettingsForm, isChanged } from "./settings_form.js";

const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const GEAR = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>';

export class SettingsGear {
  /** @param {HTMLElement} root the layer over the canvas @param {object} doc @param {object[]} specs @param {{note: Function}} deps */
  constructor(root, doc, specs, { note }) {
    this.root = root;
    this.doc = doc;
    this.specs = specs;

    this.button = make("button", "gs-gear");
    this.button.type = "button";
    this.button.title = "Graph settings";
    this.button.setAttribute("aria-haspopup", "dialog");
    this.button.setAttribute("aria-expanded", "false");
    this.button.innerHTML = GEAR;
    this.dot = make("span", "gs-gear-dot");
    this.dot.hidden = true;
    this.button.append(this.dot);

    this.overlay = make("div", "gs-overlay");
    this.overlay.hidden = true;
    this.overlay.setAttribute("role", "dialog");
    this.overlay.setAttribute("aria-label", "Graph settings");
    const head = make("div", "gs-overlay-head");
    const titles = make("div", "");
    titles.append(make("div", "gs-overlay-title", "Graph settings"), make("div", "gs-overlay-sub", "Apply to every run of this graph."));
    const close = make("button", "gs-close", "×");
    close.type = "button";
    close.title = "Close (Esc)";
    close.addEventListener("click", () => this.close());
    head.append(titles, close);
    const body = make("div", "gs-overlay-body gs-list");
    this.form = new SettingsForm(body, doc, specs, { note });
    const foot = make("div", "gs-overlay-foot");
    this.advanced = make("button", "btn btn-secondary btn-small", "Show advanced");
    this.advanced.type = "button";
    this.advanced.addEventListener("click", () => {
      this.form.setAdvanced(!this.form.showAdvanced);
      this.advanced.textContent = this.form.showAdvanced ? "Hide advanced" : "Show advanced";
    });
    this.resetAll = make("button", "btn btn-secondary btn-small", "Reset all");
    this.resetAll.type = "button";
    this.resetAll.addEventListener("click", () => this.form.resetAll());
    foot.append(this.advanced, this.resetAll);
    if (!specs.some((spec) => spec.advanced)) this.advanced.hidden = true;
    this.overlay.append(head, body, foot);
    root.append(this.button, this.overlay);

    this.button.addEventListener("click", () => (this.overlay.hidden ? this.open() : this.close()));
    this._onKey = (e) => { if (e.key === "Escape") { this.close(); this.button.focus(); } };
    this._onPointer = (e) => { if (!this.root.contains(e.target)) this.close(); };
    doc.on("settings", () => this._mark());
    doc.on("reset", () => this._mark());
    this._mark();
  }

  open() {
    this.overlay.hidden = false;
    this.button.setAttribute("aria-expanded", "true");
    document.addEventListener("keydown", this._onKey);
    document.addEventListener("pointerdown", this._onPointer, true);
  }

  close() {
    this.overlay.hidden = true;
    this.button.setAttribute("aria-expanded", "false");
    document.removeEventListener("keydown", this._onKey);
    document.removeEventListener("pointerdown", this._onPointer, true);
  }

  /** The dot on the gear: how many settings differ from their defaults. */
  _mark() {
    const count = this.specs.filter((spec) => isChanged(this.doc, spec)).length;
    this.dot.hidden = count === 0;
    this.dot.textContent = String(count);
    this.resetAll.disabled = count === 0;
    this.button.title = count ? `Graph settings (${count} changed)` : "Graph settings";
  }
}
