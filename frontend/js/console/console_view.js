/* The lines of the system console: draws the journal's entries (time, who said it, what), keeps the view at the newest line unless the person
   scrolled up to read, and filters. Clearing hides what is shown now; the history stays on the server. The window around it (dragging, docking) is shell.js. */

import { DEFAULT_FILTER, FILTERS, asText, problemCount, stamp, visible } from "./journal_model.js";

const MAX_DRAWN = 500;
const STORE = "shell.console.view.v1";

function remembered() {
  try { return JSON.parse(localStorage.getItem(STORE) || "{}"); } catch { return {}; }
}

export class ConsoleView {
  /** `parts`: {lines, filter (a select), clear, copy} elements; `onUnread(n)` hears how many warnings and errors came in while `isOpen()` was false. */
  constructor(parts, { isOpen, onUnread = () => {} }) {
    this.parts = parts;
    this.isOpen = isOpen;
    this.onUnread = onUnread;
    this.entries = [];
    const saved = remembered();
    this.filter = FILTERS[saved.filter] ? saved.filter : DEFAULT_FILTER;
    this.clearedTo = Number.isInteger(saved.clearedTo) ? saved.clearedTo : 0;
    this.unread = 0;
    for (const [name, { label }] of Object.entries(FILTERS)) parts.filter.append(new Option(label, name));
    parts.filter.value = this.filter;
    parts.filter.onchange = () => { this.filter = parts.filter.value; this.#keep(); this.draw(); };
    parts.clear.onclick = () => { this.#clear(); };
    parts.copy.onclick = () => this.#copy();
  }

  /** The window was opened: what came in meanwhile is now seen. */
  seen() {
    this.unread = 0;
    this.onUnread(0);
    this.parts.lines.scrollTop = this.parts.lines.scrollHeight;
  }

  /** A batch from the journal client. */
  add(batch, { initial = false } = {}) {
    this.entries.push(...batch);
    if (this.entries.length > 2 * MAX_DRAWN) this.entries.splice(0, this.entries.length - 2 * MAX_DRAWN);
    if (initial) { this.draw(); return; }
    const shown = visible(batch, this.filter, this.clearedTo);
    const lines = this.parts.lines;
    const stuck = lines.scrollHeight - lines.scrollTop - lines.clientHeight < 28;
    for (const entry of shown) lines.append(this.#line(entry));
    while (lines.children.length > MAX_DRAWN) lines.firstElementChild.remove();
    if (stuck) lines.scrollTop = lines.scrollHeight;
    if (!this.isOpen()) { this.unread += problemCount(shown); this.onUnread(this.unread); }
  }

  draw() {
    const lines = this.parts.lines;
    const shown = visible(this.entries, this.filter, this.clearedTo).slice(-MAX_DRAWN);
    lines.replaceChildren(...(shown.length ? shown.map((e) => this.#line(e)) : [this.#empty()]));
    lines.scrollTop = lines.scrollHeight;
  }

  #empty() {
    const line = document.createElement("div");
    line.className = "console-line info";
    line.textContent = this.clearedTo || this.filter !== DEFAULT_FILTER ? "Nothing to show here (cleared or filtered)." : "Nothing has happened yet.";
    return line;
  }

  #line(entry) {
    const line = document.createElement("div");
    line.className = `console-line ${entry.level}`;
    const when = document.createElement("span");
    when.className = "cl-time";
    when.textContent = stamp(entry.at);
    const who = document.createElement("span");
    who.className = "cl-src";
    who.textContent = entry.source;
    const text = document.createElement("span");
    text.className = "cl-text";
    text.textContent = entry.text; // textContent, never markup: entries carry file paths, prompts and error text
    line.append(when, who, text);
    return line;
  }

  #clear() {
    const numbered = this.entries.filter((e) => e.seq !== null).map((e) => e.seq);
    this.clearedTo = numbered.length ? Math.max(...numbered) : this.clearedTo;
    this.entries = this.entries.filter((e) => e.seq === null || e.seq > this.clearedTo);
    this.#keep();
    this.draw();
  }

  #copy() {
    const text = visible(this.entries, this.filter, this.clearedTo).map((e) => asText(e)).join("\n");
    navigator.clipboard?.writeText(text).catch(() => {});
    this.parts.copy.textContent = "Copied";
    setTimeout(() => { this.parts.copy.textContent = "Copy"; }, 1200);
  }

  #keep() {
    try { localStorage.setItem(STORE, JSON.stringify({ filter: this.filter, clearedTo: this.clearedTo })); } catch { /* a convenience only */ }
  }
}
