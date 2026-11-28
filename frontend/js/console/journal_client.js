/* The page's side of the journal (docs/contracts/journal.md): the history on load, then what is new, kept in step with the server.

   Every page has the same console because the entries live on the server: a reload, another tab or a restart of the server loses nothing.
   `write` sends a line to the server and the line comes back through the next poll like everyone's; when the server cannot be reached the line is
   kept here, marked local, so a message about a failure is not lost to the failure. */

import { api } from "../api.js";

/** Fired on `window` when an error entry that is new to this page arrives: the docked console listens and opens itself, so a failure is seen. */
export const CONSOLE_ATTENTION = "console:attention";

const FIRST_PAGE = 300;
const KEEP = 1000;
const POLL_MS = 1500;
const HIDDEN_POLL_MS = 10000;

export class JournalClient {
  constructor({ call = api, shout = (name) => window.dispatchEvent(new Event(name)), hidden = () => document.hidden } = {}) {
    this.call = call;
    this.shout = shout;
    this.hidden = hidden;
    this.entries = [];
    this.cursor = 0;
    this.loaded = false;
    this.listeners = new Set();
    this.timer = 0;
    this.started = false;
    this.reachable = true;
    this.polling = null;
  }

  /** `listener(batch, {initial})` hears every batch of new entries, the history first. Returns the way to stop listening. */
  subscribe(listener) {
    this.listeners.add(listener);
    if (this.entries.length) listener(this.entries.slice(), { initial: true });
    return () => this.listeners.delete(listener);
  }

  start() {
    if (this.started) return;
    this.started = true;
    document.addEventListener("visibilitychange", () => { if (!this.hidden()) this.poll(); });
    this.poll({ initial: true });
  }

  async poll({ initial = false } = {}) {
    if (this.polling) return this.polling;
    this.polling = this.#poll(initial).finally(() => { this.polling = null; this.#again(); });
    return this.polling;
  }

  async #poll(initial) {
    try {
      const first = !this.loaded;
      const got = await this.call(`/journal?after=${this.cursor}&limit=${first ? FIRST_PAGE : 1000}`);
      this.loaded = true;
      if (!this.reachable) { this.reachable = true; this.#local("success", "The server answers again."); }
      this.#take(got.entries || [], initial || first);
    } catch (err) {
      if (this.reachable) { this.reachable = false; this.#local("warn", "The server does not answer; this page keeps trying."); }
    }
  }

  #again() {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.poll(), this.hidden() ? HIDDEN_POLL_MS : POLL_MS);
  }

  /** Send one line of this page to the journal. */
  async write(level, text) {
    try {
      await this.call("/journal", { method: "POST", body: { level, text } });
      this.poll();
    } catch (err) {
      this.#local(level, text);
    }
  }

  #local(level, text) {
    this.#take([{ seq: null, at: Date.now() / 1000, level, source: "page", text }], false);
  }

  #take(rows, initial) {
    const fresh = rows.filter((e) => e.seq === null || e.seq > this.cursor);
    if (!fresh.length) return;
    for (const e of fresh) if (e.seq !== null) this.cursor = Math.max(this.cursor, e.seq);
    this.entries.push(...fresh);
    if (this.entries.length > KEEP) this.entries.splice(0, this.entries.length - KEEP);
    for (const listener of this.listeners) listener(fresh, { initial });
    if (!initial && fresh.some((e) => e.level === "error")) this.shout(CONSOLE_ATTENTION);
  }
}

/** The one client of this page. */
export const journal = new JournalClient();
