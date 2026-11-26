/* Following this app's runs: the one the page is watching, the history and queue cards' data, and the polling.

   Run state is the executions API, polled; live numbers are the monitor streams (board/live.js), not this. */

import { api } from "../api.js";
import { errText } from "../lib/errors.js";

const ACTIVE = new Set(["planned", "queued", "running"]);
const enc = encodeURIComponent;

export class RunTracker {
  /** `hooks.displays()` redraws the live cards, `hooks.state()` the toolbar; `notify(message, kind)` tells the person. */
  constructor({ name, hooks, notify }) {
    this.name = name;
    this.hooks = hooks;
    this.notify = notify;
    this.execution = null; // {execution_id, status, started_at, finished_at, error, done, total}
    this.runs = null;      // {history: this app's runs, planned: every run waiting for memory}, newest first; null before the first answer
    this.pollTimer = 0;
    this.refreshTimer = 0; // while this app has a run waiting and none is followed: look again, the waiting run starts a moment after the last one ends
  }

  get active() { return !!this.execution && ACTIVE.has(this.execution.status); }

  /** Start a run; rejects with the server's refusal, which the page words. */
  async start(values, mode) {
    const exec = await api(`/apps/${enc(this.name)}/run`, { method: "POST", body: { values, mode } });
    const waiting = exec.status === "planned";
    this.notify(waiting ? `Run #${exec.execution_id} is waiting its turn.` : `Run #${exec.execution_id} started from "${this.name}".`, "success");
    if (!(waiting && this.active)) {
      this.execution = { ...exec, done: 0, total: 0 };
      this.schedulePoll(400);
    }
    this.hooks.displays();
    this.hooks.state();
    this.refresh();
    return exec;
  }

  async stop() {
    if (!this.execution) return;
    const id = this.execution.execution_id;
    await api(`/graphs/executions/${id}/stop`, { method: "POST" });
    this.notify(`Run #${id} stopped.`, "warn");
    this.schedulePoll(200);
    this.refresh();
  }

  /** A waiting run is cancelled the way a running one is stopped. */
  async cancel(id) {
    try {
      await api(`/graphs/executions/${id}/stop`, { method: "POST" });
      this.notify(`Run #${id} cancelled.`, "warn");
    } finally {
      await this.refresh();
    }
  }

  /** The history and queue cards' data, and: when the run this page followed has ended and the next one of this app has
      started, follow that. */
  async refresh() {
    try {
      const [mine, all] = await Promise.all([
        api(`/graphs/executions?limit=30&app=${enc(this.name)}`),
        api("/graphs/executions?limit=100"),
      ]);
      this.runs = {
        history: mine.executions || [],
        planned: (all.executions || []).filter((r) => r.status === "planned").sort((a, b) => a.execution_id - b.execution_id),
      };
      const next = this.runs.history.find((r) => r.status === "running" || r.status === "queued");
      if (next && !this.active) {
        this.execution = { ...next, done: 0, total: 0 };
        this.schedulePoll(0);
      } else if (!this.active && this.runs.history.some((r) => r.status === "planned")) {
        clearTimeout(this.refreshTimer);
        this.refreshTimer = setTimeout(() => this.refresh(), 1500);
      }
    } catch (err) {
      this.notify(`Could not read the run list: ${errText(err)}`, "warn");
    }
    this.hooks.displays();
  }

  /** On opening: follow whatever is running (from any page), or show how the latest run ended. */
  async adopt() {
    try {
      const list = await api("/graphs/executions?limit=1");
      const latest = (list.executions || [])[0];
      if (latest && ACTIVE.has(latest.status)) {
        this.execution = { ...latest, done: 0, total: 0 };
        this.schedulePoll(0);
      } else if (latest && (!this.execution || this.execution.execution_id === latest.execution_id)) {
        this.execution = { ...latest, done: 0, total: 0 };
        await this.poll(false);
      }
      this.hooks.displays();
      this.hooks.state();
    } catch (err) {
      this.notify(`Could not read run state: ${errText(err)}`, "warn");
    }
  }

  schedulePoll(delay) {
    clearTimeout(this.pollTimer);
    this.pollTimer = setTimeout(() => this.poll(true), delay);
  }

  async poll(again) {
    if (!this.execution) return;
    try {
      const d = await api(`/graphs/executions/${this.execution.execution_id}`);
      this.execution = {
        execution_id: d.execution_id, status: d.status, error: d.error,
        started_at: d.started_at, finished_at: d.finished_at,
        done: (d.results || []).length, total: ((d.graph && d.graph.nodes) || []).length,
      };
    } catch (err) {
      this.notify(`Could not read run state: ${errText(err)}`, "warn");
    }
    this.hooks.displays();
    this.hooks.state();
    this.refresh();
    if (again && this.active) this.schedulePoll(1500);
  }
}
