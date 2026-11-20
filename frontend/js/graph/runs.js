/* ---------------------------------------------------------------------------
   graph/runs.js -- starting, stopping and listing runs of the open graph.

   The graph is sent as `{graph: <document without layout and views>}`
   (graph-json.md); the server owns everything else. A node's result shows on
   its card as `done 1.2s` / `error`, from the stored results of the run
   being looked at and from `graph_execution_progressed` while one is live.
   The stored results are also re-read on every poll and stream (re)open:
   a progress event is a delta, and a delta that was lost is not replayed.
   A refused start explains itself: the issue list for an invalid graph, the
   ledger's breakdown for a memory refusal.
   --------------------------------------------------------------------------- */

import { api, ApiError } from "../api.js";
import { errText } from "../lib/errors.js";

const ACTIVE = new Set(["planned", "queued", "running"]);
const POLL_MS = 2500;

const seconds = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`);

/** The run whose results belong on the cards: the one looked at, else (while following) the newest one. */
export function statusRun(items, selected, follow = true) {
  if (selected !== null) return selected;
  return follow ? items[0]?.execution_id ?? null : null;
}

export class Runs {
  constructor(root, canvas, { onNote, onRunState, onIssues, onMemoryRefusal, onSelect, onOpen, request = api }) {
    this.root = root;
    this.canvas = canvas;
    this._request = request;
    this.onNote = onNote;
    this.onRunState = onRunState;
    this.onIssues = onIssues;
    this.onMemoryRefusal = onMemoryRefusal || (() => {});
    this.onSelect = onSelect || (() => {});
    this.onOpen = onOpen || (() => {});
    this.items = [];
    this.selected = null;
    this._statusRun = null; // the run whose results are drawn; null when the cards show none
    this._follow = true;    // show the newest run until the person explicitly leaves the runs
    this._statusActive = false; // the status run was still going at the last detail read
    this._statusFailed = false; // the last detail read failed; keep polling until one succeeds
    this._refreshSeq = 0;
    this._timer = null;
    this._queued = false;
  }

  /** After the event stream (re)opens: the list and the run whose results are drawn are read again. */
  async resync() {
    await this.refresh();
    if (this.selected !== null) await this.select(this.selected, { open: false });
  }

  async refresh() {
    clearTimeout(this._timer);
    this._timer = null;
    const seq = ++this._refreshSeq;
    let failed = false;
    try {
      const items = (await this._request("/graphs/executions?limit=50")).executions || [];
      if (seq !== this._refreshSeq) return; // a newer refresh owns the answer
      this.items = items;
      this.render();
    } catch (err) {
      if (seq !== this._refreshSeq) return;
      failed = true;
      this.onNote(errText(err), "error");
    }
    const wanted = statusRun(this.items, this.selected, this._follow);
    if (wanted !== this._statusRun) {
      this.canvas.clearStatuses();
      this._statusRun = wanted;
      this._statusActive = false;
      this._statusFailed = false;
    }
    // A progress event is a delta and may be lost (a reconnect, a full client
    // queue). The run row holds every result, so reading it while the run is
    // open is what makes a missed node recover without a reload.
    if (this._statusRun !== null) await this._syncStatuses(this._statusRun);
    if (seq !== this._refreshSeq) return;
    const active = this.items.some((e) => ACTIVE.has(e.status)) || this._statusActive;
    if (active || failed || this._statusFailed) this._timer = setTimeout(() => this.refresh(), POLL_MS);
    if (!failed) this.onRunState(active);
  }

  /** Puts a run's stored results on the cards; never clears the live badges that are already there. */
  async _syncStatuses(id) {
    let detail;
    try {
      detail = await this._request(`/graphs/executions/${id}`);
    } catch (err) {
      if (this._statusRun !== id) return null;
      // One line per failure, not one per poll; the poll keeps retrying.
      if (!this._statusFailed) this.onNote(errText(err), "error");
      this._statusFailed = true;
      return null;
    }
    if (this._statusRun !== id) return null;
    this._statusFailed = false;
    this._statusActive = ACTIVE.has(detail.status);
    this._applyResults(detail);
    return detail;
  }

  _applyResults(detail) {
    for (const r of detail.results || []) this.canvas.setNodeStatus(r.node_id, r.ok ? "done" : "error", seconds(r.duration_ms ?? 0), r.error || "");
  }

  /** The run being looked at, as the status display reads it; null before the first. */
  current() {
    return this.items.find((e) => e.execution_id === this.selected) || this.items[0] || null;
  }

  /** Look at a run: its curves and statuses follow. `open` also puts the run's own frozen graph on the canvas (an
   *  instance); a run just started from the canvas is followed without that, the canvas already is its graph. */
  async select(id, { open = false } = {}) {
    // Clear before the fetch, not after it: progress events for the new run can
    // arrive while the detail is on its way, and clearing afterwards wipes them.
    if (this._statusRun !== id) this.canvas.clearStatuses();
    this._statusRun = id;
    this._statusActive = false;
    this._statusFailed = false;
    this.selected = id;
    this._follow = true;
    this.render();
    try {
      const detail = await this._request(`/graphs/executions/${id}`);
      if (this._statusRun !== id) return; // another run was picked meanwhile
      if (open) this.onOpen(detail);
      this._statusActive = ACTIVE.has(detail.status);
      this._applyResults(detail);
      if (detail.error) this.onNote(`Run #${id}: ${detail.error}`, "error");
      this.onSelect(id, this._statusActive);
    } catch (err) {
      this.onNote(errText(err), "error");
      if (this._statusRun === id) {
        this._statusFailed = true; // the poll retries the read
        if (!this._timer) this._timer = setTimeout(() => this.refresh(), POLL_MS);
      }
    }
  }

  /** `mode`: what to do when the card has no room -- "auto" plans the run, "parallel" refuses, "queue" waits its turn. */
  async run(graph, mode = "auto", name = null) {
    try {
      const exec = await this._request("/graphs/run", { method: "POST", body: { graph, mode, name } });
      // The new run's cards start clean; `refresh` then reads its stored
      // results and keeps the poll going while it is active.
      this.selected = exec.execution_id;
      this._statusRun = exec.execution_id;
      this._statusActive = ACTIVE.has(exec.status);
      this._statusFailed = false;
      this._follow = true;
      this.canvas.clearStatuses();
      this.render();
      await this.refresh();
      this.onSelect(exec.execution_id, this._statusActive);
      this.onNote(exec.status === "planned" ? `Run #${exec.execution_id} planned: it starts when there is room.` : `Run #${exec.execution_id} started.`, "success");
      return exec;
    } catch (err) {
      if (err instanceof ApiError && err.code === "graph_invalid") {
        const issues = Array.isArray(err.details) ? err.details : err.details?.issues;
        if (issues) this.onIssues(issues);
        this.onNote(`Graph invalid: ${issues ? issues.length : "?"} issue(s).`, "error");
      } else if (err instanceof ApiError && err.code === "memory_unavailable") {
        this.onMemoryRefusal(err.details, errText(err));
        this.onNote(errText(err), "error");
      } else {
        this.onNote(errText(err), "error");
      }
      return null;
    }
  }

  /** Ask for a new name for a run (empty clears it). */
  async rename(item) {
    const answer = window.prompt(`Name for run #${item.execution_id} (empty clears it)`, item.name || "");
    if (answer === null) return;
    try {
      await this._request(`/graphs/executions/${item.execution_id}`, { method: "PATCH", body: { name: answer } });
      await this.refresh();
    } catch (err) {
      this.onNote(errText(err), "error");
    }
  }

  /** Stop looking at any run: the canvas goes back to the live streams and the cards show no results. */
  deselect() {
    this.selected = null;
    this._follow = false;
    this._statusRun = null;
    this._statusActive = false;
    this._statusFailed = false;
    this.render();
    this.canvas.clearStatuses();
    this.onSelect(null, false);
  }

  /** Stop (or cancel, when it is still planned) one run: the given one, else what `stopTarget` picks. */
  async stop(id = null) {
    const target = id ?? stopTarget(this.items, this.selected);
    if (target === null) return this.onNote("No active run to stop.", "warn");
    try {
      const exec = await this._request(`/graphs/executions/${target}/stop`, { method: "POST" });
      this.onNote(`Run #${exec.execution_id} stopped.`, "warn");
      await this.refresh();
    } catch (err) {
      this.onNote(errText(err), "error");
    }
  }

  async wipe() {
    if (!window.confirm("Delete all run history?")) return;
    try {
      const res = await this._request("/graphs/executions", { method: "DELETE" });
      this.canvas.clearStatuses();
      this.selected = null;
      this._statusRun = null;
      this._statusActive = false;
      this._statusFailed = false;
      this._follow = true;
      this.onNote(`Deleted ${res.deleted} run(s).`, "warn");
      await this.refresh();
    } catch (err) {
      this.onNote(errText(err), "error");
    }
  }

  /** Feed parsed `/events` messages here; everything but stream (re)opens and graph runs is ignored. */
  onEvent(ev) {
    if (!ev || typeof ev.type !== "string") return;
    if (ev.type === "stream_opened") {
      // Progress is a delta and is not replayed. The server says "no resync"
      // when lifecycle events covered the gap, but node results were still
      // lost, so the run on the cards is read again either way.
      if (ev.resync_required === false) this.resync();
      return;
    }
    if (!ev.type.startsWith("graph_execution")) return;
    if (ev.type === "graph_execution_progressed") {
      if (this.selected === null && this._follow && ev.execution_id !== this._statusRun) {
        // Following the newest run without having picked one: adopt it now
        // rather than waiting for the next poll.
        this.canvas.clearStatuses();
        this._statusRun = ev.execution_id;
        this._statusActive = true;
        this._statusFailed = false;
      }
      if (ev.execution_id === this._statusRun) {
        this.canvas.setNodeStatus(ev.node_id, ev.ok ? "done" : "error", seconds(ev.duration_ms || 0), ev.error || "");
      }
      return;
    }
    if (ev.type === "graph_execution_failed" && ev.error) this.onNote(`Run #${ev.execution_id} failed: ${ev.error}`, "error");
    if (ev.type === "graph_execution_finished") this.onNote(`Run #${ev.execution_id} finished (${ev.nodes} nodes).`, "success");
    if (this._queued) return;
    this._queued = true;
    setTimeout(() => { this._queued = false; this.refresh(); }, 200);
  }

  render() {
    this.root.replaceChildren();
    if (!this.items.length) {
      const line = document.createElement("div");
      line.className = "console-line info";
      line.textContent = "No runs yet.";
      this.root.append(line);
      return;
    }
    for (const e of this.items) {
      const row = document.createElement("div");
      row.className = `exec-row${this.selected === e.execution_id ? " selected" : ""}`;
      const id = Object.assign(document.createElement("span"), { className: "exec-id", textContent: `#${e.execution_id}${e.name ? " " + e.name : ""}`, title: "Double-click to rename" });
      id.addEventListener("dblclick", (ev) => { ev.stopPropagation(); this.rename(e); });
      const status = Object.assign(document.createElement("span"), { className: `exec-status st-${e.status}`, textContent: e.status + (e.error ? ` -- ${e.error}` : "") });
      const meta = Object.assign(document.createElement("span"), { className: "exec-meta", textContent: ago(e.created_at) });
      row.append(id, status, meta);
      if (ACTIVE.has(e.status)) {
        const label = e.status === "planned" ? "Cancel this planned run" : "Stop this run";
        const stop = Object.assign(document.createElement("button"), { className: "btn btn-danger btn-small exec-stop", type: "button", textContent: "\u25A0", title: label });
        stop.addEventListener("click", (ev) => { ev.stopPropagation(); this.stop(e.execution_id); });
        row.append(stop);
      }
      row.addEventListener("click", () => this.select(e.execution_id, { open: true }));
      this.root.append(row);
    }
  }
}

/**
 * The run the Stop button acts on: the one looked at when it is running, else a running one (the newest), else the one
 * looked at when it is only planned, else the newest planned. A planned row is usually the one just started and looked
 * at, so "the selected one" alone would cancel it while the run that holds the card goes on.
 */
export function stopTarget(items, selected) {
  const status = (n) => (items.find((e) => e.execution_id === n) || {}).status;
  if (selected !== null && status(selected) === "running") return selected;
  const running = items.find((e) => e.status === "running");
  if (running) return running.execution_id;
  if (selected !== null && ACTIVE.has(status(selected))) return selected;
  return items.find((e) => ACTIVE.has(e.status))?.execution_id ?? null;
}

export function ago(iso, now = Date.now()) {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const s = Math.max(0, (now - t) / 1000);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}
