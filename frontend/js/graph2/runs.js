/* ---------------------------------------------------------------------------
   graph2/runs.js -- starting, stopping and listing runs of the open graph.

   The graph is sent as `{graph: <document without layout and views>}`
   (graph-json.md); the server owns everything else. A node's result shows on
   its card as `done 1.2s` / `error`, from the stored results of the run
   being looked at and from `graph_execution_progressed` while one is live.
   A refused start explains itself: the issue list for an invalid graph, the
   ledger's breakdown for a memory refusal.
   --------------------------------------------------------------------------- */

import { api, ApiError } from "../api.js";
import { errText } from "../lib/errors.js";

const ACTIVE = new Set(["pending", "running", "stopping"]);
const TERMINAL = new Set(["finished", "failed", "stopped"]);

const seconds = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`);

export class Runs {
  constructor(root, canvas, { onNote, onRunState, onIssues, onMemoryRefusal }) {
    this.root = root;
    this.canvas = canvas;
    this.onNote = onNote;
    this.onRunState = onRunState;
    this.onIssues = onIssues;
    this.onMemoryRefusal = onMemoryRefusal || (() => {});
    this.items = [];
    this.selected = null;
    this.liveId = null;
    this._timer = null;
    this._queued = false;
  }

  /** After the event stream (re)opens: nothing is replayed, so the list and the open run are read again. */
  async resync() {
    await this.refresh();
    if (this.selected !== null) await this.select(this.selected);
  }

  async refresh() {
    try {
      this.items = (await api("/graphs/executions?limit=50")).executions || [];
      this.render();
      clearTimeout(this._timer);
      if (this.items.some((e) => ACTIVE.has(e.status))) this._timer = setTimeout(() => this.refresh(), 2500);
      this.onRunState(this.items.some((e) => e.status === "running"));
    } catch (err) {
      this.onNote(errText(err), "error");
    }
  }

  async select(id) {
    this.selected = id;
    this.render();
    try {
      const detail = await api(`/graphs/executions/${id}`);
      this.canvas.clearStatuses();
      for (const r of detail.results || []) this.canvas.setNodeStatus(r.node_id, r.ok ? "done" : "error", seconds(r.duration_ms), r.error || "");
      if (detail.error) this.onNote(`Run #${id}: ${detail.error}`, "error");
    } catch (err) {
      this.onNote(errText(err), "error");
    }
  }

  async run(graph) {
    try {
      const exec = await api("/graphs/run", { method: "POST", body: { graph } });
      this.liveId = exec.execution_id;
      this.canvas.clearStatuses();
      await this.refresh();
      this.select(exec.execution_id);
      this.onNote(`Run #${exec.execution_id} started.`, "success");
      return exec;
    } catch (err) {
      if (err instanceof ApiError && err.code === "graph_invalid") {
        const issues = Array.isArray(err.details) ? err.details : err.details?.issues;
        if (issues) this.onIssues(issues);
        this.onNote(`Graph invalid: ${issues ? issues.length : "?"} issue(s).`, "error");
      } else if (err instanceof ApiError && err.code === "memory_unavailable") {
        this.onMemoryRefusal(err.details, errText(err));
        this.onNote(errText(err), "error");
      } else if (err instanceof ApiError && err.code === "graph_execution_active") {
        this.onNote("Another run is active -- stop it first.", "warn");
        this.refresh();
      } else {
        this.onNote(errText(err), "error");
      }
      return null;
    }
  }

  async stop() {
    let target = this.selected;
    if (target === null || !ACTIVE.has((this.items.find((e) => e.execution_id === target) || {}).status)) {
      const running = this.items.find((e) => ACTIVE.has(e.status));
      if (!running) return this.onNote("No active run to stop.", "warn");
      target = running.execution_id;
    }
    try {
      const exec = await api(`/graphs/executions/${target}/stop`, { method: "POST" });
      this.onNote(`Run #${exec.execution_id} stopped.`, "warn");
      await this.refresh();
    } catch (err) {
      this.onNote(errText(err), "error");
    }
  }

  async wipe() {
    if (!window.confirm("Delete all run history?")) return;
    try {
      const res = await api("/graphs/executions", { method: "DELETE" });
      this.canvas.clearStatuses();
      this.selected = null;
      this.onNote(`Deleted ${res.deleted} run(s).`, "warn");
      await this.refresh();
    } catch (err) {
      this.onNote(errText(err), "error");
    }
  }

  /** Feed parsed `/events` messages here; everything but graph runs is ignored. */
  onEvent(ev) {
    if (!ev || typeof ev.type !== "string" || !ev.type.startsWith("graph_execution")) return;
    if (ev.type === "graph_execution_progressed" && ev.execution_id === this.liveId) {
      this.canvas.setNodeStatus(ev.node_id, ev.ok ? "done" : "error", seconds(ev.duration_ms || 0));
      return;
    }
    if (TERMINAL.has(ev.type.replace("graph_execution_", "")) && ev.execution_id === this.liveId) this.liveId = null;
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
      const id = Object.assign(document.createElement("span"), { textContent: `#${e.execution_id}` });
      const status = Object.assign(document.createElement("span"), { className: `exec-status st-${e.status}`, textContent: e.status + (e.error ? ` -- ${e.error}` : "") });
      const meta = Object.assign(document.createElement("span"), { className: "exec-meta", textContent: ago(e.created_at) });
      row.append(id, status, meta);
      row.addEventListener("click", () => this.select(e.execution_id));
      this.root.append(row);
    }
  }
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
