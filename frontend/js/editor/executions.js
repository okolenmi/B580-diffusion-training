/* ---------------------------------------------------------------------------
   editor/executions.js -- run lifecycle panel (M7).

   Source of truth is the API (`/graphs/executions`); SSE /events only
   *notifies* us to refresh and streams per-node progress onto the canvas
   badges (docs 03 §2: SSE for live, polling as fallback while active).

   Flow: run(payload) -> 201 summary (422 graph_invalid surfaces issues,
   409 graph_execution_active is the single-active rule), the list
   refreshes, the new execution is selected and its detail fetch paints
   per-node results (ok/error + duration) onto the canvas. Stop acts on
   the selected active execution; wipe deletes history.
   --------------------------------------------------------------------------- */

import { api, ApiError } from "../api.js";

const ACTIVE = new Set(["queued", "running"]);
const TERMINAL = new Set(["finished", "error", "stopped"]);

export class Executions {
  constructor(root, canvas, { onNote, onRunState, onIssues }) {
    this.root = root;          // #exec-list
    this.canvas = canvas;
    this.onNote = onNote;
    this.onRunState = onRunState;  // (bool running) -> toggles Stop button
    this.onIssues = onIssues;      // (issues[]) -> renders the issues panel
    this.items = [];
    this.selected = null;
    this.liveId = null;        // execution this page started (badge stream)
    this._timer = null;
    this._refreshQueued = false;
  }

  /* ---- data ---- */

  async refresh() {
    try {
      const data = await api("/graphs/executions?limit=50");
      this.items = data.executions || [];
      this.render();
      this._schedulePoll();
    } catch (err) {
      this.onNote(_errText(err), "error");
    }
  }

  _schedulePoll() {
    clearTimeout(this._timer);
    if (this.items.some((e) => ACTIVE.has(e.status))) {
      this._timer = setTimeout(() => this.refresh(), 2500);
    }
    this.onRunState(this.items.some((e) => e.status === "running"));
  }

  _debouncedRefresh() {
    if (this._refreshQueued) return;
    this._refreshQueued = true;
    setTimeout(() => {
      this._refreshQueued = false;
      this.refresh();
    }, 200);
  }

  async select(id) {
    this.selected = id;
    this.render();
    try {
      const detail = await api(`/graphs/executions/${id}`);
      this.canvas.clearStatuses();
      for (const r of detail.results || []) {
        this.canvas.setNodeStatus(r.node_id, r.ok ? "ok" : "err", `${Math.round(r.duration_ms)}ms`);
      }
      if (detail.error) this.onNote(`Execution #${id}: ${detail.error}`, "error");
    } catch (err) {
      this.onNote(_errText(err), "error");
    }
  }

  /* ---- actions ---- */

  /** @returns the new execution, or null (issues/errors already handled). */
  async run(payload) {
    try {
      const exec = await api("/graphs/run", { method: "POST", body: payload });
      this.liveId = exec.execution_id;
      this.canvas.clearStatuses();
      await this.refresh();
      this.select(exec.execution_id);
      this.onNote(`Execution #${exec.execution_id} started.`, "success");
      return exec;
    } catch (err) {
      if (err instanceof ApiError && err.code === "graph_invalid") {
        // 422 details IS the issue list (docs 01 §error envelope,
        // docs 02 §8) -- not details.issues.
        const issues = Array.isArray(err.details)
          ? err.details
          : err.details && Array.isArray(err.details.issues)
            ? err.details.issues
            : null;
        if (issues) this.onIssues(issues);
        this.onNote(
          `Graph invalid: ${issues ? issues.length : "?"} issue(s).`,
          "error",
        );
      } else if (err instanceof ApiError && err.code === "graph_execution_active") {
        this.onNote("Another execution is active -- stop it first.", "warn");
        this.refresh();
      } else {
        this.onNote(_errText(err), "error");
      }
      return null;
    }
  }

  async stop() {
    // Prefer the selected execution when it is stoppable; otherwise fall
    // back to anything running, so the button always means what it says.
    let target = this.selected;
    if (target === null || !ACTIVE.has((this.items.find((e) => e.execution_id === target) || {}).status)) {
      const running = this.items.find((e) => ACTIVE.has(e.status));
      if (!running) {
        this.onNote("No active execution to stop.", "warn");
        return;
      }
      target = running.execution_id;
    }
    try {
      const exec = await api(`/graphs/executions/${target}/stop`, { method: "POST" });
      this.onNote(`Execution #${exec.execution_id} stopped.`, "warn");
      await this.refresh();
    } catch (err) {
      this.onNote(_errText(err), "error");
    }
  }

  async wipe() {
    if (!window.confirm("Delete all execution history?")) return;
    try {
      const res = await api("/graphs/executions", { method: "DELETE" });
      this.canvas.clearStatuses();
      this.selected = null;
      this.onNote(`Deleted ${res.deleted} execution(s).`, "warn");
      await this.refresh();
    } catch (err) {
      this.onNote(_errText(err), "error");
    }
  }

  /* ---- events ---- */

  /** Feed parsed /events messages here; ignores everything else. */
  onEvent(ev) {
    if (!ev || typeof ev.type !== "string" || !ev.type.startsWith("graph_execution")) return;
    if (ev.type === "graph_execution_progressed" && ev.execution_id === this.liveId) {
      this.canvas.setNodeStatus(ev.node_id, ev.ok ? "ok" : "err", `${Math.round(ev.duration_ms || 0)}ms`);
      return; // badge only; the list refreshes on the terminal event
    }
    if (TERMINAL.has(ev.type.replace("graph_execution_", "")) && ev.execution_id === this.liveId) {
      this.liveId = null;
    }
    if (ev.type === "graph_execution_failed" && ev.error) {
      this.onNote(`Execution #${ev.execution_id} failed: ${ev.error}`, "error");
    }
    if (ev.type === "graph_execution_finished") {
      this.onNote(`Execution #${ev.execution_id} finished (${ev.nodes} nodes).`, "success");
    }
    this._debouncedRefresh();
  }

  /* ---- render ---- */

  render() {
    this.root.replaceChildren();
    if (!this.items.length) {
      const line = document.createElement("div");
      line.className = "console-line info";
      line.textContent = "No executions yet.";
      this.root.appendChild(line);
      return;
    }
    for (const e of this.execRows()) this.root.appendChild(e);
  }

  execRows() {
    return this.items.map((e) => {
      const row = document.createElement("div");
      row.className = "exec-row" + (this.selected === e.execution_id ? " selected" : "");
      const left = document.createElement("span");
      left.textContent = `#${e.execution_id}`;
      const status = document.createElement("span");
      status.className = "exec-status st-" + e.status;
      status.textContent = e.status + (e.error ? ` -- ${e.error}` : "");
      const meta = document.createElement("span");
      meta.className = "exec-meta";
      meta.textContent = _ago(e.created_at);
      row.append(left, status, meta);
      row.addEventListener("click", () => this.select(e.execution_id));
      return row;
    });
  }
}

function _errText(err) {
  if (err instanceof ApiError) return `${err.code}: ${err.message}`;
  return String((err && err.message) || err);
}

function _ago(iso) {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}
