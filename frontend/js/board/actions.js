/* ---------------------------------------------------------------------------
   board/actions.js -- the control inside an *action* card: a button that asks a
   node of the running graph to do something ("Save now").

   The page sends only which node and which action (`POST /graphs/executions/<id>/actions`);
   what happens is the node's own declaration. The server answers "queued"; the node's answers
   -- queued, delivered, rejected, failed -- arrive as `{kind: "action"}` frames on its stream,
   which `StreamState.actions` keeps, so the line under the button is what the node said, and
   it is still there after a reload (the stream replays).
   --------------------------------------------------------------------------- */

const h = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

async function postAction(executionId, node, action) {
  const res = await fetch(`/api/v1/graphs/executions/${executionId}/actions`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ node, action }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((body.error && body.error.message) || `HTTP ${res.status}`);
  return body;
}

const SAYS = {
  queued: "Sent…",
  delivered: "Done",
  rejected: "Not accepted",
  failed: "Failed",
};

export class ActionCard {
  /**
   * @param {object} p
   * @param {object} p.descriptor  the action as the modules list it: name, label, doc, confirm, available, reason
   * @param {object} p.node        the node that owns it ({id, display_name})
   * @param {object} p.state       the node's StreamState (its `actions` map)
   * @param {object} p.ctx         `{execution: () => {execution_id, status}|null}`
   * @param {(msg: string, kind?: string) => void} [p.onNote]
   * @param {Function} [p.post]    `(executionId, node, action) => Promise`; replaceable for tests
   * @param {(text: string) => boolean} [p.confirm]
   */
  constructor({ descriptor, node, state, ctx, onNote = () => {}, post = postAction, confirm = (t) => window.confirm(t) }) {
    this.descriptor = descriptor;
    this.node = node;
    this.state = state;
    this.ctx = ctx || {};
    this.onNote = onNote;
    this.post = post;
    this.confirm = confirm;
    this.commandId = null;
    this.sending = false;
    this.problem = "";

    this.el = h("div", "bd-action");
    this.button = h("button", "btn btn-secondary", descriptor.label);
    this.button.type = "button";
    this.button.addEventListener("click", () => this.press());
    this.note = h("div", "bd-action-note");
    this.note.setAttribute("aria-live", "polite");
    this.el.append(this.button, this.note);
    this.update();
  }

  get _running() {
    const run = this.ctx.execution ? this.ctx.execution() : null;
    return run && run.status === "running" ? run : null;
  }

  /** What the node last said about this action (the command this card sent, else the latest for the action). */
  _answer() {
    const map = this.state && this.state.actions;
    if (!map || !map.size) return null;
    if (this.commandId && map.has(this.commandId)) return map.get(this.commandId);
    let last = null;
    for (const a of map.values()) if (a.action === this.descriptor.name) last = a;
    return last;
  }

  async press() {
    const run = this._running;
    if (!run || this.sending || !this.descriptor.available) return;
    if (this.descriptor.confirm && !this.confirm(this.descriptor.confirm)) return;
    this.sending = true;
    this.problem = "";
    this.update();
    try {
      const queued = await this.post(run.execution_id, this.node.id, this.descriptor.name);
      this.commandId = queued.command_id;
    } catch (err) {
      this.problem = err.message || String(err);
      this.onNote(`${this.descriptor.label}: ${this.problem}`, "warn");
    } finally {
      this.sending = false;
      this.update();
    }
  }

  /** Nothing to release (the board owns the stream); a board drops every card it built when its tab is left. */
  destroy() {}

  update() {
    const { descriptor } = this;
    const running = this._running;
    this.button.disabled = !running || this.sending || !descriptor.available;
    const answer = this._answer();
    let text = "";
    if (!descriptor.available) text = descriptor.reason || "Nothing in this graph can act on it.";
    else if (this.problem) text = this.problem;
    else if (this.sending) text = SAYS.queued;
    else if (answer) text = `${SAYS[answer.status] || answer.status}${answer.detail ? `: ${answer.detail}` : ""}`;
    else if (!running) text = "Available while a run is going.";
    this.note.textContent = text;
    this.note.dataset.status = this.problem ? "failed" : answer ? answer.status : "";
    this.button.title = descriptor.doc || "";
  }
}
