/* ---------------------------------------------------------------------------
   graph2/shapes.js -- the shape of every node on the canvas: the catalog's
   static ports, or the ports the server resolved for the node's current
   params (rework/contracts/resolve.md).

   `ShapeStore.shapeOf(node)` is what `GraphDoc` and the views read. A class
   with `dynamic: {depends_on, debounce_ms}` is re-resolved (debounced,
   sequence-tagged, superseded answers ignored) when one of those params
   changes. States per node: ready | needs_choice | error | resolving; on
   `error` the last good shape stays. `shape` events tell views what to patch.
   --------------------------------------------------------------------------- */

const DEFAULT_DEBOUNCE_MS = 300;

export class ShapeStore {
  /**
   * @param {{resolve: (className:string, body:object)=>Promise<object>, settings?: ()=>object,
   *          timers?: {set:Function, clear:Function}}} io  `resolve` posts to /graphs/nodes/{class}/resolve.
   */
  constructor({ resolve, settings = () => ({}), timers = { set: (fn, ms) => setTimeout(fn, ms), clear: (id) => clearTimeout(id) } }) {
    this._resolve = resolve;
    this._settings = settings;
    this._timers = timers;
    this.classes = new Map(); // class_name -> NodeInfo
    this.catalog = null;
    this._state = new Map(); // uid -> {state, shape, info, message, focus, choices, detail, seq, timer}
    this._handlers = new Set();
  }

  /** Subscribes to `{uid, state}` notifications (every state or shape change of a node). */
  onShape(handler) {
    this._handlers.add(handler);
    return () => this._handlers.delete(handler);
  }

  setCatalog(catalog) {
    this.catalog = catalog;
    this.classes = new Map(catalog.nodes.map((n) => [n.class_name, n]));
  }

  info(className) {
    return this.classes.get(className) || null;
  }

  static _static(info) {
    return { inputs: info.inputs, outputs: info.outputs, events_in: info.events_in || [], events_out: info.events_out || [] };
  }

  /** The node's ports: the last resolved shape, else the class's static ports; null for a class the catalog lacks. */
  shapeOf(node) {
    const entry = this._state.get(node.uid);
    if (entry && entry.shape) return entry.shape;
    const info = this.info(node.class_name);
    return info ? ShapeStore._static(info) : null;
  }

  /** `{state, info, message, focus, choices, detail}` of a node; `idle` for a node never resolved. */
  stateOf(uid) {
    const e = this._state.get(uid);
    return e ? { state: e.state, info: e.info, message: e.message, focus: e.focus, choices: e.choices, detail: e.detail } : { state: "idle" };
  }

  forget(uid) {
    const e = this._state.get(uid);
    if (e && e.timer) this._timers.clear(e.timer);
    this._state.delete(uid);
  }

  _notify(uid) {
    for (const h of [...this._handlers]) h({ uid, ...this.stateOf(uid) });
  }

  /** Whether a change of `param` on a node of this class needs a new resolve. */
  dependsOn(className, param) {
    const info = this.info(className);
    return Boolean(info && info.dynamic && info.dynamic.depends_on.includes(param));
  }

  /** Resolves now (a node just added, or a document just loaded). Static classes stay static. */
  resolveNow(node) {
    const info = this.info(node.class_name);
    if (!info || !info.dynamic) return Promise.resolve();
    return this._run(node);
  }

  /** Resolves after the class's debounce; later calls for the same node restart the wait. */
  schedule(node) {
    const info = this.info(node.class_name);
    if (!info || !info.dynamic) return;
    const entry = this._entry(node.uid);
    if (entry.timer) this._timers.clear(entry.timer);
    entry.state = "resolving";
    this._notify(node.uid);
    entry.timer = this._timers.set(() => {
      entry.timer = null;
      this._run(node);
    }, info.dynamic.debounce_ms ?? DEFAULT_DEBOUNCE_MS);
  }

  _entry(uid) {
    if (!this._state.has(uid)) this._state.set(uid, { state: "idle", shape: null, seq: 0, timer: null });
    return this._state.get(uid);
  }

  async _run(node) {
    const entry = this._entry(node.uid);
    const seq = ++entry.seq;
    entry.state = "resolving";
    this._notify(node.uid);
    let answer;
    try {
      answer = await this._resolve(node.class_name, { params: node.params, settings: this._settings() });
    } catch (err) {
      if (seq !== entry.seq) return;
      entry.state = "error";
      entry.message = err.message || String(err);
      this._notify(node.uid);
      return;
    }
    if (seq !== entry.seq) return; // a newer request is on its way
    entry.state = answer.state;
    entry.info = answer.info || "";
    entry.message = answer.message || null;
    entry.focus = answer.focus || null;
    entry.choices = answer.choices || {};
    entry.detail = answer.detail || {};
    if (answer.state !== "error") {
      entry.shape = { inputs: answer.inputs, outputs: answer.outputs, events_in: this.info(node.class_name).events_in || [], events_out: this.info(node.class_name).events_out || [] };
    }
    this._notify(node.uid);
  }
}
