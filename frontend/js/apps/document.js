/* The app being edited: the document, the draft graph its field cards write into, and what was last saved.

   The document is `{graph, root, style, source_graph, ...}` (docs/contracts/apps.md), saved with PUT. What a card can
   bind to is the backend's answer (`POST /boards/resolve`), never guessed here. */

import { api } from "../api.js";
import { autoPage, containerKey, leafCards, refineContainer, refKey } from "../board/model.js";

const enc = encodeURIComponent;

/** The graph as an app stores it: what a run reads, without the editor's layout and views. */
export function snapshotGraph(graph) {
  const copy = structuredClone(graph);
  delete copy.views;
  for (const node of copy.nodes || []) { delete node.layout; node.params = node.params || {}; }
  copy.settings = copy.settings || {};
  return copy;
}

/** The "draft graph" the cards edit: a stand-in whose parameter objects *are* the snapshot's, so a field card writes
    straight into what gets saved and run. */
function makeDraft(app) {
  const graph = app.graph;
  const memory = new Proxy({}, {
    get: (_t, key) => graph.settings[`memory.${String(key)}`],
    set: (_t, key, value) => { graph.settings[`memory.${String(key)}`] = value; return true; },
    deleteProperty: (_t, key) => { delete graph.settings[`memory.${String(key)}`]; return true; },
    has: (_t, key) => `memory.${String(key)}` in graph.settings,
  });
  return { get nodes() { return graph.nodes.map((n) => ({ id: n.uid, params: n.params })); }, memory };
}

export class AppDocument {
  constructor(name) {
    this.name = name;
    this.isNew = false;
    this.description = "";
    this.version = 0;
    this.app = null;     // {graph, root, style, source_graph, ...}
    this.draft = null;   // stand-in graph the field cards write into: {nodes, memory}
    this.saved = null;   // {layout, graph, exposed} as last saved; null = never
  }

  /** Load an existing app, or start a new one from a library graph (`?new=1&graph=...`). Returns the first resolution. */
  async load(query) {
    if (query.get("new")) return this.#startNew(query.get("graph") || "");
    const t = await api(`/apps/${enc(this.name)}`);
    this.description = t.description || "";
    this.version = t.version;
    this.app = t.doc;
    this.app.root = refineContainer(this.app.root); // a page saved on the coarse grid opens on the fine one, unchanged to the eye
    this.app.graph = snapshotGraph(this.app.graph);
    this.draft = makeDraft(this.app);
    const resolution = await this.resolve();
    this.markSaved();
    return resolution;
  }

  async #startNew(graphName) {
    const taken = (await api("/apps")).apps || [];
    if (taken.some((t) => t.name === this.name)) throw new Error(`an app called "${this.name}" already exists`);
    const lib = await api(`/graphs/library/${enc(graphName)}`);
    this.isNew = true;
    this.description = `Built on the graph "${graphName}".`;
    this.app = {
      format: 1, source_graph: graphName, style: { accent: "blue", density: "comfortable" },
      graph: snapshotGraph(lib.graph),
      root: { columns: 12, tabs: [{ id: "main", title: "Main", blocks: [] }] },
    };
    this.draft = makeDraft(this.app);
    this.app.root.tabs = autoPage((await this.resolve()).modules, this.app.root.columns);
    this.app.root = refineContainer(this.app.root);
    this.saved = null;
    return this.resolve();
  }

  /** Every card of the page, at any depth (the page edits `app.root` in place, so this is always current). */
  rootBlocks() { return this.app.root.tabs.flatMap((t) => t.blocks); }

  /** The document as it should be PUT. */
  payload() { return { ...this.app, format: 1 }; }

  resolve() { return api("/boards/resolve", { method: "POST", body: { doc: this.payload() } }); }

  #exposedKey() {
    return JSON.stringify([...leafCards(this.rootBlocks())].filter((b) => b.ref.kind === "field").map((b) => refKey(b.ref)).sort());
  }

  #graphKey() { return JSON.stringify(this.app.graph); }

  markSaved() {
    this.saved = { layout: containerKey(this.app.root), graph: this.#graphKey(), exposed: this.#exposedKey() };
  }

  layoutDirty() { return !this.saved || this.saved.layout !== containerKey(this.app.root); }
  valuesDirty() { return !this.saved || this.saved.graph !== this.#graphKey(); }
  isDirty() { return this.layoutDirty() || this.valuesDirty(); }
  /** The page shows a field the *saved* app does not expose: the one reason a run is refused client-side. */
  exposureChanged() { return !this.saved || this.saved.exposed !== this.#exposedKey(); }

  async save() {
    const out = await api(`/apps/${enc(this.name)}`, { method: "PUT", body: { description: this.description, doc: this.payload() } });
    this.isNew = false;
    this.version = out.version;
    this.markSaved();
    return out;
  }

  async remove() {
    if (!this.isNew) await api(`/apps/${enc(this.name)}`, { method: "DELETE" });
    this.markSaved(); // nothing left to lose: no unload prompt
  }

  /** The source graph's current state, and what taking it would change. */
  async compareWithSource() {
    const lib = await api(`/graphs/library/${enc(this.app.source_graph)}`);
    const next = snapshotGraph(lib.graph);
    const diff = await api(`/apps/${enc(this.name)}/diff`, { method: "POST", body: { graph: next } });
    return { next, diff };
  }

  adoptGraph(next) {
    this.app.graph = next;
    this.draft = makeDraft(this.app);
  }
}
