/* ---------------------------------------------------------------------------
   graph2/app.js -- the page: builds the document, the shape store and the
   canvas, and connects them to the side panels and the server. Nothing here
   holds graph state of its own; each part listens to the document.
   --------------------------------------------------------------------------- */

import { api } from "../api.js";
import { el } from "../lib/dom.js";
import { errText } from "../lib/errors.js";
import { subscribeEvents } from "../lib/events.js";
import { log, logError } from "../lib/log.js";
import { Canvas } from "./canvas.js";
import { DraftSaver, readDraft } from "./draft.js";
import { FileIO } from "./file_io.js";
import { arrange, isUnplaced } from "./layout.js";
import { GraphDoc, endpointHost } from "./doc.js";
import { Inspector } from "./inspector.js";
import { Library } from "./library.js";
import { Monitor } from "./monitor.js";
import { Palette, QuickAdd, compatibleClasses } from "./palette.js";
import { IssuesPanel, MemoryPanel } from "./panels.js";
import { Runs } from "./runs.js";
import { SettingsPanel } from "./settings_panel.js";
import { ShapeStore } from "./shapes.js";
import { StarterDialog } from "./starters.js";

const LAYOUT_KEY = "g2.layout.v1";

function readLayout() {
  try {
    return { left: true, right: true, runs: false, ...JSON.parse(localStorage.getItem(LAYOUT_KEY)) };
  } catch {
    return { left: true, right: true, runs: false };
  }
}

function bindLayout() {
  const layout = readLayout();
  const apply = () => {
    document.body.classList.toggle("ed-left-off", !layout.left);
    document.body.classList.toggle("ed-right-off", !layout.right);
    el("btn-toggle-left").setAttribute("aria-pressed", String(layout.left));
    el("btn-toggle-right").setAttribute("aria-pressed", String(layout.right));
    el("exec-section").open = layout.runs;
  };
  const keep = () => {
    apply();
    try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout)); } catch { /* a convenience only */ }
  };
  el("btn-toggle-left").addEventListener("click", () => { layout.left = !layout.left; keep(); });
  el("btn-toggle-right").addEventListener("click", () => { layout.right = !layout.right; keep(); });
  el("exec-section").addEventListener("toggle", () => { layout.runs = el("exec-section").open; keep(); });
  apply();
}

/** What a picker offers: model files of the node's `scope`, or the image libraries the nodes can open. */
const PICKER_TTL_MS = 5000;
const pickerAnswers = new Map();

function pickerOptions(spec, values) {
  const route = spec.picker === "dataset" ? "/catalog/libraries" : spec.picker === "model_file" ? `/catalog/models/${encodeURIComponent(spec.ui?.picker_scope || spec.scope || values.scope || "checkpoints")}` : null;
  if (!route) return Promise.resolve([]);
  const kept = pickerAnswers.get(route);
  if (kept && Date.now() - kept.at < PICKER_TTL_MS) return kept.answer;
  const answer = api(route).then((reply) => (spec.picker === "dataset" ? reply.names || [] : reply.files || []));
  pickerAnswers.set(route, { at: Date.now(), answer });
  answer.catch(() => pickerAnswers.delete(route));
  return answer;
}

async function boot() {
  bindLayout();
  const doc = new GraphDoc({ shapeOf: (node) => shapes.shapeOf(node) });
  const shapes = new ShapeStore({
    resolve: (className, body) => api(`/graphs/nodes/${encodeURIComponent(className)}/resolve`, { method: "POST", body }),
    settings: () => doc.settings,
  });

  let catalog;
  try {
    catalog = await api("/catalog");
  } catch (err) {
    logError(err);
    el("palette").replaceChildren(Object.assign(document.createElement("div"), { className: "console-line error", textContent: `Catalog failed to load: ${errText(err)}` }));
    return;
  }
  shapes.setCatalog(catalog);

  const memory = new MemoryPanel(el("memory"), doc, { onNote: log });
  let canvas = null;
  let inspector = null;
  let quick = null;

  const addAt = (className, point) => {
    const { uid } = doc.addNode(className, { x: point.x, y: point.y });
    canvas.select([uid]);
    return uid;
  };
  let cascade = 0;
  const addAtCenter = (className) => {
    const c = canvas.viewCenter();
    const step = cascade++;
    return addAt(className, { x: c.x - 340 + (step % 3) * 320, y: c.y - 120 + (Math.floor(step / 3) % 3) * 200 });
  };

  canvas = new Canvas({
    root: el("g2-canvas"),
    doc,
    shapes,
    options: pickerOptions,
    note: log,
    hooks: {
      onSelect: (uids) => {
        inspector.show(uids.length === 1 ? uids[0] : null);
        el("ed-selection").textContent = uids.length > 1 ? `${uids.length} nodes selected` : "";
      },
      onQuickAdd: (point, event) => quick.open({ x: event.clientX, y: event.clientY }, null, "Add node"),
      onDropEmpty: ({ from, reverse, events, point, client }) => {
        if (events) return;
        const shape = shapes.shapeOf(doc.nodes.get(from.uid));
        const port = (reverse ? shape?.inputs : shape?.outputs)?.find((p) => p.name === from.port);
        if (!port) return;
        quick.open(client, compatibleClasses(catalog.nodes, port, !reverse), "Add and connect");
        quick.pending = { from, reverse, point };
      },
    },
  });
  inspector = new Inspector(el("inspector"), { doc, shapes, views: canvas.views });

  quick = new QuickAdd({
    shapes,
    onPick: (className, port, possible) => {
      const where = quick.pending?.point || canvas.viewCenter();
      const uid = addAt(className, where);
      if (possible && quick.pending) log(`${port.label || port.name} exists once the node is set up, so the node was added unconnected.`);
      else if (port && quick.pending) {
        const { from, reverse } = quick.pending;
        const [source, target] = reverse ? [{ uid, port: port.name }, from] : [from, { uid, port: port.name }];
        doc.connect(source.uid, source.port, target.uid, target.port);
      }
      quick.pending = null;
    },
  });

  new Palette(el("palette"), el("palette-search"), { shapes, onAdd: addAtCenter }).render();
  new SettingsPanel(el("graph-settings"), doc, catalog.settings || [], { note: log });

  const issues = new IssuesPanel(el("issues"), { onFocus: (uid) => canvas.focusNode(uid) });
  const monitor = new Monitor(el("monitor"), doc, shapes);
  el("monitor-section").addEventListener("toggle", () => monitor.shown());
  const runs = new Runs(el("exec-list"), canvas, {
    onNote: log,
    onIssues: (list) => issues.show(list),
    onMemoryRefusal: (details, reason) => memory.refusal(details, reason),
    onRunState: (running) => {
      el("btn-stop").hidden = !running;
      if (running) el("monitor-section").open = true;
      memory.refreshLater();
    },
  });
  const library = new Library(
    { list: el("lib-list"), nameInput: el("lib-name"), saveBtn: el("lib-save") },
    doc,
    { onNote: log, onLoaded: () => canvas.fit() },
  );

  const arrangeAll = () => {
    const wires = [...doc.edges, ...doc.eventEdges].map((e) => [endpointHost(e.from_node), endpointHost(e.to_node)]);
    const place = arrange([...doc.nodes.keys()], wires, (uid) => canvas.views.get(uid)?.el.offsetHeight || 240, (uid) => canvas.views.get(uid)?.el.offsetWidth || 280);
    for (const [uid, at] of place) doc.moveNode(uid, at.x, at.y);
    canvas.fit();
  };
  // A graph from the command line has no positions: place it when it arrives.
  doc.on("reset", () => { if (isUnplaced([...doc.nodes.values()])) arrangeAll(); });
  new FileIO(
    { importBtn: el("file-import"), exportBtn: el("file-export"), input: el("file-input"), nameInput: el("lib-name") },
    doc,
    { onNote: log, onLoaded: () => canvas.fit() },
  );
  const starters = new StarterDialog(doc, { options: pickerOptions, note: log, onBuilt: () => canvas.fit() });
  el("btn-starter").addEventListener("click", () => starters.open());
  el("btn-arrange").addEventListener("click", arrangeAll);

  // A class with dynamic ports is asked for its shape when it appears, when the document is replaced,
  // and (debounced) when one of the params it depends on changes.
  const resolveAll = () => { for (const node of doc.nodes.values()) shapes.resolveNow(node); };
  doc.on("node:add", (e) => shapes.resolveNow(doc.nodes.get(e.uid)));
  doc.on("node:remove", (e) => shapes.forget(e.uid));
  doc.on("node:params", (e) => {
    const node = doc.nodes.get(e.uid);
    if (node && shapes.dependsOn(node.class_name, e.name)) shapes.schedule(node);
  });
  doc.on("settings", () => resolveAll());
  doc.on("reset", resolveAll);
  doc.on("change", () => {
    el("btn-run").disabled = !doc.nodes.size;
    memory.refreshLater();
  });

  const draft = readDraft();
  if (draft) {
    try {
      doc.load(draft);
      canvas.fit();
      log(`Restored your last canvas (${doc.nodes.size} nodes) from this browser.`);
    } catch (err) {
      log(`The kept canvas could not be restored: ${errText(err)}`, "warn");
    }
  }
  const saver = new DraftSaver(doc);
  window.addEventListener("pagehide", () => saver.flush());
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") saver.flush(); });
  el("btn-run").disabled = !doc.nodes.size;

  el("btn-validate").addEventListener("click", async () => {
    if (!doc.nodes.size) return log("Canvas is empty.", "warn");
    try {
      const answer = await api("/graphs/validate", { method: "POST", body: { graph: doc.runPayload() } });
      issues.show(answer.issues || []);
      await memory.refresh();
      log(answer.ok ? "Validation passed." : `Validation: ${answer.issues.length} issue(s).`, answer.ok ? "success" : "warn");
    } catch (err) {
      logError(err);
    }
  });
  el("btn-run").addEventListener("click", () => {
    if (!doc.nodes.size) return log("Canvas is empty.", "warn");
    return runs.run(doc.runPayload());
  });
  el("btn-stop").addEventListener("click", () => runs.stop());
  el("btn-wipe-exec").addEventListener("click", () => runs.wipe());
  el("btn-fit").addEventListener("click", () => canvas.fit());
  el("btn-clear").addEventListener("click", () => {
    if (!doc.nodes.size || !window.confirm("Clear the canvas?")) return;
    doc.clear();
    log("Canvas cleared.", "warn");
  });

  subscribeEvents({
    onEvent: (event) => runs.onEvent(event),
    onResync: (reason) => { log(reason); runs.resync(); },
    onNotice: (message) => log(message, "error"),
    onError: (message) => log(message, "warn"),
  });

  runs.refresh();
  library.refresh();
  memory.refresh();
  log("Graph editor (new) ready.");
}

boot();
