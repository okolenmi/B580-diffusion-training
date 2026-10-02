/* ---------------------------------------------------------------------------
   editor.js -- graph editor page entry (M7).

   Boots the catalog, GraphDoc and panels; owns the three cross-cutting
   concerns:
     - doc.onChange -> canvas re-render (always) + inspector refresh
      (whenever the change did NOT come from the inspector itself, so
      its text inputs keep focus while canvas widgets stay in sync);
     - one /events SSE subscription -> executions.onEvent (graph lifecycle)
       with the floating system console as the human-readable tail;
      - the collapsible layout (left/right rails + executions drawer),
        persisted in localStorage so the canvas keeps its gained space
        across visits.

   Everything network-shaped goes through api.js (single envelope decoder).
   --------------------------------------------------------------------------- */

import { api, ApiError } from "./api.js";
import { subscribeEvents } from "./lib/events.js";
import { GraphDoc } from "./editor/state.js";
import { Canvas } from "./editor/canvas.js";
import { Inspector } from "./editor/inspector.js";
import { Palette } from "./editor/palette.js";
import { Executions } from "./editor/executions.js";
import { Library } from "./editor/library.js";

const el = (id) => document.getElementById(id);

/* ---- system console: same capped pattern as views/dashboard.js ----
   There is no page-local log anymore: every note rides the floating
   console the shell mounts (#console-output). */

function log(message, kind = "info") {
  const out = el("console-output");
  const line = document.createElement("div");
  line.className = `console-line ${kind}`;
  line.textContent = message;
  out.appendChild(line);
  while (out.children.length > 60) out.removeChild(out.firstChild);
  out.scrollTop = out.scrollHeight;
}

function logError(err) {
  if (err instanceof ApiError) log(`${err.code}: ${err.message}`, "error");
  else log(String((err && err.message) || err), "error");
}

/* ---- layout: collapsible rails + executions drawer (persisted) ----
   Defaults: both rails visible, executions collapsed -- the canvas is
   the page, everything else is furniture it can do without. */

const LAYOUT_KEY = "ed.layout.v1";
const layout = Object.assign(
  { left: true, right: true, exec: false },
  (() => {
    try { return JSON.parse(localStorage.getItem(LAYOUT_KEY)) || {}; }
    catch (err) {
      // Corrupted storage -> defaults win. Benign, but quiet here reads
      // as "nothing was ever saved".
      console.warn("editor: stored graph unreadable, starting empty", err);
      return {};
    }
  })(),
);

function applyLayout() {
  document.body.classList.toggle("ed-left-off", !layout.left);
  document.body.classList.toggle("ed-right-off", !layout.right);
  el("exec-section").open = layout.exec;
  el("btn-toggle-left").setAttribute("aria-pressed", String(layout.left));
  el("btn-toggle-right").setAttribute("aria-pressed", String(layout.right));
}

function saveLayout() {
  try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout)); }
  catch { /* private mode / quota: layout is a convenience, not data */ }
}

function bindLayoutControls() {
  el("btn-toggle-left").addEventListener("click", () => {
    layout.left = !layout.left;
    applyLayout();
    saveLayout();
  });
  el("btn-toggle-right").addEventListener("click", () => {
    layout.right = !layout.right;
    applyLayout();
    saveLayout();
  });
  // the drawer opens/closes by its own <summary>; remember which
  el("exec-section").addEventListener("toggle", () => {
    layout.exec = el("exec-section").open;
    saveLayout();
  });
}

applyLayout(); // as early as a deferred module allows
bindLayoutControls();

/* ---- issues panel ---- */

function renderIssues(issues) {
  const root = el("issues");
  root.replaceChildren();
  if (!issues.length) {
    const line = document.createElement("div");
    line.className = "console-line success";
    line.textContent = "Graph is valid.";
    root.appendChild(line);
    return;
  }
  for (const issue of issues) {
    const row = document.createElement("div");
    row.className = `issue ${issue.severity === "error" ? "error" : "warn"}`;
    if (issue.node_id) {
      const chip = document.createElement("span");
      chip.className = "issue-node";
      chip.textContent = issue.node_id;
      chip.title = "Select this node";
      chip.addEventListener("click", () => canvas.focusNode(issue.node_id));
      row.appendChild(chip);
    }
    const text = document.createElement("span");
    text.textContent = issue.message + (issue.param ? ` [${issue.param}]` : "");
    row.appendChild(text);
    root.appendChild(row);
  }
}

/* ---- boot ---- */

const doc = new GraphDoc();
let canvas = null;
let inspector = null;

async function boot() {
  // catalog first: palette + doc's class lookup both need it
  let catalog = null;
  try {
    catalog = await api("/graphs/nodes");
  } catch (err) {
    logError(err);
    el("palette").replaceChildren();
    const line = document.createElement("div");
    line.className = "console-line error";
    line.textContent = "Catalog failed to load.";
    el("palette").appendChild(line);
  }
  if (catalog) {
    for (const classes of Object.values(catalog.domains || {})) {
      for (const cls of classes) doc.classByName[cls.class_name] = cls;
    }
  }

  inspector = new Inspector(el("inspector"), doc, { onNote: log });
  canvas = new Canvas(el("canvas-inner"), doc, {
    onSelect: (sel) => {
      if (!sel) {
        inspector.select(null);
        el("ed-selection").textContent = "";
        return;
      }
      if (sel.kind === "node") {
        inspector.select(sel.id);
        el("ed-selection").textContent = "";
      } else {
        const e = sel.edge;
        el("ed-selection").innerHTML = "";
        const code = document.createElement("code");
        code.textContent = `${e.from_node}.${e.from_port} → ${e.to_node}.${e.to_port}`;
        el("ed-selection").append("Wire ", code, " (Del removes, Esc deselects)");
        inspector.select(null);
      }
    },
    onNote: log,
  });

  doc.onChange = (reason, origin) => {
    canvas.render();
    // The inspector skips rebuilding only for ITS OWN commits (so
    // text/number inputs keep focus); canvas widgets, drags and
    // structural changes all refresh it.
    if (reason !== "params" || origin !== "inspector") inspector.refresh();
    el("btn-run").disabled = doc.size === 0;
  };
  el("btn-run").disabled = true;

  // palette: drop at the plane point under the viewport center
  const palette = new Palette(el("palette"), el("palette-search"), {
    onNote: log,
    onAdd: (className, x, y) => {
      const node = doc.addNode(className, x, y);
      canvas.selectNode(node.id);
      return node;
    },
  });
  let dropCount = 0;
  palette.dropPosition = () => {
    // cascade side by side (3 columns) so successive nodes never stack --
    // stacked nodes make wire starts ambiguous (topmost port wins the hit)
    const c = canvas.viewportCenter();
    const i = dropCount++;
    return {
      x: c.x - 115 + (i % 3) * 260,
      y: c.y - 60 + (Math.floor(i / 3) % 3) * 170,
    };
  };
  if (catalog) palette.load(catalog);

  // infinite plane: open on the origin circle at the viewport center
  canvas.centerOn(0, 0);

  // executions + library
  const executions = new Executions(el("exec-list"), canvas, {
    onNote: log,
    onIssues: renderIssues,
    onRunState: (running) => {
      el("btn-stop").hidden = !running;
    },
  });
  const library = new Library(
    {
      list: el("lib-list"),
      nameInput: el("lib-name"),
      saveBtn: el("lib-save"),
      importBtn: el("lib-import"),
    },
    doc,
    {
      onNote: log,
      onLoaded: () => {
        // bring the loaded graph into view (origin circle when empty)
        canvas.frameAll();
        const first = doc.nodes.keys().next();
        if (!first.done) canvas.selectNode(first.value);
        else inspector.select(null);
      },
    },
  );
  executions.refresh();
  library.refresh();

  // toolbar
  el("btn-validate").addEventListener("click", async () => {
    if (!doc.size) {
      log("Canvas is empty.", "warn");
      return;
    }
    try {
      const res = await api("/graphs/validate", { method: "POST", body: doc.toRunPayload() });
      renderIssues(res.issues || []);
      log(res.ok ? "Validation passed." : `Validation: ${res.issues.length} issue(s).`,
          res.ok ? "success" : "warn");
    } catch (err) {
      logError(err);
    }
  });

  el("btn-run").addEventListener("click", async () => {
    if (!doc.size) {
      log("Canvas is empty.", "warn");
      return;
    }
    await executions.run(doc.toRunPayload());
  });

  el("btn-stop").addEventListener("click", () => executions.stop());
  el("btn-wipe-exec").addEventListener("click", () => executions.wipe());
  el("btn-clear").addEventListener("click", () => {
    if (!doc.size) return;
    if (!window.confirm("Clear the canvas?")) return;
    doc.clear();
    inspector.select(null);
    log("Canvas cleared.", "warn");
  });

  // live events (SSE notifies, API stays the source of truth). /events has
  // no replay, so onOpen -- which fires on every (re)open -- is where the
  // authoritative state is refetched (docs 07 F-09).
  // subscribeEvents owns three things this used to re-implement: resync
  // on every open, counting unreadable frames out loud, and the transport
  // error notice. The behaviour is unchanged -- it was already correct
  // here -- but there is now one copy of it (docs 08 N-06).
  subscribeEvents({
    onEvent: (e) => executions.onEvent(e),
    onResync: (reason) => {
      log(reason);
      executions.resync();
    },
    onNotice: (m) => log(m, "error"),
    onError: (m) => log(m, "warn"),
  });

  log("Editor ready.");
}

boot();
