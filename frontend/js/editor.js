/* ---------------------------------------------------------------------------
   editor.js -- graph editor page entry (M7).

   Boots the catalog, GraphDoc and panels; owns the two cross-cutting
   concerns:
     - doc.onChange -> canvas re-render (always) + inspector refresh
       (structural changes only; "params" commits come from the inspector
       itself so text inputs keep focus);
     - one /events SSE subscription -> executions.onEvent (graph lifecycle)
       with the console as the human-readable tail.

   Everything network-shaped goes through api.js (single envelope decoder).
   --------------------------------------------------------------------------- */

import { api, sse, ApiError } from "./api.js";
import { GraphDoc } from "./editor/state.js";
import { Canvas } from "./editor/canvas.js";
import { Inspector } from "./editor/inspector.js";
import { Palette } from "./editor/palette.js";
import { Executions } from "./editor/executions.js";
import { Library } from "./editor/library.js";

const el = (id) => document.getElementById(id);

/* ---- page console (same capped pattern as views/dashboard.js) ---- */

function log(message, kind = "info") {
  const out = el("ed-log");
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

  doc.onChange = (reason) => {
    canvas.render();
    if (reason !== "params") inspector.refresh();
    el("btn-run").disabled = doc.size === 0;
  };
  el("btn-run").disabled = true;

  // palette: drop at the canvas viewport center, jittered per drop
  const palette = new Palette(el("palette"), el("palette-search"), {
    onNote: log,
    onAdd: (className, x, y) => {
      const node = doc.addNode(className, x, y);
      canvas.selectNode(node.id);
      return node;
    },
  });
  const scroller = el("graph-canvas");
  let dropCount = 0;
  palette.dropPosition = () => {
    const jitter = (dropCount++ % 5) * 30;
    return {
      x: scroller.scrollLeft + scroller.clientWidth / 2 - 115 + jitter,
      y: scroller.scrollTop + scroller.clientHeight / 2 - 60 + jitter,
    };
  };
  if (catalog) palette.load(catalog);

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

  // live events (SSE notifies, API stays the source of truth)
  sse("/events", {
    onOpen: () => log("Event stream connected."),
    onError: () => log("Event stream reconnecting…", "warn"),
    onMessage: (msg) => {
      try {
        executions.onEvent(JSON.parse(msg.data));
      } catch {
        /* unparsable frame: ignore, never guess */
      }
    },
  });

  log("Editor ready.");
}

boot();
