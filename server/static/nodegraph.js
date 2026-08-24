/* ---------------------------------------------------------------------------
   Node Graph Editor -- interactive version of the old read-only playground.
   ES6 classes throughout (GraphModel/GraphNode/Connection/GraphView/
   AssetBrowserDialog), unlike this codebase's other .js files, deliberately:
   this *is* the node-graph design the project's OOP rule is about, not a UI
   feature layered on top of one. See docs/node_architecture_refactor_plan.md.
   --------------------------------------------------------------------------- */

(function () {
  "use strict";

  const STORAGE_KEY = "ng_graph_v1";
  const NODE_WIDTH = 270;
  const ZOOM_MIN = 0.2;
  const ZOOM_MAX = 2.5;

  // Closed set: these are the only types that show a typed-in widget. Every
  // other type name (ModelWeights, TrainableModel, OptimizerHandle, ...) is
  // wire-only. Deliberately hardcoded rather than "derived from whatever
  // appears as some node's output type" -- that derivation looked elegant
  // but was wrong: a node like the checkpoint saver legitimately outputs a
  // `str` (the resolved save path), and that alone made every unrelated
  // `str` input across the whole registry lose its text box.
  const PRIMITIVE_TYPES = new Set(["int", "float", "str", "bool", "Path", "Any", "Callable"]);

  const PATH_KIND_TO_ASSET_KIND = { checkpoint: "checkpoint", dataset: "dataset", lora_output: "lora" };

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = String(s);
    return d.innerHTML;
  }

  function isTypingTarget(el) {
    return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
  }

  function coerceWidgetValue(rawString, typeStr) {
    if (typeStr === "int") return parseInt(rawString, 10);
    if (typeStr === "float") return parseFloat(rawString);
    if (typeStr === "bool") return rawString === "true";
    if (typeStr === "str" || typeStr === "Path") return rawString;
    // Any / unrecognized: best-effort JSON, else pass the raw string through.
    try { return JSON.parse(rawString); } catch (e) { return rawString; }
  }

  /* One spawned node instance on the canvas. Port *declarations* (name,
     type, required, default, doc, path_kind) live on classInfo -- this only
     holds per-instance state: position, and widget values for unconnected
     primitive inputs. */
  class GraphNode {
    constructor(id, classInfo, x, y) {
      this.id = id;
      this.classInfo = classInfo;
      this.x = x;
      this.y = y;
      this.displayMode = "full"; // "full" | "compact" | "minimized"
      this.paramValues = {};
      for (const p of classInfo.inputs) {
        if (p.default !== null && p.default !== undefined) {
          try { this.paramValues[p.name] = JSON.parse(p.default.replace(/'/g, '"')); }
          catch (e) { /* leave unset; server default applies if still unconnected+unset */ }
        }
      }
      this.portOffsets = { inputs: {}, outputs: {} }; // filled in after each render, see GraphView.measurePorts
    }
  }

  class Connection {
    constructor(id, fromNode, fromPort, toNode, toPort) {
      this.id = id;
      this.fromNode = fromNode;
      this.fromPort = fromPort;
      this.toNode = toNode;
      this.toPort = toPort;
    }
  }

  /* Pure graph state + rules -- no DOM in here. GraphView owns rendering
     and turns user gestures into calls on this. */
  class GraphModel {
    constructor(registry) {
      this.registry = registry;             // {domain: [classInfo, ...]}
      this.classByName = {};
      for (const domain of Object.keys(registry)) {
        for (const c of registry[domain]) this.classByName[c.class_name] = c;
      }
      this.nodes = new Map();
      this.connections = new Map();
      this._nextId = 1;
    }

    isHandleType(typeStr) {
      return !PRIMITIVE_TYPES.has(typeStr);
    }

    addNode(classInfo, x, y) {
      const id = "n" + this._nextId++;
      const node = new GraphNode(id, classInfo, x, y);
      this.nodes.set(id, node);
      return node;
    }

    removeNode(nodeId) {
      this.nodes.delete(nodeId);
      for (const [cid, c] of this.connections) {
        if (c.fromNode === nodeId || c.toNode === nodeId) this.connections.delete(cid);
      }
    }

    existingConnectionInto(nodeId, portName) {
      for (const c of this.connections.values()) {
        if (c.toNode === nodeId && c.toPort === portName) return c;
      }
      return null;
    }

    typesCompatible(fromTypeMro, toTypeStr, toTypeMro) {
      if (toTypeStr === "Any" || toTypeMro.includes("Any")) return true;
      if (fromTypeMro.includes("Any")) return true;
      return fromTypeMro.includes(toTypeStr);
    }

    addConnection(fromNode, fromPort, toNode, toPort) {
      if (fromNode === toNode) return null;
      const existing = this.existingConnectionInto(toNode, toPort);
      if (existing) this.connections.delete(existing.id);
      const id = "c" + this._nextId++;
      const conn = new Connection(id, fromNode, fromPort, toNode, toPort);
      this.connections.set(id, conn);
      return conn;
    }

    removeConnection(connId) {
      this.connections.delete(connId);
    }

    /* {ok, problems: [{nodeId, portName, reason}]} -- required handle
       inputs need a connection, required primitive inputs need either a
       connection or a set widget value. */
    validate() {
      const problems = [];
      for (const node of this.nodes.values()) {
        for (const p of node.classInfo.inputs) {
          if (!p.required) continue;
          const connected = !!this.existingConnectionInto(node.id, p.name);
          if (connected) continue;
          if (this.isHandleType(p.type)) {
            problems.push({ nodeId: node.id, portName: p.name, reason: "needs a connection" });
          } else {
            const v = node.paramValues[p.name];
            if (v === undefined || v === null || v === "") {
              problems.push({ nodeId: node.id, portName: p.name, reason: "needs a value" });
            }
          }
        }
      }
      return { ok: problems.length === 0, problems };
    }

    toRunPayload() {
      const nodes = [];
      for (const node of this.nodes.values()) {
        const params = {};
        for (const p of node.classInfo.inputs) {
          if (this.isHandleType(p.type)) continue; // supplied via edges, not params
          const v = node.paramValues[p.name];
          if (v !== undefined && v !== null && v !== "") params[p.name] = v;
        }
        nodes.push({ id: node.id, class_name: node.classInfo.class_name, params });
      }
      const edges = [];
      for (const c of this.connections.values()) {
        edges.push({ from_node: c.fromNode, from_port: c.fromPort, to_node: c.toNode, to_port: c.toPort });
      }
      return { nodes, edges };
    }

    serialize() {
      return {
        nodes: Array.from(this.nodes.values()).map(n => ({
          id: n.id, class_name: n.classInfo.class_name, x: n.x, y: n.y,
          paramValues: n.paramValues, displayMode: n.displayMode,
        })),
        connections: Array.from(this.connections.values()).map(c => ({
          fromNode: c.fromNode, fromPort: c.fromPort, toNode: c.toNode, toPort: c.toPort,
        })),
        nextId: this._nextId,
      };
    }

    restore(saved) {
      for (const n of saved.nodes || []) {
        const classInfo = this.classByName[n.class_name];
        if (!classInfo) { console.warn("Skipping unknown saved node class:", n.class_name); continue; }
        const node = new GraphNode(n.id, classInfo, n.x, n.y);
        node.paramValues = n.paramValues || {};
        node.displayMode = n.displayMode || "full";
        this.nodes.set(n.id, node);
      }
      for (const c of saved.connections || []) {
        if (!this.nodes.has(c.fromNode) || !this.nodes.has(c.toNode)) continue;
        const id = "c" + this._nextId++;
        this.connections.set(id, new Connection(id, c.fromNode, c.fromPort, c.toNode, c.toPort));
      }
      this._nextId = Math.max(this._nextId, (saved.nextId || 1));
    }
  }

  /* Browse/create-folder/pick-a-filename dialog for a "lora_output"-style
     save target. Talks to /api/nodegraph/assets/{kind}/browse and /mkdir --
     both sandboxed server-side (see paths.py's resolve_safe_model_path);
     this dialog never constructs a filesystem path itself, only relative
     path *strings* that the server resolves and validates on every call. */
  class AssetBrowserDialog {
    constructor(assetKind, opts) {
      this.assetKind = assetKind;
      this.onConfirm = opts.onConfirm;
      this.path = "";
      this.filename = "";
      const initial = opts.initialValue || "";
      const slash = initial.lastIndexOf("/");
      if (slash >= 0) { this.path = initial.slice(0, slash); this.filename = initial.slice(slash + 1); }
      else { this.filename = initial; }
      this.el = null;
    }

    async open() {
      this.el = document.createElement("div");
      this.el.className = "ng-modal-overlay";
      this.el.addEventListener("mousedown", (e) => { if (e.target === this.el) this.close(); });
      document.body.appendChild(this.el);
      await this.render();
    }

    close() {
      if (this.el) { this.el.remove(); this.el = null; }
    }

    async render() {
      if (!this.el) return;
      const res = await fetch(`/api/nodegraph/assets/${this.assetKind}/browse?path=${encodeURIComponent(this.path)}`);
      const data = res.ok ? await res.json() : { folders: [], files: [] };

      const segments = this.path ? this.path.split("/") : [];
      let crumbHtml = `<span class="ng-crumb" data-path="">(root)</span>`;
      let acc = "";
      for (const seg of segments) {
        acc = acc ? acc + "/" + seg : seg;
        crumbHtml += ` / <span class="ng-crumb" data-path="${escapeHtml(acc)}">${escapeHtml(seg)}</span>`;
      }

      const entries = data.folders.map(f => `<div class="ng-modal-entry ng-modal-folder" data-name="${escapeHtml(f)}">\u{1F4C1} ${escapeHtml(f)}/</div>`).join("")
        + data.files.map(f => `<div class="ng-modal-entry ng-modal-file" data-name="${escapeHtml(f)}">\u{1F4C4} ${escapeHtml(f)}</div>`).join("");

      this.el.innerHTML = `
        <div class="ng-modal">
          <div class="ng-modal-title">Save into ${escapeHtml(this.assetKind)} directory</div>
          <div class="ng-modal-crumbs">${crumbHtml}</div>
          <div class="ng-modal-list">${entries || '<div class="ng-modal-empty">(empty)</div>'}</div>
          <div class="ng-modal-newfolder">
            <input type="text" id="ng-modal-newfolder-input" placeholder="new folder name">
            <button id="ng-modal-newfolder-btn" type="button">+ Folder</button>
          </div>
          <div class="ng-modal-filename">
            <label>Filename</label>
            <input type="text" id="ng-modal-filename-input" value="${escapeHtml(this.filename)}" placeholder="my_lora.safetensors">
          </div>
          <div class="ng-modal-actions">
            <button id="ng-modal-cancel" type="button">Cancel</button>
            <button id="ng-modal-save" type="button">Save here</button>
          </div>
        </div>`;

      this.el.querySelectorAll(".ng-crumb").forEach(el => {
        el.addEventListener("click", () => { this.path = el.dataset.path; this.render(); });
      });
      this.el.querySelectorAll(".ng-modal-folder").forEach(el => {
        el.addEventListener("click", () => {
          this.path = this.path ? this.path + "/" + el.dataset.name : el.dataset.name;
          this.render();
        });
      });
      this.el.querySelectorAll(".ng-modal-file").forEach(el => {
        el.addEventListener("click", () => {
          this.el.querySelector("#ng-modal-filename-input").value = el.dataset.name;
        });
      });
      this.el.querySelector("#ng-modal-newfolder-btn").addEventListener("click", async () => {
        const name = this.el.querySelector("#ng-modal-newfolder-input").value.trim();
        if (!name) return;
        const rel = this.path ? this.path + "/" + name : name;
        const r = await fetch(`/api/nodegraph/assets/${this.assetKind}/mkdir`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ relative_path: rel }),
        });
        if (r.ok) { this.path = rel; this.render(); }
        else { const err = await r.json(); alert(err.detail || "Could not create folder."); }
      });
      this.el.querySelector("#ng-modal-cancel").addEventListener("click", () => this.close());
      this.el.querySelector("#ng-modal-save").addEventListener("click", () => {
        const filename = this.el.querySelector("#ng-modal-filename-input").value.trim();
        if (!filename) { alert("Enter a filename."); return; }
        if (filename.includes("/") || filename.includes("\\")) { alert("Filename can't contain a path separator -- use the folder browser above instead."); return; }
        const relPath = this.path ? this.path + "/" + filename : filename;
        this.onConfirm(relPath);
        this.close();
      });
    }
  }

  /* Rendering + interaction. Owns the DOM, delegates all graph-shape
     decisions to GraphModel. */
  class GraphView {
    constructor(model, els) {
      this.model = model;
      this.els = els; // {palette, canvas, wires, viewport, canvasWrap, runBtn, clearBtn, runStatus, results, zoomIn, zoomOut, zoomPct}
      this.pendingWire = null; // {nodeId, portName, isOutput} while dragging a new connection
      this.dragState = null;   // {node, startX, startY, origX, origY} while dragging a node
      this.panState = null;    // {startX, startY, origPanX, origPanY} while panning the canvas
      this.spaceHeld = false;
      this.zoom = 1;
      this.panX = 0;
      this.panY = 0;
      this._spawnCascade = 0;
      this.assetCache = {}; // assetKind -> {options, upload_supported, ...}, warmed in init()
      this.monitorConnections = new Map(); // nodeId -> {eventSource, monitorId, quickInfoEl, lastData}
      // Kept alive across renderAll() calls (which rebuild all node DOM from
      // scratch) rather than reconnected every render -- see
      // trackMonitorConnection(). Cleaned up when a node is removed, see
      // cleanupMonitorConnections().
    }

    async init() {
      await this.warmAssetCache();
      this.renderPalette();
      this.els.runBtn.addEventListener("click", () => this.runGraph());
      this.els.stopBtn.addEventListener("click", () => this.stopGraph());
      this.els.clearBtn.addEventListener("click", () => this.clearAll());
      this.els.zoomIn.addEventListener("click", () => this.setZoomCentered(this.zoom + 0.1));
      this.els.zoomOut.addEventListener("click", () => this.setZoomCentered(this.zoom - 0.1));
      this.els.zoomPct.addEventListener("click", () => this.setZoomCentered(1.0));
      this.els.canvasWrap.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
      this.els.canvasWrap.addEventListener("mousedown", (e) => this.onCanvasWrapMouseDown(e), true);
      document.addEventListener("mousemove", (e) => this.onDocMouseMove(e));
      document.addEventListener("mouseup", (e) => this.onDocMouseUp(e));
      document.addEventListener("keydown", (e) => this.onKeyDown(e));
      document.addEventListener("keyup", (e) => this.onKeyUp(e));
      this.applyViewportTransform();
      this.drawOriginMarker();
      this.renderAll();
    }

    drawOriginMarker() {
      const ns = "http://www.w3.org/2000/svg";
      const group = document.createElementNS(ns, "g");
      group.setAttribute("class", "ng-origin-marker");
      const circle = document.createElementNS(ns, "circle");
      circle.setAttribute("cx", "0"); circle.setAttribute("cy", "0"); circle.setAttribute("r", "14");
      const hLine = document.createElementNS(ns, "line");
      hLine.setAttribute("x1", "-20"); hLine.setAttribute("y1", "0"); hLine.setAttribute("x2", "20"); hLine.setAttribute("y2", "0");
      const vLine = document.createElementNS(ns, "line");
      vLine.setAttribute("x1", "0"); vLine.setAttribute("y1", "-20"); vLine.setAttribute("x2", "0"); vLine.setAttribute("y2", "20");
      group.appendChild(circle);
      group.appendChild(hLine);
      group.appendChild(vLine);
      this.els.wires.appendChild(group);
    }

    async warmAssetCache() {
      const kinds = ["checkpoint", "lora", "dataset"];
      await Promise.all(kinds.map(async (kind) => {
        try {
          const res = await fetch(`/api/nodegraph/assets/${kind}`);
          if (res.ok) this.assetCache[kind] = await res.json();
        } catch (e) { console.warn(`Could not load ${kind} asset list:`, e); }
      }));
    }

    async refreshAssetCache(assetKind) {
      try {
        const res = await fetch(`/api/nodegraph/assets/${assetKind}`);
        if (res.ok) this.assetCache[assetKind] = await res.json();
      } catch (e) { console.warn(`Could not refresh ${assetKind} asset list:`, e); }
    }

    // ---- palette / spawn ----

    renderPalette() {
      const p = this.els.palette;
      p.innerHTML = "";
      for (const domain of Object.keys(this.model.registry).sort()) {
        const group = document.createElement("div");
        group.className = "ng-palette-group";
        const title = document.createElement("div");
        title.className = "ng-palette-group-title";
        title.textContent = domain;
        group.appendChild(title);
        for (const classInfo of this.model.registry[domain]) {
          const item = document.createElement("button");
          item.className = "ng-palette-item";
          // display_name is a friendlier label ("Comfy UNet LoRA"); class_name
          // (the real, stable identifier) stays discoverable via the tooltip
          // rather than disappearing, since it's still what saved-graph JSON
          // and error messages elsewhere in this file refer to a node by.
          item.textContent = classInfo.display_name || classInfo.class_name;
          item.title = classInfo.doc
            ? `${classInfo.class_name} \u2014 ${classInfo.doc}`
            : classInfo.class_name;
          item.addEventListener("click", () => this.spawn(classInfo));
          group.appendChild(item);
        }
        p.appendChild(group);
      }
    }

    spawn(classInfo) {
      const rect = this.els.canvasWrap.getBoundingClientRect();
      const logicalCenterX = (rect.width / 2 - this.panX) / this.zoom;
      const logicalCenterY = (rect.height / 2 - this.panY) / this.zoom;
      const cascade = (this._spawnCascade % 8) * 24;
      this._spawnCascade++;
      const node = this.model.addNode(classInfo, logicalCenterX - NODE_WIDTH / 2 + cascade, logicalCenterY - 60 + cascade);
      this.renderAll();
      this.persist();
      return node;
    }

    clearAll() {
      if (!confirm("Clear the whole graph?")) return;
      this.model.nodes.clear();
      this.model.connections.clear();
      localStorage.removeItem(STORAGE_KEY);
      this.renderAll();
    }

    persist() {
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(this.model.serialize())); }
      catch (e) { console.warn("Could not persist graph:", e); }
    }

    // ---- pan / zoom ----

    applyViewportTransform() {
      this.els.viewport.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
      this.els.zoomPct.textContent = Math.round(this.zoom * 100) + "%";
      this.els.canvasWrap.style.backgroundPosition = `${this.panX}px ${this.panY}px`;
    }

    zoomAt(factor, screenX, screenY) {
      const newZoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, this.zoom * factor));
      const actualFactor = newZoom / this.zoom;
      this.panX = screenX - (screenX - this.panX) * actualFactor;
      this.panY = screenY - (screenY - this.panY) * actualFactor;
      this.zoom = newZoom;
      this.applyViewportTransform();
      this.redrawWires();
    }

    setZoomCentered(newZoom) {
      const rect = this.els.canvasWrap.getBoundingClientRect();
      this.zoomAt(Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, newZoom)) / this.zoom, rect.width / 2, rect.height / 2);
    }

    onWheel(e) {
      e.preventDefault();
      const rect = this.els.canvasWrap.getBoundingClientRect();
      const factor = Math.exp(-e.deltaY * 0.001);
      this.zoomAt(factor, e.clientX - rect.left, e.clientY - rect.top);
    }

    onKeyDown(e) {
      if (e.code === "Escape") {
        this.closeSuggestionMenu();
        return;
      }
      if (e.code === "Space" && !isTypingTarget(e.target)) {
        this.spaceHeld = true;
        this.els.canvasWrap.style.cursor = "grab";
        e.preventDefault();
      }
    }

    onKeyUp(e) {
      if (e.code === "Space") {
        this.spaceHeld = false;
        this.els.canvasWrap.style.cursor = "";
      }
    }

    beginPan(e) {
      this.panState = { startX: e.clientX, startY: e.clientY, origPanX: this.panX, origPanY: this.panY };
      this.els.canvasWrap.classList.add("panning");
    }

    onCanvasWrapMouseDown(e) {
      const onNodeOrPort = e.target.closest(".ng-node") || e.target.closest(".ng-port-dot");
      if (this.spaceHeld && onNodeOrPort) {
        e.stopPropagation();
        e.preventDefault();
        this.beginPan(e);
        return;
      }
      if (!onNodeOrPort) {
        this.beginPan(e);
      }
    }

    // ---- rendering ----

    renderAll() {
      this.els.canvas.innerHTML = "";
      for (const node of this.model.nodes.values()) this.els.canvas.appendChild(this.buildNodeEl(node));
      for (const node of this.model.nodes.values()) this.measurePorts(node);
      this.redrawWires();
      this.updateRunButton();
      this.cleanupMonitorConnections();
    }

    cleanupMonitorConnections() {
      for (const [nodeId, conn] of this.monitorConnections) {
        if (!this.model.nodes.has(nodeId)) {
          conn.eventSource.close();
          this.monitorConnections.delete(nodeId);
        }
      }
    }

    buildNodeEl(node) {
      const ci = node.classInfo;
      const mode = node.displayMode || "full";
      const el = document.createElement("div");
      el.className = "ng-node ng-node-" + mode;
      el.style.left = node.x + "px";
      el.style.top = node.y + "px";
      el.style.width = NODE_WIDTH + "px";
      el.dataset.nodeId = node.id;

      const header = document.createElement("div");
      header.className = "ng-node-header";
      header.innerHTML = `
        <div class="ng-node-header-row">
          <div class="ng-node-title" title="${escapeHtml(ci.class_name)}">${escapeHtml(ci.class_name)}</div>
          <div class="ng-node-modes">
            <button class="ng-mode-btn" data-mode="minimized" title="Minimized: name + required/connected dots only">\u2022</button>
            <button class="ng-mode-btn" data-mode="compact" title="Compact: all ports, no widgets or descriptions">\u25b2</button>
            <button class="ng-mode-btn" data-mode="full" title="Full detail">\u25cf</button>
          </div>
          <button class="ng-node-del" title="Delete node">\u00d7</button>
        </div>
        ${mode !== "minimized" ? `<div class="ng-node-module">${escapeHtml(ci.module)}</div>` : ""}
      `;
      header.querySelector(".ng-node-del").addEventListener("click", (e) => {
        e.stopPropagation();
        this.model.removeNode(node.id);
        this.renderAll();
        this.persist();
      });
      header.querySelectorAll(".ng-mode-btn").forEach((btn) => {
        if (btn.dataset.mode === mode) btn.classList.add("active");
        btn.addEventListener("click", (e) => {
          e.stopPropagation();
          node.displayMode = btn.dataset.mode;
          this.renderAll();
          this.persist();
        });
      });
      header.addEventListener("mousedown", (e) => {
        if (e.target.closest(".ng-node-del") || e.target.closest(".ng-mode-btn") || this.spaceHeld) return;
        e.stopPropagation();
        this.dragState = { node, startX: e.clientX, startY: e.clientY, origX: node.x, origY: node.y };
      });
      el.appendChild(header);

      if (mode === "minimized") {
        const cols = document.createElement("div");
        cols.className = "ng-min-columns";
        const inCol = document.createElement("div");
        inCol.className = "ng-min-col ng-min-col-inputs";
        for (const p of ci.inputs) {
          if (!(p.required || this.model.existingConnectionInto(node.id, p.name))) continue;
          inCol.appendChild(this.buildDotOnlyRow(node, p, false));
        }
        const outCol = document.createElement("div");
        outCol.className = "ng-min-col ng-min-col-outputs";
        for (const p of ci.outputs) outCol.appendChild(this.buildDotOnlyRow(node, p, true));
        cols.appendChild(inCol);
        cols.appendChild(outCol);
        el.appendChild(cols);
        this.appendMonitorControls(node, el, mode);
        return el;
      }

      if (ci.doc && mode === "full") {
        const doc = document.createElement("div");
        doc.className = "ng-node-doc";
        doc.textContent = ci.doc;
        el.appendChild(doc);
      }

      const inLabel = document.createElement("div");
      inLabel.className = "ng-ports-label";
      inLabel.textContent = "Inputs";
      el.appendChild(inLabel);
      for (const p of ci.inputs) {
        el.appendChild(mode === "compact" ? this.buildPortRow(node, p, false) : this.buildInputBlock(node, p));
      }

      const outLabel = document.createElement("div");
      outLabel.className = "ng-ports-label";
      outLabel.textContent = "Outputs";
      el.appendChild(outLabel);
      for (const p of ci.outputs) el.appendChild(this.buildPortRow(node, p, true));

      this.appendMonitorControls(node, el, mode);
      return el;
    }

    // ---- MonitorNode-family UI: "Look inside" + live quick-info on the node itself ----

    appendMonitorControls(node, el, mode) {
      if (node.classInfo.domain !== "monitor") return;
      if (!node.paramValues.monitor_id) {
        node.paramValues.monitor_id = this.generateMonitorId();
        this.persist();
      }
      const monitorId = node.paramValues.monitor_id;

      const row = document.createElement("div");
      row.className = "ng-monitor-controls";
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ng-lookinside-btn";
      btn.textContent = mode === "minimized" ? "\u2197" : "Look inside \u2197";
      btn.title = "Open the live dashboard for this monitor in a new tab";
      btn.addEventListener("mousedown", (e) => e.stopPropagation());
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        window.open(`/nodegraph/monitor/${encodeURIComponent(monitorId)}`, "_blank");
      });
      row.appendChild(btn);

      if (mode !== "minimized") {
        const quickInfo = document.createElement("span");
        quickInfo.className = "ng-monitor-quickinfo";
        quickInfo.textContent = "no data yet";
        row.appendChild(quickInfo);
        this.trackMonitorConnection(node.id, monitorId, quickInfo);
      } else {
        this.trackMonitorConnection(node.id, monitorId, null);
      }
      el.appendChild(row);
    }

    generateMonitorId() {
      return "mon-" + Math.random().toString(36).slice(2, 10);
    }

    trackMonitorConnection(nodeId, monitorId, quickInfoEl) {
      const existing = this.monitorConnections.get(nodeId);
      if (existing && existing.monitorId === monitorId) {
        // Same subscription, just rebind to this render's (freshly-built)
        // display element and repaint the last known value immediately so
        // it doesn't flash "no data yet" after every renderAll().
        existing.quickInfoEl = quickInfoEl;
        if (quickInfoEl && existing.lastData) quickInfoEl.textContent = this.formatQuickInfo(existing.lastData);
        return;
      }
      if (existing) existing.eventSource.close(); // monitor_id changed under us -- reconnect

      const source = new EventSource(`/api/nodegraph/monitor/${encodeURIComponent(monitorId)}/stream`);
      const record = { eventSource: source, monitorId, quickInfoEl, lastData: null };
      source.onmessage = (ev) => {
        let data;
        try { data = JSON.parse(ev.data); } catch (e) { return; }
        if (data.step === undefined) return; // control message ({"type":"connected"}) or unrelated shape
        record.lastData = data;
        if (record.quickInfoEl && record.quickInfoEl.isConnected) {
          record.quickInfoEl.textContent = this.formatQuickInfo(data);
        }
      };
      this.monitorConnections.set(nodeId, record);
    }

    formatQuickInfo(data) {
      const stepStr = data.total_steps ? `${data.step}/${data.total_steps}` : `${data.step}`;
      const lossStr = data.loss !== undefined && data.loss !== null ? data.loss.toFixed(4) : "?";
      return `step ${stepStr} \u00b7 loss ${lossStr}`;
    }

    buildPortRow(node, port, isOutput) {
      const row = document.createElement("div");
      row.className = "ng-port" + (isOutput ? " output" : "");
      if (port.doc) row.title = port.doc;
      const connected = !isOutput && !!this.model.existingConnectionInto(node.id, port.name);
      const dotClasses = ["ng-port-dot"];
      if (isOutput) dotClasses.push("output");
      if (!isOutput && port.required && !connected) dotClasses.push("required-unmet");
      if (connected) dotClasses.push("connected");
      row.innerHTML = `
        <span class="${dotClasses.join(" ")}" data-node-id="${node.id}" data-port-name="${escapeHtml(port.name)}" data-is-output="${isOutput ? 1 : 0}"></span>
        <span class="ng-port-name">${escapeHtml(port.name)}${port.required ? "*" : ""}</span>
        <span class="ng-port-type">${escapeHtml(port.type)}</span>
      `;
      const dot = row.querySelector(".ng-port-dot");
      dot.addEventListener("mousedown", (e) => {
        e.stopPropagation();
        this.beginWire(node.id, port.name, isOutput);
      });
      return row;
    }

    buildDotOnlyRow(node, port, isOutput) {
      const row = document.createElement("div");
      row.className = "ng-port ng-port-dotonly" + (isOutput ? " output" : "");
      row.title = port.name + (port.doc ? ": " + port.doc : "") + ` (${port.type})`;
      const connected = !isOutput && !!this.model.existingConnectionInto(node.id, port.name);
      const dotClasses = ["ng-port-dot"];
      if (isOutput) dotClasses.push("output");
      if (!isOutput && port.required && !connected) dotClasses.push("required-unmet");
      if (connected) dotClasses.push("connected");
      const dot = document.createElement("span");
      dot.className = dotClasses.join(" ");
      dot.dataset.nodeId = node.id;
      dot.dataset.portName = port.name;
      dot.dataset.isOutput = isOutput ? "1" : "0";
      dot.addEventListener("mousedown", (e) => {
        e.stopPropagation();
        this.beginWire(node.id, port.name, isOutput);
      });
      row.appendChild(dot);
      return row;
    }

    buildInputBlock(node, port) {
      const wrapper = document.createElement("div");
      const connected = !!this.model.existingConnectionInto(node.id, port.name);
      wrapper.appendChild(this.buildPortRow(node, port, false));

      if (this.model.isHandleType(port.type)) {
        if (connected) {
          const conn = this.model.existingConnectionInto(node.id, port.name);
          const fromNode = this.model.nodes.get(conn.fromNode);
          const hint = document.createElement("div");
          hint.className = "ng-widget-row";
          hint.innerHTML = `<span class="ng-port-hint">\u2190 ${escapeHtml(fromNode ? fromNode.classInfo.class_name : "?")}.${escapeHtml(conn.fromPort)}</span>`;
          wrapper.appendChild(hint);
        }
        return wrapper; // handle types are wire-only, no widget either way
      }

      if (connected) return wrapper; // has a value via wire, no widget needed

      if (port.path_kind === "lora_output") {
        wrapper.appendChild(this.buildSaveAsWidget(node, port));
        return wrapper;
      }
      if (port.path_kind) {
        wrapper.appendChild(this.buildPickerWidget(node, port));
        return wrapper;
      }

      const row = document.createElement("div");
      row.className = "ng-widget-row";
      const current = node.paramValues[port.name];
      if (port.type === "bool") {
        row.innerHTML = `<label class="ng-checkbox-label"><input type="checkbox" ${current ? "checked" : ""}> true</label>`;
        row.querySelector("input").addEventListener("change", (e) => {
          node.paramValues[port.name] = e.target.checked;
          this.updatePortDotState(node.id, port.name);
          this.updateRunButton();
          this.persist();
        });
      } else {
        const inputType = (port.type === "int" || port.type === "float") ? "number" : "text";
        const step = port.type === "int" ? "1" : "any";
        const val = current === undefined || current === null ? "" : current;
        const input = document.createElement("input");
        input.type = inputType;
        if (inputType === "number") input.step = step;
        input.placeholder = port.required ? "required" : "default: " + (port.default || "\u2014");
        input.value = val;
        if (port.doc) input.title = port.doc;
        // Targeted update only -- never renderAll() from a keystroke. An
        // earlier version called renderAll() here, which rebuilds every
        // node's DOM (including this very input) on every character typed,
        // dropping focus after one keystroke. This just flips a class on
        // the one dot that could be affected.
        input.addEventListener("input", (e) => {
          node.paramValues[port.name] = e.target.value === "" ? undefined : coerceWidgetValue(e.target.value, port.type);
          this.updatePortDotState(node.id, port.name);
          this.updateRunButton();
          this.persist();
        });
        row.appendChild(input);
      }
      wrapper.appendChild(row);
      return wrapper;
    }

    buildPickerWidget(node, port) {
      const assetKind = PATH_KIND_TO_ASSET_KIND[port.path_kind] || port.path_kind;
      const row = document.createElement("div");
      row.className = "ng-widget-row ng-picker-row";
      const select = document.createElement("select");
      if (port.doc) select.title = port.doc;
      this.populatePickerOptions(select, assetKind, node.paramValues[port.name]);
      select.addEventListener("change", (e) => {
        node.paramValues[port.name] = e.target.value || undefined;
        this.updatePortDotState(node.id, port.name);
        this.updateRunButton();
        this.persist();
      });
      row.appendChild(select);

      const info = this.assetCache[assetKind];
      if (!info || info.upload_supported) {
        const uploadBtn = document.createElement("button");
        uploadBtn.type = "button";
        uploadBtn.className = "ng-picker-upload";
        uploadBtn.textContent = "\u2191";
        uploadBtn.title = "Upload a new file";
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.style.display = "none";
        fileInput.addEventListener("change", async () => {
          const file = fileInput.files[0];
          if (!file) return;
          const ok = await this.uploadFile(assetKind, file);
          if (!ok) return;
          await this.refreshAssetCache(assetKind);
          this.populatePickerOptions(select, assetKind, file.name);
          node.paramValues[port.name] = file.name;
          this.updatePortDotState(node.id, port.name);
          this.updateRunButton();
          this.persist();
        });
        uploadBtn.addEventListener("click", () => fileInput.click());
        row.appendChild(uploadBtn);
        row.appendChild(fileInput);
      }
      return row;
    }

    buildSaveAsWidget(node, port) {
      const assetKind = PATH_KIND_TO_ASSET_KIND[port.path_kind] || port.path_kind;
      const row = document.createElement("div");
      row.className = "ng-widget-row ng-picker-row";
      const input = document.createElement("input");
      input.type = "text";
      input.placeholder = "e.g. style_v2/checkpoint_1000.safetensors";
      input.value = node.paramValues[port.name] || "";
      if (port.doc) input.title = port.doc;
      input.addEventListener("input", (e) => {
        node.paramValues[port.name] = e.target.value === "" ? undefined : e.target.value;
        this.updatePortDotState(node.id, port.name);
        this.updateRunButton();
        this.persist();
      });
      const browseBtn = document.createElement("button");
      browseBtn.type = "button";
      browseBtn.className = "ng-picker-upload";
      browseBtn.textContent = "\u2026";
      browseBtn.title = "Browse / Save As";
      browseBtn.addEventListener("click", () => {
        const dialog = new AssetBrowserDialog(assetKind, {
          initialValue: node.paramValues[port.name] || "",
          onConfirm: (relPath) => {
            node.paramValues[port.name] = relPath;
            input.value = relPath;
            this.updatePortDotState(node.id, port.name);
            this.updateRunButton();
            this.persist();
          },
        });
        dialog.open();
      });
      row.appendChild(input);
      row.appendChild(browseBtn);
      return row;
    }

    populatePickerOptions(select, assetKind, currentValue) {
      const info = this.assetCache[assetKind];
      const options = info ? info.options : [];
      select.innerHTML = `<option value="">\u2014 choose \u2014</option>` +
        options.map(o => `<option value="${escapeHtml(o.value)}"${o.value === currentValue ? " selected" : ""}>${escapeHtml(o.label)}</option>`).join("");
    }

    async uploadFile(assetKind, file) {
      const form = new FormData();
      form.append("file", file);
      try {
        const res = await fetch(`/api/nodegraph/assets/${assetKind}/upload`, { method: "POST", body: form });
        if (!res.ok) { const err = await res.json(); alert(err.detail || "Upload failed."); return false; }
        return true;
      } catch (e) {
        alert("Upload failed: " + e.message);
        return false;
      }
    }

    // ---- targeted (non-destructive) updates ----

    updatePortDotState(nodeId, portName) {
      const dot = this.els.canvas.querySelector(
        `.ng-port-dot[data-node-id="${nodeId}"][data-is-output="0"][data-port-name="${CSS.escape(portName)}"]`);
      if (!dot) return;
      const node = this.model.nodes.get(nodeId);
      const port = node.classInfo.inputs.find(p => p.name === portName);
      const connected = !!this.model.existingConnectionInto(nodeId, portName);
      const v = node.paramValues[portName];
      const unmet = !!port.required && !connected && (v === undefined || v === null || v === "");
      dot.classList.toggle("required-unmet", unmet);
      dot.classList.toggle("connected", connected);
    }

    // ---- ports / wires ----

    measurePorts(node) {
      const el = this.els.canvas.querySelector(`.ng-node[data-node-id="${node.id}"]`);
      if (!el) return;
      const nodeRect = el.getBoundingClientRect();
      node.portOffsets = { inputs: {}, outputs: {} };
      el.querySelectorAll(".ng-port-dot").forEach((dot) => {
        const r = dot.getBoundingClientRect();
        // getBoundingClientRect() reflects the current CSS zoom transform,
        // so these deltas are in screen pixels; divide by zoom to store
        // them in the same logical units as node.x/node.y.
        const offset = { x: (r.left + r.width / 2 - nodeRect.left) / this.zoom, y: (r.top + r.height / 2 - nodeRect.top) / this.zoom };
        const bucket = dot.dataset.isOutput === "1" ? node.portOffsets.outputs : node.portOffsets.inputs;
        bucket[dot.dataset.portName] = offset;
      });
    }

    portCanvasPos(nodeId, portName, isOutput) {
      const node = this.model.nodes.get(nodeId);
      if (!node) return null;
      const bucket = isOutput ? node.portOffsets.outputs : node.portOffsets.inputs;
      const off = bucket[portName];
      if (!off) return null;
      return { x: node.x + off.x, y: node.y + off.y };
    }

    wirePath(x1, y1, x2, y2) {
      const midX = (x1 + x2) / 2;
      return `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;
    }

    redrawWires() {
      const svg = this.els.wires;
      svg.querySelectorAll("path.ng-wire").forEach(p => p.remove());
      for (const c of this.model.connections.values()) {
        const from = this.portCanvasPos(c.fromNode, c.fromPort, true);
        const to = this.portCanvasPos(c.toNode, c.toPort, false);
        if (!from || !to) continue;
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("class", "ng-wire");
        path.setAttribute("d", this.wirePath(from.x, from.y, to.x, to.y));
        path.dataset.connId = c.id;
        path.addEventListener("click", () => {
          this.model.removeConnection(c.id);
          this.renderAll();
          this.persist();
        });
        svg.appendChild(path);
      }
    }

    beginWire(nodeId, portName, isOutput) {
      this.pendingWire = { nodeId, portName, isOutput };
      this.highlightCompatibleTargets(nodeId, portName, isOutput);
    }

    highlightCompatibleTargets(sourceNodeId, sourcePortName, sourceIsOutput) {
      const sourceNode = this.model.nodes.get(sourceNodeId);
      if (!sourceNode) return;
      const sourcePort = (sourceIsOutput ? sourceNode.classInfo.outputs : sourceNode.classInfo.inputs)
        .find(p => p.name === sourcePortName);
      if (!sourcePort) return;

      this.els.canvas.querySelectorAll(".ng-port-dot").forEach((dot) => {
        const isOutputDot = dot.dataset.isOutput === "1";
        if (isOutputDot === sourceIsOutput || dot.dataset.nodeId === sourceNodeId) return;
        const targetNode = this.model.nodes.get(dot.dataset.nodeId);
        if (!targetNode) return;
        const targetPort = (isOutputDot ? targetNode.classInfo.outputs : targetNode.classInfo.inputs)
          .find(p => p.name === dot.dataset.portName);
        if (!targetPort) return;
        const compatible = sourceIsOutput
          ? this.model.typesCompatible(sourcePort.type_mro, targetPort.type, targetPort.type_mro)
          : this.model.typesCompatible(targetPort.type_mro, sourcePort.type, sourcePort.type_mro);
        if (compatible) dot.classList.add("compatible-target");
      });
    }

    clearCompatibleHighlights() {
      this.els.canvas.querySelectorAll(".ng-port-dot.compatible-target")
        .forEach(dot => dot.classList.remove("compatible-target"));
    }

    // clientX/clientY -> logical canvas coordinates. canvas.getBoundingClientRect()
    // already reflects the current pan+zoom (it's a live transformed element),
    // so this only needs to divide out the zoom -- no separate pan bookkeeping.
    canvasMousePos(e) {
      const rect = this.els.canvas.getBoundingClientRect();
      return { x: (e.clientX - rect.left) / this.zoom, y: (e.clientY - rect.top) / this.zoom };
    }

    onDocMouseMove(e) {
      if (this.panState) {
        this.panX = this.panState.origPanX + (e.clientX - this.panState.startX);
        this.panY = this.panState.origPanY + (e.clientY - this.panState.startY);
        this.applyViewportTransform();
        this.redrawWires();
        return;
      }
      if (this.dragState) {
        const dx = (e.clientX - this.dragState.startX) / this.zoom;
        const dy = (e.clientY - this.dragState.startY) / this.zoom;
        const node = this.dragState.node;
        node.x = this.dragState.origX + dx;
        node.y = this.dragState.origY + dy;
        const el = this.els.canvas.querySelector(`.ng-node[data-node-id="${node.id}"]`);
        if (el) { el.style.left = node.x + "px"; el.style.top = node.y + "px"; }
        this.redrawWires();
        return;
      }
      if (this.pendingWire) {
        const from = this.portCanvasPos(this.pendingWire.nodeId, this.pendingWire.portName, this.pendingWire.isOutput);
        if (!from) return;
        const mouse = this.canvasMousePos(e);
        let path = this.els.wiresTop.querySelector("path.ng-pending");
        if (!path) {
          path = document.createElementNS("http://www.w3.org/2000/svg", "path");
          path.setAttribute("class", "ng-pending");
          this.els.wiresTop.appendChild(path);
        }
        const d = this.pendingWire.isOutput
          ? this.wirePath(from.x, from.y, mouse.x, mouse.y)
          : this.wirePath(mouse.x, mouse.y, from.x, from.y);
        path.setAttribute("d", d);
      }
    }

    onDocMouseUp(e) {
      if (this.panState) {
        this.panState = null;
        this.els.canvasWrap.classList.remove("panning");
      }
      if (this.dragState) {
        this.dragState = null;
        this.persist();
      }
      if (this.pendingWire) {
        const target = document.elementFromPoint(e.clientX, e.clientY);
        const pending = this.els.wiresTop.querySelector("path.ng-pending");
        if (pending) pending.remove();
        this.clearCompatibleHighlights();
        const wire = this.pendingWire;
        this.pendingWire = null;
        if (target && target.classList && target.classList.contains("ng-port-dot")) {
          this.tryCompleteWire(wire, {
            nodeId: target.dataset.nodeId,
            portName: target.dataset.portName,
            isOutput: target.dataset.isOutput === "1",
          });
        } else {
          this.suggestNodesForDroppedWire(wire, e.clientX, e.clientY);
        }
      }
    }

    // ---- compatible-node suggestion menu (dropping a wire on empty space) ----

    suggestNodesForDroppedWire(wire, screenX, screenY) {
      const node = this.model.nodes.get(wire.nodeId);
      if (!node) return;
      const port = (wire.isOutput ? node.classInfo.outputs : node.classInfo.inputs).find(p => p.name === wire.portName);
      if (!port) return;

      const matches = [];
      for (const domain of Object.keys(this.model.registry)) {
        for (const classInfo of this.model.registry[domain]) {
          const candidates = wire.isOutput ? classInfo.inputs : classInfo.outputs;
          for (const candidate of candidates) {
            const compatible = wire.isOutput
              ? this.model.typesCompatible(port.type_mro, candidate.type, candidate.type_mro)
              : this.model.typesCompatible(candidate.type_mro, port.type, port.type_mro);
            if (compatible) { matches.push({ classInfo, port: candidate }); break; }
          }
        }
      }
      this.showSuggestionMenu(screenX, screenY, matches, wire);
    }

    showSuggestionMenu(screenX, screenY, matches, wire) {
      this.closeSuggestionMenu();
      const menu = document.createElement("div");
      menu.className = "ng-suggest-menu";
      menu.style.left = Math.max(4, Math.min(screenX, window.innerWidth - 220)) + "px";
      menu.style.top = Math.max(4, Math.min(screenY, window.innerHeight - 300)) + "px";

      const header = document.createElement("div");
      header.className = "ng-suggest-header";
      header.textContent = wire.isOutput ? "Connect to an input on\u2026" : "Connect to an output on\u2026";
      menu.appendChild(header);

      if (matches.length === 0) {
        const empty = document.createElement("div");
        empty.className = "ng-suggest-empty";
        empty.textContent = "No compatible nodes.";
        menu.appendChild(empty);
      } else {
        for (const m of matches) {
          const item = document.createElement("div");
          item.className = "ng-suggest-item";
          item.innerHTML = `${escapeHtml(m.classInfo.class_name)}<div class="ng-suggest-item-port">${wire.isOutput ? "\u2192" : "\u2190"} ${escapeHtml(m.port.name)}</div>`;
          item.addEventListener("click", () => {
            this.spawnAndConnect(m.classInfo, m.port, wire, screenX, screenY);
            this.closeSuggestionMenu();
          });
          menu.appendChild(item);
        }
      }

      document.body.appendChild(menu);
      this._suggestMenu = menu;
      // Dismiss on the next click anywhere outside the menu. Deferred by a
      // tick so the mouseup that opened the menu doesn't immediately count
      // as the dismissing click.
      setTimeout(() => {
        this._suggestMenuDismiss = (ev) => { if (!menu.contains(ev.target)) this.closeSuggestionMenu(); };
        document.addEventListener("mousedown", this._suggestMenuDismiss);
      }, 0);
    }

    closeSuggestionMenu() {
      if (this._suggestMenu) { this._suggestMenu.remove(); this._suggestMenu = null; }
      if (this._suggestMenuDismiss) {
        document.removeEventListener("mousedown", this._suggestMenuDismiss);
        this._suggestMenuDismiss = null;
      }
    }

    spawnAndConnect(classInfo, matchedPort, wire, screenX, screenY) {
      const pos = this.canvasMousePos({ clientX: screenX, clientY: screenY });
      const newNode = this.model.addNode(classInfo, pos.x - NODE_WIDTH / 2, pos.y - 20);
      if (wire.isOutput) {
        this.model.addConnection(wire.nodeId, wire.portName, newNode.id, matchedPort.name);
      } else {
        this.model.addConnection(newNode.id, matchedPort.name, wire.nodeId, wire.portName);
      }
      this.renderAll();
      this.persist();
    }

    tryCompleteWire(a, b) {
      let out = a.isOutput ? a : (b.isOutput ? b : null);
      let inp = !a.isOutput ? a : (!b.isOutput ? b : null);
      if (!out || !inp || out === inp || out.nodeId === inp.nodeId) return;

      const outNode = this.model.nodes.get(out.nodeId);
      const inNode = this.model.nodes.get(inp.nodeId);
      if (!outNode || !inNode) return;
      const outPort = outNode.classInfo.outputs.find(p => p.name === out.portName);
      const inPort = inNode.classInfo.inputs.find(p => p.name === inp.portName);
      if (!outPort || !inPort) return;

      if (!this.model.typesCompatible(outPort.type_mro, inPort.type, inPort.type_mro)) {
        this.setStatus(`Can't connect ${outPort.type} \u2192 ${inPort.type}: incompatible types.`, true);
        return;
      }
      this.model.addConnection(out.nodeId, out.portName, inp.nodeId, inp.portName);
      this.renderAll();
      this.persist();
    }

    // ---- run ----

    setStatus(text, isError) {
      this.els.runStatus.textContent = text;
      this.els.runStatus.style.color = isError ? "var(--red)" : "var(--text-dim)";
    }

    updateRunButton() {
      const { ok, problems } = this.model.validate();
      this.els.runBtn.disabled = this.model.nodes.size === 0;
      if (this.model.nodes.size === 0) { this.setStatus(""); return; }
      this.setStatus(ok ? "Graph is complete." : `${problems.length} required input(s) unset.`, !ok);
    }

    async runGraph() {
      const { ok, problems } = this.model.validate();
      if (!ok) {
        this.renderResults(null, problems);
        return;
      }
      this.els.runBtn.disabled = true;
      this.els.stopBtn.style.display = "inline-block";
      this.els.stopBtn.disabled = false;
      this.setStatus("Running\u2026");
      try {
        const res = await fetch("/api/nodegraph/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(this.model.toRunPayload()),
        });
        const data = await res.json();
        if (!res.ok) { this.setStatus(data.detail || `HTTP ${res.status}`, true); this.renderResults(null, [], data.detail); return; }
        this.currentExecutionId = data.execution_id;
        await this.pollExecution(data.execution_id);
      } catch (err) {
        this.setStatus("Request failed: " + err.message, true);
      } finally {
        this.currentExecutionId = null;
        this.els.stopBtn.style.display = "none";
        this.updateRunButton();
      }
    }

    async pollExecution(executionId) {
      // 1s between polls -- frequent enough to feel responsive without
      // hammering the server for a run that can last a long time.
      while (true) {
        await new Promise(r => setTimeout(r, 1000));
        const res = await fetch(`/api/nodegraph/run/${executionId}`);
        if (!res.ok) { this.setStatus(`HTTP ${res.status} polling status`, true); return; }
        const data = await res.json();
        if (data.status === "running") continue;
        if (data.status === "finished") {
          this.renderResults(data.results, []);
          this.setStatus("Done.");
        } else if (data.status === "stopped") {
          this.renderResults(data.results, []);
          this.setStatus("Stopped -- results below are from before the stop.");
        } else {
          this.setStatus(data.error || "Failed.", true);
          this.renderResults(null, [], data.error);
        }
        return;
      }
    }

    async stopGraph() {
      if (!this.currentExecutionId) return;
      this.els.stopBtn.disabled = true;
      this.setStatus("Stopping\u2026");
      try {
        await fetch(`/api/nodegraph/run/${this.currentExecutionId}/stop`, { method: "POST" });
      } catch (err) {
        this.setStatus("Stop request failed: " + err.message, true);
      }
      // Not setting status/results here -- the pollExecution() loop
      // already in flight will pick up status="stopped" on its next poll
      // and finish the run's normal cleanup (button states etc).
    }

    renderResults(results, problems, hardError) {
      const el = this.els.results;
      el.innerHTML = "";
      if (hardError) {
        el.innerHTML = `<div class="ng-result-line ng-result-err">${escapeHtml(hardError)}</div>`;
        return;
      }
      if (problems && problems.length) {
        el.innerHTML = `<div class="ng-result-line ng-result-err">Can't run yet:</div>` + problems.map(p => {
          const node = this.model.nodes.get(p.nodeId);
          const title = node ? node.classInfo.class_name : p.nodeId;
          return `<div class="ng-result-line ng-result-err">&nbsp;&nbsp;${escapeHtml(title)}.${escapeHtml(p.portName)} ${escapeHtml(p.reason)}</div>`;
        }).join("");
        return;
      }
      if (!results) return;
      const reached = new Set(results.map(r => r.node_id));
      for (const r of results) {
        const node = this.model.nodes.get(r.node_id);
        const title = node ? node.classInfo.class_name : r.node_id;
        if (r.ok) {
          el.innerHTML += `<div class="ng-result-line ng-result-ok">\u2713 ${escapeHtml(title)}: ${escapeHtml(JSON.stringify(r.outputs))}</div>`;
        } else {
          el.innerHTML += `<div class="ng-result-line ng-result-err">\u2717 ${escapeHtml(title)}: ${escapeHtml(r.error)}</div>`;
        }
      }
      for (const node of this.model.nodes.values()) {
        if (!reached.has(node.id)) {
          el.innerHTML += `<div class="ng-result-line ng-result-skip">\u2014 ${escapeHtml(node.classInfo.class_name)}: not reached (upstream failure)</div>`;
        }
      }
    }
  }

  async function boot() {
    const els = {
      palette: document.getElementById("ng-palette"),
      canvas: document.getElementById("ng-canvas"),
      wires: document.getElementById("ng-wires"),
      wiresTop: document.getElementById("ng-wires-top"),
      viewport: document.getElementById("ng-viewport"),
      canvasWrap: document.getElementById("ng-canvas-wrap"),
      runBtn: document.getElementById("ng-run-btn"),
      stopBtn: document.getElementById("ng-stop-btn"),
      clearBtn: document.getElementById("ng-clear-btn"),
      runStatus: document.getElementById("ng-run-status"),
      results: document.getElementById("ng-results"),
      zoomIn: document.getElementById("ng-zoom-in"),
      zoomOut: document.getElementById("ng-zoom-out"),
      zoomPct: document.getElementById("ng-zoom-pct"),
    };
    try {
      const res = await fetch("/api/nodegraph/registry");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const registry = await res.json();
      const model = new GraphModel(registry);
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try { model.restore(JSON.parse(saved)); } catch (e) { console.warn("Could not restore saved graph:", e); }
      }
      const view = new GraphView(model, els);
      await view.init();
    } catch (err) {
      els.palette.textContent = "Failed to load node registry: " + err.message;
    }
  }

  boot();
})();
