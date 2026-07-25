/* ---------------------------------------------------------------------------
   Node Graph Editor -- interactive version of the old read-only playground.
   ES6 classes throughout (GraphModel/GraphNode/Connection/GraphView), unlike
   this codebase's other .js files, deliberately: this *is* the node-graph
   design the project's OOP rule is about, not a UI feature layered on top
   of one. See docs/node_architecture_refactor_plan.md.
   --------------------------------------------------------------------------- */

(function () {
  "use strict";

  const STORAGE_KEY = "ng_graph_v1";
  const NODE_WIDTH = 270;

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = String(s);
    return d.innerHTML;
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
     type, required, default) live on classInfo -- this only holds the
     per-instance state: position, and widget values for unconnected
     primitive inputs. */
  class GraphNode {
    constructor(id, classInfo, x, y) {
      this.id = id;
      this.classInfo = classInfo;
      this.x = x;
      this.y = y;
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
      this.handleTypes = new Set();          // any type name that appears as *some* node's output -> wire-only
      for (const domain of Object.keys(registry)) {
        for (const c of registry[domain]) for (const o of c.outputs) this.handleTypes.add(o.type);
      }
      this.nodes = new Map();
      this.connections = new Map();
      this._nextId = 1;
    }

    isHandleType(typeStr) {
      return this.handleTypes.has(typeStr);
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
          id: n.id, class_name: n.classInfo.class_name, x: n.x, y: n.y, paramValues: n.paramValues,
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

  /* Rendering + interaction. Owns the DOM, delegates all graph-shape
     decisions to GraphModel. */
  class GraphView {
    constructor(model, els) {
      this.model = model;
      this.els = els; // {palette, canvas, wires, canvasWrap, runBtn, clearBtn, runStatus, results}
      this.pendingWire = null; // {fromNode, fromPort, isOutput} while dragging a new connection
      this.dragState = null;   // {node, startX, startY, origX, origY} while dragging a node
      this._spawnCascade = 0;
    }

    init() {
      this.renderPalette();
      this.els.runBtn.addEventListener("click", () => this.runGraph());
      this.els.clearBtn.addEventListener("click", () => this.clearAll());
      document.addEventListener("mousemove", (e) => this.onDocMouseMove(e));
      document.addEventListener("mouseup", (e) => this.onDocMouseUp(e));
      this.renderAll();
    }

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
          item.textContent = classInfo.class_name;
          item.title = classInfo.doc || "";
          item.addEventListener("click", () => this.spawn(classInfo));
          group.appendChild(item);
        }
        p.appendChild(group);
      }
    }

    spawn(classInfo) {
      const wrap = this.els.canvasWrap;
      const baseX = wrap.scrollLeft + 60 + (this._spawnCascade % 8) * 24;
      const baseY = wrap.scrollTop + 60 + (this._spawnCascade % 8) * 24;
      this._spawnCascade++;
      const node = this.model.addNode(classInfo, baseX, baseY);
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

    renderAll() {
      this.els.canvas.innerHTML = "";
      for (const node of this.model.nodes.values()) this.els.canvas.appendChild(this.buildNodeEl(node));
      for (const node of this.model.nodes.values()) this.measurePorts(node);
      this.redrawWires();
      this.updateRunButton();
    }

    buildNodeEl(node) {
      const ci = node.classInfo;
      const el = document.createElement("div");
      el.className = "ng-node";
      el.style.left = node.x + "px";
      el.style.top = node.y + "px";
      el.style.width = NODE_WIDTH + "px";
      el.dataset.nodeId = node.id;

      const header = document.createElement("div");
      header.className = "ng-node-header";
      header.innerHTML = `
        <button class="ng-node-del" title="Delete node">\u00d7</button>
        <div class="ng-node-title">${escapeHtml(ci.class_name)}</div>
        <div class="ng-node-module">${escapeHtml(ci.module)}</div>
      `;
      header.querySelector(".ng-node-del").addEventListener("click", (e) => {
        e.stopPropagation();
        this.model.removeNode(node.id);
        this.renderAll();
        this.persist();
      });
      header.addEventListener("mousedown", (e) => {
        if (e.target.closest(".ng-node-del")) return;
        this.dragState = { node, startX: e.clientX, startY: e.clientY, origX: node.x, origY: node.y };
      });
      el.appendChild(header);

      if (ci.doc) {
        const doc = document.createElement("div");
        doc.className = "ng-node-doc";
        doc.textContent = ci.doc;
        el.appendChild(doc);
      }

      const inLabel = document.createElement("div");
      inLabel.className = "ng-ports-label";
      inLabel.textContent = "Inputs";
      el.appendChild(inLabel);
      for (const p of ci.inputs) el.appendChild(this.buildInputBlock(node, p));

      const outLabel = document.createElement("div");
      outLabel.className = "ng-ports-label";
      outLabel.textContent = "Outputs";
      el.appendChild(outLabel);
      for (const p of ci.outputs) el.appendChild(this.buildPortRow(node, p, true));

      return el;
    }

    buildPortRow(node, port, isOutput) {
      const row = document.createElement("div");
      row.className = "ng-port" + (isOutput ? " output" : "");
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

      const row = document.createElement("div");
      row.className = "ng-widget-row";
      const current = node.paramValues[port.name];
      if (port.type === "bool") {
        row.innerHTML = `<label style="font-size:11px;"><input type="checkbox" ${current ? "checked" : ""}> use value</label>`;
        row.querySelector("input").addEventListener("change", (e) => {
          node.paramValues[port.name] = e.target.checked;
          this.updateRunButton();
          this.persist();
        });
      } else {
        const inputType = (port.type === "int" || port.type === "float") ? "number" : "text";
        const step = port.type === "int" ? "1" : "any";
        const val = current === undefined || current === null ? "" : current;
        row.innerHTML = `<input type="${inputType}" ${inputType === "number" ? `step="${step}"` : ""} placeholder="${port.required ? "required" : "default: " + (port.default || "\u2014")}" value="${escapeHtml(val)}">`;
        row.querySelector("input").addEventListener("input", (e) => {
          node.paramValues[port.name] = e.target.value === "" ? undefined : coerceWidgetValue(e.target.value, port.type);
          this.refreshPortHighlights();
          this.updateRunButton();
          this.persist();
        });
      }
      wrapper.appendChild(row);
      return wrapper;
    }

    measurePorts(node) {
      const el = this.els.canvas.querySelector(`.ng-node[data-node-id="${node.id}"]`);
      if (!el) return;
      const nodeRect = el.getBoundingClientRect();
      node.portOffsets = { inputs: {}, outputs: {} };
      el.querySelectorAll(".ng-port-dot").forEach((dot) => {
        const r = dot.getBoundingClientRect();
        const offset = { x: r.left + r.width / 2 - nodeRect.left, y: r.top + r.height / 2 - nodeRect.top };
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

    refreshPortHighlights() {
      // Cheap enough to just re-render; keeps widget-vs-wire-vs-unmet state
      // (dot color, hint text) always derived from the model, never hand-toggled.
      this.renderAll();
    }

    beginWire(nodeId, portName, isOutput) {
      this.pendingWire = { nodeId, portName, isOutput };
    }

    canvasMousePos(e) {
      const rect = this.els.canvas.getBoundingClientRect();
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    onDocMouseMove(e) {
      if (this.dragState) {
        const dx = e.clientX - this.dragState.startX;
        const dy = e.clientY - this.dragState.startY;
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
        let path = this.els.wires.querySelector("path.ng-pending");
        if (!path) {
          path = document.createElementNS("http://www.w3.org/2000/svg", "path");
          path.setAttribute("class", "ng-pending");
          this.els.wires.appendChild(path);
        }
        const d = this.pendingWire.isOutput
          ? this.wirePath(from.x, from.y, mouse.x, mouse.y)
          : this.wirePath(mouse.x, mouse.y, from.x, from.y);
        path.setAttribute("d", d);
      }
    }

    onDocMouseUp(e) {
      if (this.dragState) {
        this.dragState = null;
        this.persist();
      }
      if (this.pendingWire) {
        const target = document.elementFromPoint(e.clientX, e.clientY);
        const pending = this.els.wires.querySelector("path.ng-pending");
        if (pending) pending.remove();
        if (target && target.classList && target.classList.contains("ng-port-dot")) {
          this.tryCompleteWire(this.pendingWire, {
            nodeId: target.dataset.nodeId,
            portName: target.dataset.portName,
            isOutput: target.dataset.isOutput === "1",
          });
        }
        this.pendingWire = null;
      }
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
      this.setStatus("Running\u2026");
      try {
        const res = await fetch("/api/nodegraph/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(this.model.toRunPayload()),
        });
        const data = await res.json();
        if (!res.ok) { this.setStatus(data.detail || `HTTP ${res.status}`, true); this.renderResults(null, [], data.detail); return; }
        this.renderResults(data.results, []);
        this.setStatus("Done.");
      } catch (err) {
        this.setStatus("Request failed: " + err.message, true);
      } finally {
        this.updateRunButton();
      }
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
      canvasWrap: document.getElementById("ng-canvas-wrap"),
      runBtn: document.getElementById("ng-run-btn"),
      clearBtn: document.getElementById("ng-clear-btn"),
      runStatus: document.getElementById("ng-run-status"),
      results: document.getElementById("ng-results"),
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
      view.init();
    } catch (err) {
      els.palette.textContent = "Failed to load node registry: " + err.message;
    }
  }

  boot();
})();
