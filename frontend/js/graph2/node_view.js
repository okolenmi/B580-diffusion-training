/* ---------------------------------------------------------------------------
   graph2/node_view.js -- one node's DOM. It owns its element and rebuilds
   only itself, only when something that shows in it changed; the canvas
   and the other nodes are not touched. A control being typed in keeps focus:
   a param change that came from this node's own control does not rebuild it
   unless another row's `visible_when` depends on that param.

   The card has browser-style tabs (node_tabs.js): Main, Optional, Addons.
   Sockets are elements carrying `data-side` ("in" | "out" | "event-in" |
   "event-out") and `data-port`, so wires attach to them by identity. A wire
   whose port is not drawn (an addon's parameter on another tab, a value
   hidden behind "Show more") runs to the hub circle, which counts them and
   opens the tab that holds the port. A minimized node draws every port as a
   pin, so a wire never loses its end.
   --------------------------------------------------------------------------- */

import { pinOffsets, pinnedHeight } from "./geometry.js";
import { activeTab, addonChoices, optionalSockets, shownOptional, tabOfHidden, tabsOf } from "./node_tabs.js";
import { fieldValues, groupPorts, groupVisible, hasSocket, isField, portVisible } from "./ports.js";
import { buildControl } from "./widgets.js";

const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const checkbox = (title = "") => Object.assign(document.createElement("input"), { type: "checkbox", title });

export class NodeView {
  /**
   * @param {{doc, shapes, node, options: (spec, values)=>Promise<string[]>, note: Function, onTitle: Function}} deps
   */
  constructor({ doc, shapes, node, options, note, onTitle }) {
    this.doc = doc;
    this.shapes = shapes;
    this.node = node;
    this.options = options;
    this.note = note;
    this.onTitle = onTitle;
    this.el = make("div", "g2-node");
    this.el.dataset.uid = node.uid;
    this.status = { state: "idle" };
    this.render();
  }

  /* ----- placement and state ----- */

  place() {
    const { x, y, w } = this.node.layout;
    this.el.style.left = `${x}px`;
    this.el.style.top = `${y}px`;
    this.el.style.width = `${w || 280}px`;
  }

  setSelected(on) {
    this.el.classList.toggle("selected", on);
  }

  /** `{state, text, title}`: the node's run state (idle|resolving|stale|running|done|error|blocked). */
  setStatus(status) {
    this.status = status;
    this.el.dataset.state = status.state;
    const badge = this.el.querySelector(".g2-badge");
    if (badge) {
      badge.textContent = status.text || (status.state === "idle" ? "" : status.state);
      badge.title = status.title || "";
    }
  }

  /** The socket element of a port, wherever it is drawn; a port that is not drawn but exists is reached through the hub. */
  socket(side, port) {
    const real = this.el.querySelector(`.g2-socket[data-side="${side}"][data-port="${CSS.escape(port)}"]`);
    if (real) return real;
    return side === "in" && this.hub && this._knownPort(port) ? this.hub : null;
  }

  /** A port the node has (its own input, an addon's parameter, an addon slot's socket), whether or not it is drawn now. */
  _knownPort(port) {
    const shape = this.shapes.shapeOf(this.node);
    if (shape?.inputs.some((p) => p.name === port)) return true;
    if (port.startsWith("slot:")) return (this.shapes.info(this.node.class_name)?.addon_slots || []).some((s) => s.socket === port);
    const [id] = port.split(".");
    return port.includes(".") && this.node.addons.some((a) => a.id === id);
  }

  /* ----- rendering ----- */

  /** True while the person is typing in one of this node's controls. */
  typing() {
    return this.el.contains(document.activeElement) && /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
  }

  /** A param changed: rebuild unless it came from this node's own control and nothing else depends on it. */
  paramChanged(name, origin) {
    if (origin === "node" && !this._gates(name) && this.typing()) return;
    this.render();
  }

  /** Whether another row's `visible_when` reads this param (`id.param` is an addon's). */
  _gates(name) {
    const dot = name.indexOf(".");
    if (dot > 0) {
      const addon = this.node.addons.find((a) => a.id === name.slice(0, dot));
      const kind = addon && this.shapes.addonKinds.get(addon.kind);
      return Boolean(kind && kind.inputs.some((p) => p.visible_when && p.visible_when[0] === name.slice(dot + 1)));
    }
    const shape = this.shapes.shapeOf(this.node);
    return shape ? shape.inputs.some((p) => p.visible_when && p.visible_when[0] === name) : false;
  }

  render() {
    const keep = this.el.querySelector(".g2-body")?.scrollTop || 0;
    this.el.replaceChildren();
    this.hub = null;
    this.place();
    const shape = this.shapes.shapeOf(this.node);
    const info = this.shapes.info(this.node.class_name);
    const layout = this.node.layout;
    this.el.classList.toggle("collapsed", Boolean(layout.collapsed));
    this.el.classList.toggle("unknown", !shape);
    this.el.append(this._head(info));
    if (!shape) {
      this.el.append(make("div", "g2-unknown", `Unknown node class ${this.node.class_name}: kept as it is, the server will report it.`));
      return;
    }
    const resolved = this.shapes.stateOf(this.node.uid);
    this.el.dataset.resolve = resolved.state;
    if (layout.collapsed) {
      this._pins(shape);
    } else {
      const tabs = tabsOf({ shape, info, node: this.node, kinds: this.shapes.addonKinds });
      const tab = activeTab(tabs, this.node);
      if (tabs.length > 1) this.el.append(this._tabStrip(tabs, tab));
      this.el.append(this._eventStrip("event-in", shape.events_in || [], "top"), this._body(shape, info, resolved, tab), ...this._outputs(shape), this._eventStrip("event-out", shape.events_out || [], "bottom"));
    }
    this._hub(shape);
    this.setStatus(this.status);
    const body = this.el.querySelector(".g2-body");
    if (body) body.scrollTop = keep;
  }

  /* ----- tabs and the hub ----- */

  _tabStrip(tabs, active) {
    const strip = make("div", "g2-tabs");
    strip.setAttribute("role", "tablist");
    for (const tab of tabs) {
      const button = make("button", `g2-tab${tab.id === active ? " active" : ""}`, tab.label);
      button.type = "button";
      button.dataset.tab = tab.id;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", String(tab.id === active));
      if (tab.count) button.append(make("span", "g2-tab-count", String(tab.count)));
      button.addEventListener("click", () => this.doc.setLayout(this.node.uid, { tab: tab.id }));
      strip.append(button);
    }
    return strip;
  }

  /** Wires that end on a port with no socket on the card gather at one circle on the left edge. */
  _hub(shape) {
    const drawn = (port) => this.el.querySelector(`.g2-socket[data-side="in"][data-port="${CSS.escape(port)}"]`);
    const hidden = this.doc.edges.filter((e) => e.to_node === this.node.uid && !drawn(e.to_port) && this._knownPort(e.to_port));
    this.hidden = hidden.map((e) => e.to_port);
    if (!hidden.length) return;
    const hub = make("button", "g2-hub", String(hidden.length));
    hub.type = "button";
    hub.title = hidden.map((e) => `${e.to_port} ← ${this.doc.nodes.get(e.from_node)?.title || this.doc.nodes.get(e.from_node)?.class_name || e.from_node}`).join("\n");
    hub.addEventListener("click", () => this.revealHidden(shape));
    this.hub = hub;
    this.el.append(hub);
  }

  /** Opens the tab (and "more") that holds the first port whose wire ends at the hub. */
  revealHidden(shape = this.shapes.shapeOf(this.node)) {
    if (!this.hidden?.length || !shape) return;
    const tab = tabOfHidden(this.hidden[0], shape, this.node);
    this.doc.setLayout(this.node.uid, tab === "main" ? { tab, advanced: true } : { tab });
  }

  _head(info) {
    const head = make("div", "g2-head");
    const fold = make("button", "g2-fold", this.node.layout.collapsed ? "▸" : "▾");
    fold.type = "button";
    fold.title = this.node.layout.collapsed ? "Expand the node" : "Minimize the node (every port keeps a pin)";
    fold.dataset.action = "fold";
    const title = make("span", "g2-title", this.node.title || (info ? info.display_name : this.node.class_name));
    title.title = info?.doc || this.node.class_name;
    title.addEventListener("dblclick", (e) => { e.stopPropagation(); this.onTitle?.(this.node.uid, title); });
    head.append(fold, title, make("span", "g2-badge"));
    return head;
  }

  _body(shape, info, resolved, tab) {
    const body = make("div", "g2-body");
    if (tab === "optional") body.append(...this._optionalTab(shape));
    else if (tab === "addons") body.append(...this._addonsTab(info));
    else body.append(...this._mainTab(shape, info, resolved));
    return body;
  }

  _mainTab(shape, info, resolved) {
    const out = [];
    if (resolved.state === "resolving") out.push(make("div", "g2-note", "processing… please wait"));
    else if (resolved.message) out.push(make("div", `g2-note ${resolved.state}`, resolved.message));
    else if (resolved.info) out.push(make("div", "g2-note info", resolved.info));
    const node = this.node;
    const values = fieldValues(shape.inputs, node.params);
    const shown = new Set(shownOptional(node, shape.inputs, (name) => this.doc.edgesInto(node.uid, name).length > 0).map((p) => p.name));
    const offCard = new Set(optionalSockets(shape.inputs).filter((p) => !shown.has(p.name)).map((p) => p.name));
    const bindOf = (port) => ({
      name: port.name, value: node.params[port.name], values,
      choices: resolved.choices && resolved.choices[port.name],
      set: (value) => this.doc.setParam(node.uid, port.name, value, "node"),
    });
    const more = Boolean(node.layout.advanced);
    out.push(...this._sections(shape.inputs.filter((p) => !offCard.has(p.name)), info?.groups, values, more, bindOf, resolved));
    const hasMore = shape.inputs.some((p) => p.advanced) || (info?.groups || []).some((g) => g.advanced || g.collapsed);
    if (hasMore) out.push(this._moreToggle(more));
    return out;
  }

  _moreToggle(more) {
    const toggle = make("button", "g2-advanced", more ? "Show less" : "Show more");
    toggle.type = "button";
    toggle.dataset.action = "advanced";
    return toggle;
  }

  /** The ports as labelled sections (plain captions, nothing to fold) with their rows; `bindOf(port)` says where a row reads and writes. */
  _sections(inputs, groupInfo, values, more, bindOf, resolved) {
    const groups = groupPorts(inputs, groupInfo);
    const labelled = groups.length > 1;
    const out = [];
    for (const { group, ports } of groups) {
      if (!groupVisible(group, values, more)) continue;
      const rows = ports.filter((p) => portVisible(p, values, more));
      if (!rows.length) continue;
      const box = make("div", "g2-group");
      box.dataset.group = group.id;
      if (labelled && group.label) box.append(make("div", "g2-group-label", group.label));
      for (const port of rows) box.append(this._row(port, resolved, bindOf(port)));
      out.push(box);
    }
    return out;
  }

  /** The optional inputs: tick one to give it a socket on the card; a wired one stays ticked. */
  _optionalTab(shape) {
    const node = this.node;
    const enabled = new Set(node.layout.optional || []);
    const rows = optionalSockets(shape.inputs).map((port) => {
      const wired = this.doc.edgesInto(node.uid, port.name).length > 0;
      const row = make("label", "g2-opt");
      row.title = port.doc || port.name;
      const box = checkbox();
      box.checked = enabled.has(port.name) || wired;
      box.disabled = wired;
      box.addEventListener("change", () => {
        const next = new Set(node.layout.optional || []);
        if (box.checked) next.add(port.name); else next.delete(port.name);
        this.doc.setLayout(node.uid, { optional: [...next] });
      });
      row.append(box, make("span", "g2-opt-name", port.label || port.name), make("span", "g2-opt-type", wired ? `${port.type} · wired` : port.type));
      return row;
    });
    return [make("div", "g2-note info", "Tick an input to give it a socket on the card."), ...rows];
  }

  _addonsTab(info) {
    const node = this.node;
    const kinds = this.shapes.addonKinds;
    const out = node.addons.map((addon) => this._addon(addon));
    const choices = addonChoices(info, kinds, node.addons);
    if (choices.length) {
      const pick = make("select", "g2-add-addon");
      pick.append(make("option", "", "+ Add addon…"));
      for (const k of choices) pick.append(make("option", "", k.name));
      choices.forEach((k, i) => Object.assign(pick.options[i + 1], { value: k.kind, title: k.doc }));
      pick.options[0].value = "";
      pick.addEventListener("change", () => { if (pick.value) this.doc.addAddon(node.uid, pick.value); });
      out.push(pick);
    } else if (!out.length) {
      out.push(make("div", "g2-note info", "No addons fit this node."));
    }
    return out;
  }

  _addon(addon) {
    const node = this.node;
    const kind = this.shapes.addonKinds.get(addon.kind);
    const box = make("div", `g2-addon${addon.enabled ? "" : " off"}`);
    box.dataset.addon = addon.id;
    const head = make("div", "g2-addon-head");
    const on = checkbox(addon.enabled ? "Switch this addon off" : "Switch this addon on");
    on.checked = addon.enabled;
    on.addEventListener("change", () => this.doc.setAddon(node.uid, addon.id, { enabled: on.checked }));
    const name = make("span", "g2-addon-name", kind?.name || addon.kind);
    name.title = kind?.doc || addon.kind;
    const remove = make("button", "g2-addon-remove", "✕");
    remove.type = "button";
    remove.title = "Remove this addon";
    remove.addEventListener("click", () => this.doc.removeAddon(node.uid, addon.id));
    head.append(on, name, make("span", "g2-addon-id", addon.id), remove);
    box.append(head);
    if (!kind) {
      box.append(make("div", "g2-note error", `Unknown addon kind ${addon.kind}: kept as it is, the server will report it.`));
      return box;
    }
    const values = fieldValues(kind.inputs, addon.params);
    const bindOf = (port) => ({
      name: `${addon.id}.${port.name}`, value: addon.params[port.name], values,
      set: (value) => this.doc.setAddonParam(node.uid, addon.id, port.name, value, "node"),
    });
    box.append(...this._sections(kind.inputs, kind.groups, values, Boolean(node.layout.advanced), bindOf, { state: "ready" }));
    return box;
  }

  _socket(side, port, pinned = false) {
    const socket = make("span", `g2-socket${pinned ? " pin" : ""}`);
    socket.dataset.side = side;
    socket.dataset.port = port.name;
    socket.dataset.type = port.type;
    const wired = side === "out" ? this.doc.edgesFrom(this.node.uid, port.name).length : this.doc.edgesInto(this.node.uid, port.name).length;
    socket.classList.toggle("wired", wired > 0);
    socket.title = `${port.label || port.name}: ${port.type}${wired > 1 ? ` (${wired} wires)` : ""}`;
    return socket;
  }

  /** One port as a row: its socket (when it takes a wire), its label, then the wire note or the value control. */
  _row(port, resolved, bind) {
    const row = make("div", "g2-row");
    row.dataset.port = bind.name;
    row.classList.toggle("optional", isField(port) && !port.required);
    const wires = this.doc.edgesInto(this.node.uid, bind.name);
    if (hasSocket(port)) row.append(this._socket("in", { ...port, name: bind.name }));
    const label = make("label", "g2-label", `${port.label || port.name}${port.required && !isField(port) ? " *" : ""}`);
    label.title = port.doc || port.name;
    row.append(label);
    if (wires.length) {
      row.append(make("span", "g2-wired", port.multi ? `${wires.length} wired` : "wired"));
    } else if (isField(port)) {
      const spec = bind.choices ? { ...port, choices: bind.choices } : port;
      row.append(buildControl(spec, {
        value: bind.value,
        options: () => this.options(spec, bind.values),
        onDraft: bind.set,
        onCommit: bind.set,
        note: this.note,
      }));
      if (resolved.focus === port.name) row.classList.add("focus");
    }
    return row;
  }

  _outputs(shape) {
    if (!shape.outputs.length) return [];
    const outs = make("div", "g2-outs");
    for (const port of shape.outputs) {
      const row = make("div", "g2-out");
      row.append(make("span", "g2-label", port.label || port.name), this._socket("out", port));
      row.title = port.doc || `${port.name}: ${port.type}`;
      outs.append(row);
    }
    return [outs];
  }

  _eventStrip(side, events, edge) {
    const strip = make("div", `g2-events ${edge}`);
    for (const event of events) {
      const socket = make("span", "g2-socket event");
      socket.dataset.side = side;
      socket.dataset.port = event.name;
      socket.title = `${event.label || event.name}${event.payload ? ` (${event.payload})` : ""}`;
      strip.append(socket);
    }
    return strip;
  }

  /** The minimized node: title row, then every port as a pin down the left (inputs) and right (outputs) edge. */
  _pins(shape) {
    const inputs = shape.inputs.filter(hasSocket);
    const height = pinnedHeight(inputs.length, shape.outputs.length);
    const body = make("div", "g2-pins");
    body.style.height = `${height - 28}px`;
    const place = (port, side, y) => {
      const pin = this._socket(side, port, true);
      pin.style.top = `${y}px`;
      pin.append(make("span", "g2-pin-label", port.label || port.name));
      body.append(pin);
    };
    const ys = pinOffsets(inputs.length, height - 28);
    inputs.forEach((port, i) => place(port, "in", ys[i]));
    const yo = pinOffsets(shape.outputs.length, height - 28);
    shape.outputs.forEach((port, i) => place(port, "out", yo[i]));
    this.el.append(this._eventStrip("event-in", shape.events_in || [], "top"), body, this._eventStrip("event-out", shape.events_out || [], "bottom"));
  }
}
