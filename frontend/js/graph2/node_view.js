/* ---------------------------------------------------------------------------
   graph2/node_view.js -- one node's DOM. It owns its element and rebuilds
   only itself, only when something that shows in it changed; the canvas
   and the other nodes are not touched. A control being typed in keeps focus:
   a param change that came from this node's own control does not rebuild it
   unless another row's `visible_when` depends on that param.

   Sockets are elements carrying `data-side` ("in" | "out" | "event-in" |
   "event-out") and `data-port`, so wires attach to them by identity. When a
   group is folded or the whole node is minimized, the same ports are drawn as
   pins, so a wire never loses its end and a port can always be wired.
   --------------------------------------------------------------------------- */

import { pinOffsets, pinnedHeight } from "./geometry.js";
import { fieldValues, groupPorts, groupVisible, hasSocket, isField, portVisible } from "./ports.js";
import { buildControl } from "./widgets.js";

const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const MAX_STRIP_PINS = 8;

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

  /** The socket element of a port, wherever it is drawn (row, group strip or pin). */
  socket(side, port) {
    return this.el.querySelector(`.g2-socket[data-side="${side}"][data-port="${CSS.escape(port)}"]`);
  }

  /* ----- rendering ----- */

  /** True while the person is typing in one of this node's controls. */
  typing() {
    return this.el.contains(document.activeElement) && /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
  }

  /** A param changed: rebuild unless it came from this node's own control and nothing else depends on it. */
  paramChanged(name, origin) {
    const shape = this.shapes.shapeOf(this.node);
    const gates = shape ? shape.inputs.some((p) => p.visible_when && p.visible_when[0] === name) : false;
    if (origin === "node" && !gates && this.typing()) return;
    this.render();
  }

  render() {
    const keep = this.el.querySelector(".g2-body")?.scrollTop || 0;
    this.el.replaceChildren();
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
      this.el.append(this._eventStrip("event-in", shape.events_in || [], "top"), this._body(shape, info, resolved), ...this._outputs(shape), this._eventStrip("event-out", shape.events_out || [], "bottom"));
    }
    this.setStatus(this.status);
    const body = this.el.querySelector(".g2-body");
    if (body) body.scrollTop = keep;
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

  _body(shape, info, resolved) {
    const body = make("div", "g2-body");
    if (resolved.state === "resolving") body.append(make("div", "g2-note", "processing… please wait"));
    else if (resolved.message) body.append(make("div", `g2-note ${resolved.state}`, resolved.message));
    else if (resolved.info) body.append(make("div", "g2-note info", resolved.info));
    const values = fieldValues(shape.inputs, this.node.params);
    const showAdvanced = Boolean(this.node.layout.advanced);
    const groups = groupPorts(shape.inputs, info?.groups);
    const labelled = groups.length > 1;
    for (const { group, ports } of groups) {
      if (!groupVisible(group, values, showAdvanced)) continue;
      const shown = ports.filter((p) => portVisible(p, values, showAdvanced));
      if (!shown.length) continue;
      body.append(this._group(group, shown, labelled, resolved));
    }
    const hiddenAdvanced = shape.inputs.some((p) => p.advanced) || (info?.groups || []).some((g) => g.advanced);
    if (hiddenAdvanced) {
      const toggle = make("button", "g2-advanced", showAdvanced ? "Hide advanced" : "Show advanced");
      toggle.type = "button";
      toggle.dataset.action = "advanced";
      body.append(toggle);
    }
    return body;
  }

  _group(group, ports, labelled, resolved) {
    const open = this.node.layout.groups && group.id in this.node.layout.groups ? this.node.layout.groups[group.id] : !group.collapsed;
    const box = make("div", "g2-group");
    box.dataset.group = group.id;
    const titled = labelled && group.label;
    if (titled) {
      const wired = ports.filter((p) => this.doc.edgesInto(this.node.uid, p.name).length).length;
      const head = make("div", "g2-group-head");
      head.dataset.action = "group";
      head.dataset.group = group.id;
      head.append(make("span", "g2-chev", open ? "▾" : "▸"), make("span", "g2-group-label", group.label), make("span", "g2-count", `${ports.length}${wired ? `, ${wired} wired` : ""}`));
      if (!open) head.append(this._strip(ports));
      box.append(head);
    }
    if (open || !titled) for (const port of ports) box.append(this._row(port, resolved));
    return box;
  }

  /** The pin strip of a folded group: a real socket per wirable port (the first few; the rest sit behind "+n"). */
  _strip(ports) {
    const strip = make("span", "g2-strip");
    const wirable = ports.filter(hasSocket);
    for (const port of wirable.slice(0, MAX_STRIP_PINS)) strip.append(this._socket("in", port));
    if (wirable.length > MAX_STRIP_PINS) strip.append(make("span", "g2-more", `+${wirable.length - MAX_STRIP_PINS}`));
    return strip;
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

  _row(port, resolved) {
    const row = make("div", "g2-row");
    row.dataset.port = port.name;
    row.classList.toggle("optional", !port.required || isField(port));
    const wires = this.doc.edgesInto(this.node.uid, port.name);
    if (hasSocket(port)) row.append(this._socket("in", port));
    const label = make("label", "g2-label", `${port.label || port.name}${port.required && !isField(port) ? " *" : ""}`);
    label.title = port.doc || port.name;
    row.append(label);
    if (wires.length) {
      row.append(make("span", "g2-wired", port.multi ? `${wires.length} wired` : "wired"));
    } else if (isField(port)) {
      const choices = resolved.choices && resolved.choices[port.name];
      const spec = choices ? { ...port, choices } : port;
      const node = this.node;
      row.append(buildControl(spec, {
        value: node.params[port.name],
        options: () => this.options(spec, fieldValues(this.shapes.shapeOf(node).inputs, node.params)),
        onDraft: (value) => this.doc.setParam(node.uid, port.name, value, "node"),
        onCommit: (value) => this.doc.setParam(node.uid, port.name, value, "node"),
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
