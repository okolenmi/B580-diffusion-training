/* ---------------------------------------------------------------------------
   graph2/ports.js -- what a port is, from the catalog's PortInfo
   (rework/contracts/catalog.md). Pure functions; no DOM, no state.

   The browser decides wire compatibility exactly as the server does: an
   output satisfies an input when the input's type name is in the output's
   `type_mro`; an event output satisfies an event input when the input's
   payload is in the output's `payload_mro`.
   --------------------------------------------------------------------------- */

/** The port takes a wire (everything except `widget_only`). */
export function hasSocket(port) {
  return !port.widget_only;
}

/** The port shows a value control (a primitive type, or a widget-only port). */
export function isField(port) {
  return Boolean(port.field) || Boolean(port.widget_only);
}

/** Why `output` cannot feed `input`, or null when it can. */
export function wireProblem(output, input) {
  if (!hasSocket(input)) return `${input.label || input.name} takes no wire`;
  const mro = output.type_mro || [output.type];
  if (!mro.includes(input.type)) return `${output.type} does not satisfy ${input.type}`;
  return null;
}

/** Why an event output cannot feed an event input, or null. */
export function eventWireProblem(output, input) {
  const mro = output.payload_mro || [output.payload];
  if (input.payload && !mro.includes(input.payload)) return `${output.payload} does not satisfy ${input.payload}`;
  return null;
}

/** `visible_when` is `[otherPort, value | [values]]` against the node's current values; absent means visible. */
export function isVisible(port, values) {
  const gate = port.visible_when;
  if (!gate) return true;
  const [other, expected] = gate;
  const now = values[other];
  return Array.isArray(expected) ? expected.includes(now) : now === expected;
}

/** The value a field shows: the document's param, else the port's default. */
export function fieldValue(port, params) {
  return Object.prototype.hasOwnProperty.call(params, port.name) ? params[port.name] : port.default;
}

/** Values of every field of a node (document params over defaults), for `visible_when` and widgets. */
export function fieldValues(inputs, params) {
  const out = {};
  for (const port of inputs) out[port.name] = fieldValue(port, params);
  return out;
}

/**
 * Inputs arranged by group in display order: `[{group, ports}]`. Ports without a group (or naming an unknown one)
 * fall into the implicit `main` group, which comes first.
 */
export function groupPorts(inputs, groups) {
  const known = new Map((groups || []).map((g) => [g.id, g]));
  const order = [];
  const byId = new Map();
  const slot = (id) => {
    if (!byId.has(id)) {
      const group = known.get(id) || { id, label: id === "main" ? "" : id, collapsed: false, advanced: false, visible_when: null, doc: "" };
      byId.set(id, { group, ports: [] });
      order.push(id);
    }
    return byId.get(id);
  };
  slot("main");
  for (const g of groups || []) slot(g.id);
  for (const port of inputs) slot(known.has(port.group) ? port.group : "main").ports.push(port);
  return order.map((id) => byId.get(id)).filter((entry) => entry.ports.length || entry.group.id !== "main");
}

/** A group is shown when its own `visible_when` holds; `advanced` and `collapsed` (rarely needed) groups and `advanced` ports also need the node's "show more". */
export function groupVisible(group, values, showAdvanced) {
  if ((group.advanced || group.collapsed) && !showAdvanced) return false;
  return isVisible(group, values);
}

export function portVisible(port, values, showAdvanced) {
  if (port.advanced && !showAdvanced) return false;
  return isVisible(port, values);
}
