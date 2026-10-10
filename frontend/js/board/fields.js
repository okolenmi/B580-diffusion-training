/* ---------------------------------------------------------------------------
   board/fields.js -- the control inside a *field* block.

   A field block edits one value of the template's draft graph: a node's
   input port, or one of the graph's own settings. The control is the graph
   editor's own (`editor/widgets.js` `buildWidget`), driven from the same port
   description the backend sent, so a value looks and validates the same on
   the page as on the canvas and cannot drift from it.

   `buildWidget` was written against the editor's document (`doc.classOf`,
   `doc.changed`, `node.params`). Rather than fork it, this module hands it a
   tiny stand-in with exactly those three members, backed by the draft:

     node fields   -> the draft graph node's real `params` object;
     graph fields  -> a `params` proxy that reads/writes `graph.memory`.

   Two controls the editor's widget cannot draw are built here:
     * "a number, or auto" (the VRAM budget);
     * a bool that shows its *default* while unset.
   --------------------------------------------------------------------------- */

import { buildWidget, rowVisible } from "../editor/widgets.js";
import { getFieldValue, setFieldValue } from "./model.js";

/** `params` stand-in for a graph-scope field: the one key is the setting's
    short name, and reads/writes go through the document model. */
function memoryParams(doc, ref) {
  const key = ref.name;
  return new Proxy({}, {
    get: (_t, prop) => (prop === key ? getFieldValue(doc, ref) : undefined),
    set: (_t, prop, value) => { if (prop === key) setFieldValue(doc, ref, value); return true; },
    deleteProperty: (_t, prop) => { if (prop === key) setFieldValue(doc, ref, undefined); return true; },
    has: (_t, prop) => prop === key && getFieldValue(doc, ref) !== undefined,
  });
}

function boolControl(descriptor, get, set) {
  const wrap = document.createElement("label");
  wrap.className = "param-label bd-bool";
  const box = document.createElement("input");
  box.type = "checkbox";
  const text = document.createElement("span");
  const show = () => {
    const v = get();
    box.checked = v === undefined || v === null ? descriptor.default === true : v === true;
    text.textContent = box.checked ? "on" : "off";
  };
  box.addEventListener("change", () => { set(box.checked); show(); });
  wrap.append(box, text);
  show();
  return wrap;
}

/** "auto" or a number: a checkbox for auto, and a number box that is
    disabled while auto is on. Stored as the string "auto" or a number. */
function autoNumberControl(descriptor, get, set) {
  const wrap = document.createElement("div");
  wrap.className = "bd-auto";
  const box = document.createElement("input");
  box.type = "checkbox";
  const boxLabel = document.createElement("label");
  boxLabel.className = "bd-auto-label";
  const tx = document.createElement("span");
  tx.textContent = "auto";
  boxLabel.append(box, tx);
  const num = document.createElement("input");
  num.type = "number";
  num.className = "cfg-input";
  if (descriptor.minimum != null) num.min = String(descriptor.minimum);
  if (descriptor.step != null) num.step = String(descriptor.step);
  if (descriptor.unit) num.title = descriptor.unit;
  num.placeholder = "whatever is free";
  const unit = document.createElement("span");
  unit.className = "bd-unit";
  unit.textContent = descriptor.unit || "";

  const read = () => {
    const v = get();
    return v === undefined || v === null ? descriptor.default : v;
  };
  const show = () => {
    const v = read();
    const auto = v === "auto";
    box.checked = auto;
    num.disabled = auto;
    num.value = auto || v === undefined || v === null ? "" : String(v);
  };
  box.addEventListener("change", () => {
    if (box.checked) set("auto");
    else {
      const floor = descriptor.minimum != null ? descriptor.minimum : 0;
      set(Math.max(floor, Number(num.value) || floor));
    }
    show();
  });
  num.addEventListener("change", () => {
    if (num.value === "") { set(descriptor.default === "auto" ? "auto" : undefined); show(); return; }
    const n = parseFloat(num.value);
    if (!Number.isFinite(n)) { show(); return; }
    set(descriptor.minimum != null ? Math.max(descriptor.minimum, n) : n);
    show();
  });
  wrap.append(boxLabel, num, unit);
  show();
  return wrap;
}

/**
 * Build the control for one field block.
 *
 * @param {object} args
 * @param {object} args.block       the block ({ref, ...})
 * @param {object} args.descriptor  the field as /templates/modules describes it
 * @param {object} args.siblings    the node's fields (for visible_when gates)
 * @param {object} args.doc         the draft template document
 * @param {(origin: string) => void} args.onChange  called after a value commit
 * @param {(msg: string, kind?: string) => void} [args.onNote]
 * @returns {{el: HTMLElement, refresh: () => void}}
 */
export function buildFieldControl({ block, descriptor, siblings, doc, onChange, onNote }) {
  const ref = block.ref;
  const holder = document.createElement("div");
  holder.className = "bd-field";
  const hint = document.createElement("div");
  hint.className = "bd-hint";
  hint.hidden = true;
  const control = document.createElement("div");
  control.className = "bd-control";
  holder.append(control, hint);

  const graphNode = ref.scope === "node"
    ? ((doc.graph && doc.graph.nodes) || []).find((n) => n.id === ref.node)
    : null;
  const params = ref.scope === "node"
    ? (graphNode ? (graphNode.params = graphNode.params || {}) : {})
    : memoryParams(doc, { ...ref, name: ref.name });
  const nodeLike = { params };
  const get = () => getFieldValue(doc, ref);
  const set = (v) => { if (setFieldValue(doc, ref, v)) onChange("value"); };

  const refresh = () => {
    if (descriptor.visible_when) {
      const visible = rowVisible(nodeLike, descriptor, siblings);
      holder.classList.toggle("is-inactive", !visible);
      hint.hidden = visible;
      const gate = descriptor.visible_when[0];
      hint.textContent = visible ? "" : `not used with the current "${gate}" setting`;
    }
  };

  const mount = () => {
    let child;
    if (descriptor.type === "bool" && !descriptor.choices) {
      child = boolControl(descriptor, get, set);
    } else if (descriptor.allow_auto) {
      child = autoNumberControl(descriptor, get, set);
    } else if (ref.scope === "graph") {
      // A graph setting with a plain type: the stand-in node serves it.
      child = buildWidget({
        doc: { classOf: () => ({ inputs: [] }), changed: () => onChange("value") },
        node: { params: memoryParams(doc, ref) },
        port: { ...descriptor, name: ref.name }, origin: "page", onNote, onRebuild: mount,
      });
    } else {
      child = buildWidget({
        doc: {
          classOf: () => ({ inputs: siblings }),
          changed: () => { refresh(); onChange("value"); },
        },
        node: nodeLike, port: descriptor, origin: "page", onNote, onRebuild: () => { mount(); refresh(); },
      });
    }
    control.replaceChildren(child);
  };
  mount();
  refresh();
  return { el: holder, refresh };
}
