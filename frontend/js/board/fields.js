/* ---------------------------------------------------------------------------
   board/fields.js -- a *field* card: the name of one value and its control.

   A field card edits one value of the app's draft graph: a node's input port, or one of the graph's own settings. The control is the
   graph editor's own (`graph/widgets.js` `buildControl`), built from the same port description the backend sent, so a model file is
   chosen from the server's list, a prompt is a text area and a number has its bounds exactly as on the node.

   Layout (`fieldLayout`, pure, in model.js): `inline` puts the name at the left and the control at the right (small values),
   `stacked` puts the name above the control (text, lists, pickers). A card's `options.layout` overrides the automatic choice and
   `options.hideLabel` leaves the name out.
   --------------------------------------------------------------------------- */

import { fieldValues, isVisible } from "../graph/ports.js";
import { buildControl } from "../graph/widgets.js";
import { pickerOptions } from "../lib/picker_options.js";
import { fieldLayout, getFieldValue, setFieldValue } from "./model.js";

const h = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

/**
 * @param {object} args
 * @param {object} args.block       the card ({ref, title, options, ...})
 * @param {object} args.descriptor  the field as /boards/modules describes it
 * @param {object[]} args.siblings  the node's fields (for visible_when gates and a picker's scope)
 * @param {object} args.doc         the draft app document
 * @param {string} args.label       the name shown (the card's title, else the field's label)
 * @param {(origin: string) => void} args.onChange  called after a value commit
 * @param {(msg: string, kind?: string) => void} [args.onNote]
 * @returns {{el: HTMLElement, refresh: () => void}}
 */
export function buildFieldControl({ block, descriptor, siblings, doc, label, onChange, onNote }) {
  const ref = block.ref;
  const layout = fieldLayout(block, descriptor);
  const holder = h("div", `bd-field bd-field-${layout}`);
  const name = h("label", "bd-flabel", label);
  name.title = descriptor.doc || label;
  name.hidden = !!(block.options && block.options.hideLabel);
  const control = h("div", "bd-control");
  const hint = h("div", "bd-hint");
  hint.hidden = true;
  holder.append(name, control, hint);

  const node = ref.scope === "node" ? ((doc.graph && doc.graph.nodes) || []).find((n) => n.id === ref.node) : null;
  const params = () => (node ? node.params || {} : {});
  // What the node's other fields hold now (document value over default): the gates and a picker's scope read these.
  const values = () => (ref.scope === "node" ? fieldValues(siblings, params()) : {});

  const refresh = () => {
    if (!descriptor.visible_when) return;
    const visible = isVisible(descriptor, values());
    holder.classList.toggle("is-inactive", !visible);
    hint.hidden = visible;
    hint.textContent = visible ? "" : `not used with the current "${descriptor.visible_when[0]}" setting`;
  };

  const set = (v) => { if (setFieldValue(doc, ref, v)) { refresh(); onChange("value"); } };
  const spec = ref.scope === "graph" ? { ...descriptor, name: ref.name } : descriptor;
  control.append(buildControl(spec, {
    value: getFieldValue(doc, ref),
    options: () => pickerOptions(spec, values()),
    onDraft: set,
    onCommit: set,
    note: onNote,
  }));
  refresh();
  return { el: holder, refresh };
}

