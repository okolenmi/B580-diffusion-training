/* ---------------------------------------------------------------------------
   graph2/widgets.js -- the ONE place a port or a setting becomes a control.

   `buildControl(spec, {value, onDraft, onCommit, options, note})`
     spec     a PortInfo (catalog.md) or a SettingSpec (settings.md: `key`
              instead of `name`, `type: "choice"`); both are read the same way
     value    what is stored now, or undefined for "the default applies"
     onDraft  a valid value while it is still being typed (not yet final)
     onCommit the final value; `undefined` removes the stored value
     options  `() => Promise<string[]>`, the names a picker offers
     note     `(text, kind) => void`, short messages

   Order of decision: picker -> choices -> bool -> int/float (slider with the
   `ui.widget` hint) -> list/dict/tuple as JSON -> text (textarea with a hint).
   An unset bool shows its default: the node runs with that value, so an
   unchecked box would say the opposite. The box stands alone; its state is its value. A setting that allows "auto" gets an
   "auto" choice next to its number.
   --------------------------------------------------------------------------- */

const JSON_TYPES = new Set(["list", "dict", "tuple"]);

const make = (tag, className, attrs = {}) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  Object.assign(node, attrs);
  return node;
};

const specName = (spec) => spec.name ?? spec.key;
const has = (v) => v !== undefined && v !== null;

function defaultLabel(spec) {
  return has(spec.default) && spec.default !== "" ? `(default: ${spec.default})` : "(default)";
}

function selectControl(spec, { value, onCommit }, choices) {
  const select = make("select", "cfg-input", { title: spec.doc || specName(spec) });
  select.append(make("option", "", { value: "", textContent: defaultLabel(spec) }));
  for (const c of choices) select.append(make("option", "", { value: c, textContent: c }));
  if (has(value) && value !== "" && !choices.includes(value)) select.append(make("option", "", { value, textContent: `${value} (not available)` }));
  select.value = has(value) ? String(value) : "";
  select.addEventListener("change", () => onCommit(select.value || undefined));
  return select;
}

function pickerControl(spec, ctx) {
  const pending = has(ctx.value) && ctx.value !== "" ? [String(ctx.value)] : [];
  const select = selectControl(spec, ctx, pending);
  Promise.resolve(ctx.options ? ctx.options() : []).then((names) => {
    const current = has(ctx.value) ? String(ctx.value) : "";
    select.replaceChildren(make("option", "", { value: "", textContent: has(spec.default) && spec.default !== "" ? `(default: ${spec.default})` : "(choose…)" }));
    for (const n of names) select.append(make("option", "", { value: n, textContent: n }));
    if (current && !names.includes(current)) select.append(make("option", "", { value: current, textContent: `${current} (not on server)` }));
    select.value = current;
  }).catch(() => { select.disabled = true; select.title = "file list unavailable"; });
  return select;
}

function boolControl(spec, { value, onCommit }) {
  const box = make("input", "", { type: "checkbox", title: spec.doc || specName(spec) });
  box.checked = has(value) ? value === true : spec.default === true;
  box.addEventListener("change", () => onCommit(box.checked)); // an explicit false overrides an earlier true
  return box;
}

/** The spinner's step for a float with no declared step: one unit of its leading digit, so 0.0002 goes to 0.0003 and 3 to 4. */
function stepFor(value, fallback) {
  const base = Math.abs(Number.isFinite(value) && value !== 0 ? value : fallback || 0);
  return base ? 10 ** Math.floor(Math.log10(base) + 1e-9) : 0.1;
}

function numberControl(spec, { value, onDraft, onCommit, note }) {
  const whole = spec.type === "int";
  const parse = (raw) => (whole ? parseInt(raw, 10) : parseFloat(raw));
  const input = make("input", "cfg-input", { type: "number", title: spec.doc || specName(spec) });
  const adaptive = !whole && !has(spec.step);
  const fit = () => { if (adaptive) input.step = String(stepFor(parseFloat(input.value), Number(spec.default))); };
  if (!whole && !adaptive) input.step = "any";
  for (const [attr, key] of [["min", "minimum"], ["max", "maximum"], ["step", "step"]]) if (has(spec[key])) input[attr] = String(spec[key]);
  input.value = has(value) && value !== "auto" ? String(value) : "";
  if (has(spec.default) && input.value === "") input.placeholder = String(spec.default);
  fit();
  input.addEventListener("input", () => {
    fit();
    const n = parse(input.value);
    if (input.value === "") onDraft(undefined);
    else if (!Number.isNaN(n)) onDraft(n);
  });
  input.addEventListener("change", () => {
    if (input.value === "") return onCommit(undefined);
    const n = parse(input.value);
    if (Number.isNaN(n)) {
      input.value = "";
      note?.(`${specName(spec)}: not a ${spec.type}, value cleared.`, "warn");
      return onCommit(undefined);
    }
    input.value = String(n); // 2e-4 is kept as 0.0002
    onCommit(n);
  });
  const slider = spec.ui?.widget === "slider" && has(spec.minimum) && has(spec.maximum);
  if (!slider) return input;
  const range = make("input", "g2-slider", { type: "range", min: String(spec.minimum), max: String(spec.maximum), step: has(spec.step) ? String(spec.step) : "any" });
  range.value = input.value || String(spec.default ?? spec.minimum);
  range.addEventListener("input", () => { input.value = range.value; onDraft(parse(range.value)); });
  range.addEventListener("change", () => onCommit(parse(range.value)));
  input.addEventListener("input", () => { range.value = input.value; });
  const row = make("div", "prow");
  row.append(range, input);
  return row;
}

/** An int or float setting that also accepts "auto": a number box plus an `auto` toggle. */
function autoNumberControl(spec, ctx) {
  const row = make("div", "prow");
  const auto = make("label", "param-label");
  const box = make("input", "", { type: "checkbox" });
  box.checked = ctx.value === "auto" || (!has(ctx.value) && spec.default === "auto");
  auto.append(box, document.createTextNode(" auto"));
  const number = numberControl(spec, { ...ctx, value: box.checked ? undefined : ctx.value });
  const field = number.querySelector ? number.querySelector("input[type=number]") || number : number;
  field.disabled = box.checked;
  box.addEventListener("change", () => {
    field.disabled = box.checked;
    ctx.onCommit(box.checked ? "auto" : undefined);
  });
  row.append(number, auto);
  return row;
}

function jsonControl(spec, { value, onDraft, onCommit, note }) {
  const area = make("textarea", "cfg-input", { rows: spec.ui?.rows || 2, title: spec.doc || `${specName(spec)} (JSON)` });
  area.value = has(value) ? (typeof value === "object" ? JSON.stringify(value) : String(value)) : "";
  area.addEventListener("input", () => {
    const raw = area.value.trim();
    if (raw === "") return onDraft(undefined);
    try { onDraft(JSON.parse(raw)); } catch { /* still being typed */ }
  });
  area.addEventListener("change", () => {
    const raw = area.value.trim();
    if (raw === "") return onCommit(undefined);
    try {
      onCommit(JSON.parse(raw));
    } catch {
      note?.(`${specName(spec)}: invalid JSON -- keeping the previous value.`, "warn");
      area.value = has(value) ? JSON.stringify(value) : "";
    }
  });
  return area;
}

function textControl(spec, { value, onDraft, onCommit }) {
  const multi = spec.ui?.widget === "textarea";
  const input = multi ? make("textarea", "cfg-input", { rows: spec.ui.rows || 3 }) : make("input", "cfg-input", { type: "text" });
  input.value = has(value) ? String(value) : "";
  input.title = spec.doc || specName(spec);
  const hint = spec.ui?.placeholder || (has(spec.default) && spec.default !== "" ? String(spec.default) : "");
  if (hint) input.placeholder = hint;
  input.addEventListener("input", () => onDraft(input.value.trim() || undefined));
  input.addEventListener("change", () => onCommit(input.value.trim() || undefined));
  return input;
}

export function buildControl(spec, ctx) {
  const choices = spec.choices && spec.choices.length ? spec.choices : null;
  if (spec.picker) return pickerControl(spec, ctx);
  if (choices) return selectControl(spec, ctx, choices);
  if (spec.type === "bool") return boolControl(spec, ctx);
  if (spec.type === "int" || spec.type === "float") return spec.allow_auto ? autoNumberControl(spec, ctx) : numberControl(spec, ctx);
  if (JSON_TYPES.has(spec.type)) return jsonControl(spec, ctx);
  return textControl(spec, ctx);
}
