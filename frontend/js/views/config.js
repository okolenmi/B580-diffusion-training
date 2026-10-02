/* ---------------------------------------------------------------------------
   config.js -- Config Editor page entry (M8a).

   Two buffers over one file:
     Form  -- GET /config/options (schema, file-independent) gives the
               field list; GET /config gives values. Edits are tracked
               per field and saved as one PATCH /config deep-merge
               (only fields you touched, only while they are visible).
     Raw   -- GET/PUT /config/raw round-trips the TOML document; a
               rejected write leaves the file untouched server-side.

   The two buffers resync after either save: writing the file (raw or
   form) re-reads the other side so they can never silently disagree
   with disk.

   Everything network-shaped goes through api.js (error envelope).
   --------------------------------------------------------------------------- */

import { api, ApiError } from "../api.js";

const el = (id) => document.getElementById(id);

/* ---- system console (same convention as dashboard.js) ---- */

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
  else log(String(err && err.message ? err.message : err), "error");
}

function errText(err) {
  return err instanceof ApiError
    ? `${err.code}: ${err.message}`
    : String(err && err.message ? err.message : err);
}

/* ---- state ---- */

let schema = [];        // GET /config/options (once per boot)
let path = null;        // loaded config path (null until first Load)
let values = {};        // nested config JSON from GET /config
let raw = "";           // raw TOML buffer
let rawDirty = false;   // raw textarea edited since last read/write
const dirty = new Set();        // form field ids edited since last read/write
const rows = new Map();         // id -> {row, input, opt, group, subgroup}
const subgroupHeaders = [];     // {el, group, subgroup}
const datalists = new Map();    // file_kind -> datalist id (built lazily)

/* current values = loaded config overlaid with live widget edits;
   visible_when and save both read from this */
const live = {};

const getDeep = (obj, dotted) =>
  dotted.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);

function setDeep(obj, dotted, value) {
  const parts = dotted.split(".");
  let cur = obj;
  for (const p of parts.slice(0, -1)) {
    if (typeof cur[p] !== "object" || cur[p] === null) cur[p] = {};
    cur = cur[p];
  }
  cur[parts[parts.length - 1]] = value;
}

/* ---- field rendering ---- */

function renderForm() {
  const wrap = el("config-form");
  wrap.replaceChildren();
  rows.clear();
  subgroupHeaders.length = 0;

  // bucket by group, then stable-sort within the group by `order`
  const groups = new Map();
  schema.forEach((opt) => {
    const g = opt.group || "General";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(opt);
  });

  for (const name of [...groups.keys()].sort()) {
    const section = document.createElement("section");
    section.className = "cfg-group";
    const h = document.createElement("h2");
    h.textContent = name;
    section.appendChild(h);

    const list = groups.get(name)
      .map((opt, i) => [opt, i])
      .sort((a, b) => ((a[0].order ?? 1e9) - (b[0].order ?? 1e9)) || (a[1] - b[1]));

    let prevSub = null;
    for (const [opt] of list) {
      // persist_locally options (start_from / reset_optimizer) are the
      // per-launch choices owned by the System tracker's start form --
      // by contract they are never written to the config file, so this
      // editor (which edits the file) does not show them. It also
      // avoids their deliberate duplicate ids (one row per method).
      if (opt.persist_locally) continue;

      if (opt.subgroup && opt.subgroup !== prevSub) {
        const sh = document.createElement("h3");
        sh.className = "cfg-subgroup";
        sh.textContent = opt.subgroup;
        section.appendChild(sh);
        subgroupHeaders.push({ el: sh, group: name, subgroup: opt.subgroup });
        prevSub = opt.subgroup;
      } else if (!opt.subgroup) {
        prevSub = null;
      }

      const row = document.createElement("div");
      row.className = "cfg-field";
      row.dataset.id = opt.id;

      const label = document.createElement("label");
      label.htmlFor = widgetId(opt.id);
      label.textContent = opt.label || opt.id;

      const control = document.createElement("div");
      control.className = "cfg-control";
      const input = buildWidget(opt);
      input.id = widgetId(opt.id);
      control.appendChild(input);
      if (opt.help) {
        const help = document.createElement("p");
        help.className = "cfg-help";
        help.textContent = opt.help;
        control.appendChild(help);
      }

      row.append(label, control);
      section.appendChild(row);
      rows.set(opt.id, { row, input, opt, group: name, subgroup: opt.subgroup || null });

      input.addEventListener("input", () => onEdit(opt.id));
      input.addEventListener("change", () => onEdit(opt.id));
    }
    wrap.appendChild(section);
  }

  applyValues();
  evalVisibility();
}

const widgetId = (id) => "f-" + id.replace(/\./g, "-");

function buildWidget(opt) {
  if (opt.type === "checkbox") {
    const input = document.createElement("input");
    input.type = "checkbox";
    return input;
  }
  if (opt.type === "select") {
    const select = document.createElement("select");
    for (const c of opt.choices || []) {
      const option = document.createElement("option");
      option.value = String(c.value);
      option.textContent = c.label || String(c.value);
      select.appendChild(option);
    }
    return select;
  }
  const input = document.createElement("input");
  input.type = opt.type === "number" ? "number" : "text";
  if (input.type === "number") {
    if (opt.min !== undefined) input.min = opt.min;
    if (opt.max !== undefined) input.max = opt.max;
    input.step = opt.step ?? "any";
  } else if (opt.placeholder) {
    input.placeholder = opt.placeholder;
  }
  if (opt.file_kind) {
    attachDatalist(input, opt.file_kind);
  }
  input.autocomplete = "off";
  input.spellcheck = false;
  return input;
}

/* asset catalogs are suggestion lists only -- a missing catalog just
   means no suggestions, so failures are silent by design */
async function attachDatalist(input, kind) {
  if (!datalists.has(kind)) {
    datalists.set(kind, null);
    try {
      const catalog = await api(`/assets/${encodeURIComponent(kind)}`);
      const list = document.createElement("datalist");
      list.id = `dl-${kind}`;
      for (const opt of catalog.options || []) {
        const o = document.createElement("option");
        o.value = opt.value;
        list.appendChild(o);
      }
      document.body.appendChild(list);
      datalists.set(kind, list.id);
    } catch (err) {
      // Suggestion list is best-effort: a failure leaves the field
      // exactly as usable as it was, so this is not shown to the user --
      // but it is logged, because "no suggestions" and "the server was
      // unreachable" look identical otherwise.
      console.warn("config: could not load suggestions", err);
      return;
    }
  }
  const id = datalists.get(kind);
  if (id) input.setAttribute("list", id);
}

/* ---- values <-> widgets ---- */

function applyValues() {
  for (const [id, r] of rows) {
    const v = getDeep(values, id);
    setWidget(r, v);
    setDeep(live, id, v); // dotted id -> nested path; flat keys would be invisible to getDeep
  }
}

function setWidget(r, v) {
  const { input, opt } = r;
  if (opt.type === "checkbox") {
    input.checked = v === true;
    return;
  }
  if (opt.type === "select") {
    ensureSelectValue(input, opt, v);
    input.value = v === null || v === undefined ? "" : String(v);
    return;
  }
  input.value = v === null || v === undefined ? "" : String(v);
}

/* Keep the select truthful: a null shows as an explicit "(none)"
   choice; a value with no matching choice appears verbatim rather
   than the select silently falling back to its first option. */
function ensureSelectValue(select, opt, v) {
  const none = select.querySelector('option[data-none]');
  const wants = v === null || v === undefined;
  if (wants && !none) {
    const o = document.createElement("option");
    o.value = "";
    o.textContent = "— none —";
    o.dataset.none = "1";
    select.prepend(o);
  } else if (!wants && none) {
    none.remove();
  }
  if (!wants && ![...select.options].some((o) => o.value === String(v))) {
    const o = document.createElement("option");
    o.value = String(v);
    o.textContent = String(v);
    select.appendChild(o);
  }
}

function readWidget(r) {
  const { input, opt } = r;
  if (opt.type === "checkbox") return input.checked;
  if (opt.type === "number") {
    return input.value === "" ? null : Number(input.value);
  }
  if (opt.type === "select") return input.value === "" ? null : input.value;
  return input.value;
}

/* ---- visibility (visible_when: {dotted path: value | [values]}) ---- */

/* A requirement may be a scalar or a list (union variants, e.g.
   start_from shows unless method == "lora"). Compare via String so
   TOML numbers/bools match their JSON echo; absent deps stay hidden. */
function conditionMet(got, want) {
  const one = (w) => got === w || String(got) === String(w);
  return Array.isArray(want) ? want.some(one) : one(want);
}

function evalVisibility() {
  for (const [id, r] of rows) {
    const cond = r.opt.visible_when;
    let visible = true;
    if (cond) {
      for (const [dep, want] of Object.entries(cond)) {
        if (!conditionMet(getDeep(live, dep), want)) {
          visible = false;
          break;
        }
      }
    }
    r.row.hidden = !visible;
  }
  // a subgroup header sits above nothing when all its fields are hidden
  for (const h of subgroupHeaders) {
    h.el.hidden = ![...rows.values()].some(
      (r) => r.group === h.group && r.subgroup === h.subgroup && !r.row.hidden
    );
  }
}

/* ---- dirty tracking + toolbar ---- */

function onEdit(id) {
  const r = rows.get(id);
  if (!r) return;
  setDeep(live, id, readWidget(r));
  dirty.add(id);
  evalVisibility();
  updateToolbar();
}

function updateToolbar() {
  const n = dirty.size;
  el("btn-save-form").disabled = n === 0;
  el("save-hint").textContent =
    n === 0 ? "No unsaved changes." : `${n} field(s) edited.`;

  const chip = el("dirty-chip");
  if (rawDirty) {
    chip.textContent = "Raw edits unsaved";
    chip.hidden = false;
  } else if (n > 0) {
    chip.textContent = `${n} unsaved`;
    chip.hidden = false;
  } else {
    chip.hidden = true;
  }
}

function showError(which, message) {
  which.textContent = message;
  which.hidden = !message;
}

/* ---- load ---- */

async function loadConfig(newPath) {
  if (!newPath) {
    showError(el("form-error"), "Enter a config path (relative to the project root).");
    el("cfg-path").focus();
    return;
  }
  try {
    showError(el("form-error"), "");
    showError(el("raw-error"), "");
    const [cfg, rawRes] = await Promise.all([
      api(`/config?path=${encodeURIComponent(newPath)}`),
      api(`/config/raw?path=${encodeURIComponent(newPath)}`),
    ]);
    path = newPath;
    values = cfg;
    raw = rawRes.content;
    rawDirty = false;
    dirty.clear();
    // deep copy: `live` is the editable overlay, `values` the pristine
    // reference (revert / null round-trip read from it)
    const clone = JSON.parse(JSON.stringify(cfg));
    for (const k of Object.keys(live)) delete live[k];
    Object.assign(live, clone);

    renderForm();
    el("raw-editor").value = raw;
    el("form-state").hidden = true;
    el("raw-state").hidden = true;
    el("form-wrap").hidden = false;
    el("raw-wrap").hidden = false;
    updateToolbar();
    log(`Loaded ${newPath}.`, "info");
  } catch (err) {
    logError(err);
    showError(el("form-error"), errText(err));
  }
}

/* ---- save: form ---- */

async function saveForm() {
  if (!path || dirty.size === 0) return;
  const overrides = {};
  let n = 0;
  for (const id of dirty) {
    const r = rows.get(id);
    if (!r || r.row.hidden) continue; // hidden fields are not ours to write
    let val = readWidget(r);
    const orig = getDeep(values, id);
    if (val === "" && orig === null) val = null; // round-trip null as null
    setDeep(overrides, id, val);
    n += 1;
  }
  if (n === 0) {
    dirty.clear();
    updateToolbar();
    return;
  }
  try {
    const merged = await api("/config", {
      method: "PATCH",
      body: { path, overrides },
    });
    values = merged;
    dirty.clear();
    applyValues();
    evalVisibility();
    updateToolbar();
    showError(el("form-error"), "");
    log(`Saved ${n} field(s) to ${path}.`, "success");
    // the file on disk changed -- refresh the raw buffer (unless the
    // user is mid-edit there; their buffer wins, disk is re-read on
    // their next Reload)
    if (!rawDirty) {
      const res = await api(`/config/raw?path=${encodeURIComponent(path)}`);
      raw = res.content;
      el("raw-editor").value = raw;
    }
  } catch (err) {
    logError(err); // config_invalid -> file untouched server-side
    showError(el("form-error"), errText(err));
  }
}

async function revertForm() {
  if (!path) return;
  try {
    values = await api(`/config?path=${encodeURIComponent(path)}`);
    dirty.clear();
    applyValues();
    evalVisibility();
    updateToolbar();
    showError(el("form-error"), "");
    log("Reverted unsaved form edits.", "info");
  } catch (err) {
    logError(err);
    showError(el("form-error"), errText(err));
  }
}

/* ---- save: raw ---- */

async function saveRaw() {
  if (!path) return;
  const hadFormEdits = dirty.size > 0;
  try {
    await api("/config/raw", {
      method: "PUT",
      body: { path, content: el("raw-editor").value },
    });
    raw = el("raw-editor").value;
    rawDirty = false;
    log(`Wrote ${path}.`, "success");
    if (hadFormEdits) {
      log("Form edits were discarded by the raw write.", "warn");
      dirty.clear();
    }
    // resync the form with what is now on disk
    values = await api(`/config?path=${encodeURIComponent(path)}`);
    applyValues();
    evalVisibility();
    updateToolbar();
    showError(el("raw-error"), "");
  } catch (err) {
    logError(err); // config_invalid -> file untouched server-side
    showError(el("raw-error"), errText(err));
  }
}

async function reloadRaw() {
  if (!path) return;
  try {
    const res = await api(`/config/raw?path=${encodeURIComponent(path)}`);
    raw = res.content;
    el("raw-editor").value = raw;
    rawDirty = false;
    updateToolbar();
    showError(el("raw-error"), "");
    log("Reloaded raw from disk.", "info");
  } catch (err) {
    logError(err);
    showError(el("raw-error"), errText(err));
  }
}

/* ---- tabs ---- */

function showTab(which) {
  const form = which === "form";
  el("tab-form").classList.toggle("active", form);
  el("tab-raw").classList.toggle("active", !form);
  el("tab-form").setAttribute("aria-selected", String(form));
  el("tab-raw").setAttribute("aria-selected", String(!form));
  el("panel-form").hidden = !form;
  el("panel-raw").hidden = form;
}

/* ---- boot ---- */

async function boot() {
  el("btn-load").addEventListener("click", () => loadConfig(el("cfg-path").value.trim()));
  el("cfg-path").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") loadConfig(el("cfg-path").value.trim());
  });
  el("tab-form").addEventListener("click", () => showTab("form"));
  el("tab-raw").addEventListener("click", () => showTab("raw"));
  el("btn-save-form").addEventListener("click", saveForm);
  el("btn-revert").addEventListener("click", revertForm);
  el("btn-save-raw").addEventListener("click", saveRaw);
  el("btn-reload-raw").addEventListener("click", reloadRaw);
  el("raw-editor").addEventListener("input", () => {
    rawDirty = true;
    updateToolbar();
  });

  // Schema is file-independent: fetch once, render nothing until a
  // config is loaded (an empty form with no values would lie).
  try {
    const res = await api("/config/options");
    schema = res.options || [];
  } catch (err) {
    logError(err);
    showError(el("form-error"), errText(err));
    return;
  }

  // Prefill + auto-load the default config when there is one.
  let auto = "";
  try {
    const settings = await api("/settings");
    auto = (settings.stored || {}).default_config || "";
  } catch (err) {
    logError(err);
  }
  if (auto) el("cfg-path").value = auto;
  if (auto) await loadConfig(auto);
}

boot().catch(logError);
