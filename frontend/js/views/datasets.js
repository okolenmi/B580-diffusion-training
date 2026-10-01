/* ---------------------------------------------------------------------------
   datasets.js -- Datasets page entry (M8c, extended in M8e/M8f).

   Two views on one page, routed by the URL:
     /datasets         -- card list (each card fronts itself with the
                          dataset's resolved preview image), create/
                          delete, "Add data" per card
     /datasets/{name}  -- detail: stats chips + Items | Sets | Tasks tabs

   Items are the curation surface with three interactions:
     * Browse mode -- filter, select, quick toggle/discard;
     * Edit mode   -- click a card to open the advanced item editor
       (prompt / negative / cfg / verdict, metadata, prev-next walk);
     * Multi-edit  -- any selection raises the bulk panel: one field
       (prompt / negative / cfg / verdict) appended, prepended or
       replaced across every selected row.

   Every item thumb carries a half-transparent ⋮ (M8f) opening the
   item context menu -- currently one option, "Set as dataset preview"
   (disabled when the item has no image or already fronts the card).

   "Add data" (cards, detail topbar, Tasks tab) opens one dialog with
   two generators: sample new images from a checkpoint (teacher) or
   import a folder of images -- both run as dataset tasks (one at a
   time per dataset, 409 dataset_task_active) polled every 4s while
   the detail view is open (dataset tasks publish no events).

   All data paths go through api.js (error envelope decoded once).
   --------------------------------------------------------------------------- */

import { api, ApiError } from "../api.js";

const el = (id) => document.getElementById(id);

/* tiny DOM builder: attrs {class, text, onclick, ...}; no innerHTML
   with server/user data anywhere (prompts are arbitrary text) */
function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v === true ? "" : v);
  }
  for (const child of children.flat(3)) if (child != null) node.append(child);
  return node;
}

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

let datasetName = null;   // detail view: the open dataset (null = list)
let detail = null;        // DatasetDetailOut
let items = [];           // current filter's rows
let itemFilter = "all";   // all | pending | used
let itemMode = "browse";  // browse | edit (toolbar toggle)
let selected = new Set(); // item ids marked in the grid
let tasks = [];           // all task rows for the dataset
let pollTimer = null;

// multi-edit panel
let bulkField = "prompt"; // prompt | neg_prompt | cfg | type
let bulkType = "";        // chosen verdict in the type row ("" = unset)

// add-data dialog
let addTarget = null;     // dataset the dialog will launch into
let addTab = "generate";  // generate | import (last used, session memory)

// item editor
let editorIndex = -1;     // index into `items` of the row being edited
let editorSnapshot = null;// field values at load/save time (dirty base)
let editorType = "";      // verdict picked in the editor

// item context menu (M8f)
let menuAnchorItem = null; // item whose ⋮ menu is open (null = closed)

const TASK_KIND_LABEL = {
  ingest_lora: "import images",
  generate_teacher: "generate",
};

const TASK_STATUS_CLASS = {
  pending: "status-idle",
  running: "status-running",
  finished: "status-completed",
  failed: "status-failed",
  killed: "status-cancelled",
};

const dsApi = (tail) =>
  `/datasets/${encodeURIComponent(datasetName)}${tail}`;

/* ---- shared formatting ---- */

function fmtBytes(n) {
  if (!Number.isFinite(n)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v >= 10 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}

function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  const s = Math.round((Date.now() - d.getTime()) / 1000);
  const rel = s < 60 ? `${s}s ago`
    : s < 3600 ? `${Math.floor(s / 60)}m ago`
    : s < 86400 ? `${Math.floor(s / 3600)}h ago`
    : `${Math.floor(s / 86400)}d ago`;
  return `${d.toLocaleString()} (${rel})`;
}

const fileUrl = (name, path) =>
  `/api/v1/datasets/${encodeURIComponent(name)}/files/` +
  path.split("/").map(encodeURIComponent).join("/");

const previewUrl = (previewPath) => fileUrl(datasetName, previewPath);

/* ---- view switching ---- */

function showView(which) {
  closeItemMenu(); // the menu is fixed-position: never outlive its view
  const list = which === "list";
  el("view-list").hidden = !list;
  el("view-detail").hidden = list;
  el("topbar-actions").hidden = !list;
  el("btn-add-data").hidden = list;
  el("btn-back").hidden = list;
}

function setTopbar(title, sub) {
  el("ds-title").textContent = title;
  el("ds-sub").textContent = sub;
}

function showState(id, message) {
  el(id).textContent = message;
  el(id).hidden = !message;
}

function showError(id, message) {
  el(id).textContent = message || "";
  el(id).hidden = !message;
}

function showTab(which) {
  closeItemMenu(); // anchors live in the items grid
  for (const name of ["items", "sets", "tasks"]) {
    el(`tab-${name}`).classList.toggle("active", name === which);
    el(`tab-${name}`).setAttribute("aria-selected", String(name === which));
    el(`panel-${name}`).hidden = name !== which;
  }
}

/* ---- list view ---- */

async function showList() {
  datasetName = null;
  detail = null;
  stopPoll();
  showView("list");
  setTopbar("Datasets", "Curate items, commit training sets, run cache sweeps");
  await renderList();
}

async function renderList() {
  showState("list-state", "Loading…");
  el("ds-grid").hidden = true;
  let res;
  try {
    res = await api("/datasets");
  } catch (err) {
    logError(err);
    showState("list-state", errText(err));
    return;
  }
  const grid = el("ds-grid");
  grid.replaceChildren();
  if (!res.datasets.length) {
    showState("list-state", "No datasets yet -- create one above.");
    return;
  }
  showState("list-state", "");
  for (const entry of res.datasets) {
    grid.appendChild(datasetCard(entry));
  }
  grid.hidden = false;
}

function datasetCard(entry) {
  const { info, stats } = entry;
  const legacy = info.format_version < 2;
  const meta = stats
    ? `${stats.items} items · ${stats.pending} pending · ${stats.committed} used · ` +
      `${stats.sets} sets · ${fmtBytes(stats.bytes)}`
    : "legacy format -- migrate to see stats";

  const nameLink = h("a", {
    class: "ds-card-name",
    href: `/datasets/${encodeURIComponent(info.name)}`,
    text: info.name,
  });
  const del = h("button", {
    class: "btn btn-danger btn-small",
    text: "Delete",
    onclick: (ev) => { ev.stopPropagation(); deleteDataset(info.name); },
    title: `Delete dataset '${info.name}' and all of its files`,
  });
  // v1 datasets refuse tasks (format gate in the builder) -- no Add data
  const add = legacy ? null : h("button", {
    class: "btn btn-start btn-small",
    text: "＋ Add data",
    onclick: (ev) => { ev.stopPropagation(); openAddDialog(info.name); },
    title: "Generate images or import a folder into this dataset",
  });

  // resolved card image (M8f): stored override or first-item fallback
  // computed server-side; absent -> honest NO PREVIEW, never a guess
  const thumb = h("div", { class: "ds-card-thumb" });
  if (entry.preview_path) {
    const img = h("img", {
      src: fileUrl(info.name, entry.preview_path),
      alt: `preview of dataset ${info.name}`,
      loading: "lazy",
    });
    const ph = h("div", { class: "no-preview", text: "NO PREVIEW", hidden: true });
    img.addEventListener("error", () => { img.hidden = true; ph.hidden = false; });
    thumb.append(img, ph);
  } else {
    thumb.append(h("div", { class: "no-preview", text: "NO PREVIEW" }));
  }

  const card = h("div", { class: "ds-card", onclick: (ev) => {
    // real link for normal/middle clicks; this handler covers the card body
    if (ev.target.closest("button, a")) return;
    location.href = nameLink.href;
  }},
    thumb,
    h("div", { class: "ds-card-head" }, nameLink,
      legacy ? h("span", { class: "ds-legacy", text: "legacy v1" }) : null),
    h("div", { class: "ds-card-desc", text: info.description || "" }),
    h("div", { class: "ds-card-meta", text: meta }),
    h("div", { class: "ds-card-meta", text: `created ${fmtTime(info.created_at)}` }),
    h("div", { class: "ds-card-foot" }, add, del),
  );
  return card;
}

async function deleteDataset(name) {
  if (!window.confirm(`Delete dataset '${name}'?\n\nThis removes its rows, sets and files permanently.`)) return;
  try {
    await api(`/datasets/${encodeURIComponent(name)}`, { method: "DELETE" });
    log(`Deleted dataset '${name}'.`, "success");
    await renderList();
  } catch (err) {
    logError(err); // 409 dataset_task_active lands here
    showError("create-error", errText(err));
    el("create-card").hidden = false; // surface it somewhere visible
  }
}

async function createDataset() {
  const name = el("new-name").value.trim();
  showError("create-error", "");
  if (!name) {
    showError("create-error", "Name is required.");
    el("new-name").focus();
    return;
  }
  try {
    await api("/datasets", {
      method: "POST",
      body: { name, description: el("new-desc").value.trim() || null },
    });
    log(`Created dataset '${name}'.`, "success");
    el("new-name").value = "";
    el("new-desc").value = "";
    el("create-card").hidden = true;
    await renderList();
  } catch (err) {
    logError(err);
    showError("create-error", errText(err));
  }
}

/* ---- detail view ---- */

async function openDetail(name) {
  datasetName = name;
  showView("detail");
  setTopbar(name, "");
  showState("detail-state", "Loading…");
  el("detail-wrap").hidden = true;
  stopPoll();
  selected = new Set();     // ids collide across datasets -- never carry over
  editorIndex = -1;
  el("item-dialog").close(); // in case a walk outlived the navigation

  try {
    detail = await api(dsApi(""));
  } catch (err) {
    logError(err);
    const hint = err instanceof ApiError && err.code === "dataset_not_migrated"
      ? ` -- ${err.message}` : "";
    showState("detail-state", `Dataset '${name}' could not be opened${hint}`);
    return;
  }

  setTopbar(detail.info.name, detail.info.description || "");
  renderStats();
  showState("detail-state", "");
  el("detail-wrap").hidden = false;

  try {
    await Promise.all([loadItems(), loadTasks()]);
  } catch (err) {
    logError(err);
  }
  renderItems();
  renderSets();
  renderTasks();
  startPoll();
  log(`Opened dataset '${name}'.`, "info");
}

function renderStats() {
  const box = el("ds-stats");
  box.replaceChildren();
  const s = detail.stats;
  if (!s) return; // never fabricate zeros for stats we do not have
  const chips = [
    ["items", String(s.items)],
    ["pending", String(s.pending)],
    ["used", String(s.committed)],
    ["bad", String(s.bad)],
    ["sets", String(s.sets)],
    ["shards", String(s.shards)],
    ["size", fmtBytes(s.bytes)],
  ];
  for (const [label, value] of chips) {
    box.appendChild(h("div", { class: "ds-stat" },
      h("b", { text: value }), h("span", { text: label })));
  }
}

async function reloadDetail() {
  try {
    detail = await api(dsApi(""));
    renderStats();
    renderSets();
  } catch (err) {
    logError(err);
  }
}

/* ---- items ---- */

async function loadItems() {
  const q = itemFilter === "pending" ? "?committed=false"
    : itemFilter === "used" ? "?committed=true" : "";
  const res = await api(dsApi(`/items${q}`));
  items = res.items;
  selected = new Set([...selected].filter((id) =>
    items.some((it) => it.id === id))); // drop rows that left the filter
}

function renderItems() {
  el("items-count").textContent =
    `${items.length} shown · ${selected.size} selected` +
    (itemMode === "edit" ? " · edit mode" : "");
  const all = items.length > 0 && selected.size === items.length;
  el("select-all").checked = all;
  const grid = el("items-grid");
  grid.replaceChildren();
  grid.classList.toggle("edit-mode", itemMode === "edit");

  if (!items.length) {
    showState("items-state",
      itemFilter === "pending"
        ? "Nothing awaiting review -- every item is already used or bad."
        : itemFilter === "used"
          ? "No items are committed to training yet."
          : "This dataset has no items yet. Use Add data (Tasks tab) " +
            "to generate or import images.");
  } else {
    showState("items-state", "");
  }
  renderBulkBar();
  for (const item of items) grid.appendChild(itemCard(item));
}

function itemCard(item) {
  const check = h("input", {
    type: "checkbox",
    class: "ds-item-check",
    checked: selected.has(item.id),
    "aria-label": `select item ${item.id}`,
    onchange: () => {
      if (check.checked) selected.add(item.id);
      else selected.delete(item.id);
      card.classList.toggle("selected", check.checked);
      renderItems();
    },
  });

  const thumbKids = [check];
  if (item.preview_path) {
    const img = h("img", {
      src: previewUrl(item.preview_path),
      alt: `preview of item ${item.id}`,
      loading: "lazy",
    });
    const placeholder = h("div", { class: "no-preview", text: "NO PREVIEW", hidden: true });
    img.addEventListener("error", () => { img.hidden = true; placeholder.hidden = false; });
    thumbKids.push(img, placeholder);
  } else {
    thumbKids.push(h("div", { class: "no-preview", text: "NO PREVIEW" }));
  }
  // half-transparent ⋮ (M8f): opens the item context menu
  thumbKids.push(h("button", {
    class: "ds-item-menu",
    type: "button",
    text: "⋮",
    title: "Item options",
    "aria-label": `Options for item ${item.id}`,
    "aria-haspopup": "menu",
    onclick: (ev) => { ev.stopPropagation(); toggleItemMenu(item, ev.currentTarget); },
  }));

  const promptBox = h("div", {
    class: `item-prompt${item.prompt ? "" : " empty"}`,
    text: item.prompt || "(no prompt -- click to edit)",
    title: "Click to open the item editor",
    onclick: (ev) => { ev.stopPropagation(); openItemEditor(item); },
  });

  const metaKids = [
    h("span", { text: `#${item.id}` }),
    h("span", { class: `item-type ${item.type}`, text: item.type }),
    h("span", { text: `${item.latent_h}×${item.latent_w}` }),
  ];
  if (item.seed !== null && item.seed !== undefined) {
    metaKids.push(h("span", { text: `seed ${item.seed}` }));
  }
  if (item.cfg !== null && item.cfg !== undefined) {
    metaKids.push(h("span", { text: `cfg ${item.cfg}` }));
  }
  if (item.neg_prompt) metaKids.push(h("span", { text: "neg" }));

  const card = h("div", {
      class: `ds-item${selected.has(item.id) ? " selected" : ""}`,
      "data-item-id": item.id,
      onclick: (ev) => {
        // Edit mode: the whole card body opens the editor. Browse mode
        // leaves clicks inert (the prompt box handles itself above) so
        // rubber-band selection never triggers an edit.
        if (itemMode !== "edit") return;
        if (ev.target.closest("button, input, a")) return;
        openItemEditor(item);
      },
    },
    h("div", { class: "ds-thumb" }, thumbKids,
      itemMode === "edit"
        ? h("span", { class: "item-edit-hint", text: "edit" })
        : null),
    h("div", { class: "ds-item-body" },
      promptBox,
      h("div", { class: "item-meta" }, metaKids),
      h("div", { class: "ds-item-actions" },
        h("button", {
          class: "btn btn-secondary btn-small",
          text: item.type === "bad" ? "mark good" : "mark bad",
          title: "Toggle the review verdict (bad items are skipped)",
          onclick: () => toggleType(item),
        }),
        h("button", {
          class: "btn btn-danger btn-small",
          text: "discard",
          title: "Delete this item and its preview",
          onclick: () => discardItems([item.id]),
        }),
      ),
    ),
  );
  return card;
}

/* ---- item context menu (M8f): set the dataset's card preview ---- */

function closeItemMenu() {
  el("item-menu").hidden = true;
  menuAnchorItem = null;
}

function openItemMenu(item, anchor) {
  const menu = el("item-menu");
  const opt = el("item-menu-preview");
  const current = Boolean(
    detail && item.preview_path && detail.preview_path === item.preview_path
  );
  menuAnchorItem = item;
  // one option, honestly disabled: nothing to show, or already showing
  opt.disabled = !item.preview_path || current;
  opt.title = !item.preview_path
    ? "this item has no preview image"
    : current ? "already the dataset preview" : "";
  menu.hidden = false;
  // fixed placement under the ⋮, flipped up / clamped near viewport edges
  const r = anchor.getBoundingClientRect();
  const w = menu.offsetWidth;
  const hgt = menu.offsetHeight;
  const left = Math.max(8, Math.min(r.right - w, window.innerWidth - w - 8));
  const top = r.bottom + 4 + hgt > window.innerHeight - 8
    ? Math.max(8, r.top - hgt - 4)
    : r.bottom + 4;
  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;
}

function toggleItemMenu(item, anchor) {
  const open = !el("item-menu").hidden && menuAnchorItem
    && menuAnchorItem.id === item.id;
  if (open) closeItemMenu();
  else openItemMenu(item, anchor);
}

async function setDatasetPreview() {
  const item = menuAnchorItem;
  closeItemMenu();
  if (!item || !detail) return;
  try {
    const res = await api(dsApi("/preview"), {
      method: "PUT",
      body: { item_id: item.id },
    });
    detail.preview_path = res.preview_path; // disables the option on this item
    log(`Dataset preview set to item #${item.id}.`, "success");
  } catch (err) {
    logError(err);
  }
}

/* ---- item editor (M8e) --------------------------------------------
   One advanced editor for every per-item edit: prompt, negative,
   cfg, verdict, read-only metadata, prev/next walk over the current
   filter. Save sends only what changed; Revert restores the snapshot.
   CFG cannot be cleared through the API (null = untouched), so the
   field is hint-labeled "empty = keep current". */

function setItemMode(mode) {
  itemMode = mode;
  for (const b of el("item-mode").children) {
    b.classList.toggle("active", b.dataset.mode === mode);
  }
  renderItems();
}

function editorItem() {
  return editorIndex >= 0 && editorIndex < items.length
    ? items[editorIndex] : null;
}

function editorValues() {
  return {
    prompt: el("item-ed-prompt").value,
    neg_prompt: el("item-ed-neg").value,
    cfg: el("item-ed-cfg").value.trim(),
    type: editorType,
  };
}

function editorDirty() {
  if (!editorSnapshot) return false;
  const v = editorValues();
  const s = editorSnapshot;
  const cfgSame = v.cfg === ""
    ? true                       // empty = keep current (API cannot clear)
    : (s.cfg === null || s.cfg === undefined
        ? false : Number(v.cfg) === s.cfg);
  return v.prompt !== s.prompt
    || v.neg_prompt !== s.neg_prompt
    || v.type !== s.type
    || !cfgSame;
}

function syncEditorDirty() {
  el("item-ed-save").disabled = !editorDirty();
}

function renderEditorType() {
  for (const b of el("item-ed-type").children) {
    b.classList.toggle("active", b.dataset.etype === editorType);
  }
}

function renderEditorMeta(item) {
  const meta = el("item-ed-meta");
  meta.replaceChildren();
  const rows = [
    ["item", `#${item.id}`],
    ["source", `#${item.source_id}`],
    ["shard", `#${item.shard_id}`],
    ["size", `${item.latent_h}×${item.latent_w}`],
    ["model", item.model_type],
    ["seed", item.seed === null || item.seed === undefined ? "—" : String(item.seed)],
    ["cfg", item.cfg === null || item.cfg === undefined ? "—" : String(item.cfg)],
    ["membership", item.committed ? "in training" : "pending"],
    ["origin", item.source_path || "—"],
  ];
  for (const [label, value] of rows) {
    meta.appendChild(h("span", {},
      h("b", { text: `${label} ` }), value));
  }
}

function renderEditorMedia(item) {
  const box = el("item-ed-media");
  box.replaceChildren();
  if (!item.preview_path) {
    box.appendChild(h("div", { class: "no-preview", text: "NO PREVIEW" }));
    return;
  }
  const img = h("img", {
    src: previewUrl(item.preview_path),
    alt: `preview of item ${item.id}`,
  });
  img.addEventListener("error", () =>
    box.replaceChildren(h("div", { class: "no-preview", text: "NO PREVIEW" })));
  box.appendChild(img);
}

function fillEditor(item) {
  const idx = items.findIndex((it) => it.id === item.id);
  if (idx < 0) return; // row left the filter under us
  editorIndex = idx;
  editorType = item.type;
  editorSnapshot = {
    prompt: item.prompt,
    neg_prompt: item.neg_prompt,
    cfg: item.cfg === null || item.cfg === undefined ? null : item.cfg,
    type: item.type,
  };
  el("item-ed-id").textContent = `#${item.id}`;
  el("item-ed-pos").textContent = `${idx + 1} of ${items.length} shown`;
  el("item-ed-prev").disabled = idx === 0;
  el("item-ed-next").disabled = idx === items.length - 1;
  el("item-ed-prompt").value = item.prompt;
  el("item-ed-neg").value = item.neg_prompt || "";
  el("item-ed-cfg").value =
    item.cfg === null || item.cfg === undefined ? "" : String(item.cfg);
  renderEditorType();
  renderEditorMeta(item);
  renderEditorMedia(item);
  showError("item-ed-error", "");
  syncEditorDirty();
}

function openItemEditor(item) {
  fillEditor(item);
  const dlg = el("item-dialog");
  if (!dlg.open) dlg.showModal();
}

function closeItemEditor() {
  const dlg = el("item-dialog");
  if (dlg.open) dlg.close();
  editorIndex = -1;
  editorSnapshot = null;
}

function editorNav(delta) {
  const item = editorItem();
  if (!item) return;
  const next = items[editorIndex + delta];
  if (!next) return;
  if (editorDirty()
      && !window.confirm("Discard unsaved changes and move on?")) return;
  fillEditor(next);
}

function revertEditor() {
  const item = editorItem();
  if (!item) return;
  fillEditor(item);
}

async function saveEditor() {
  const item = editorItem();
  if (!item || !editorSnapshot) return;
  const v = editorValues();
  const s = editorSnapshot;
  const body = {};
  if (v.prompt !== s.prompt) body.prompt = v.prompt;
  if (v.neg_prompt !== s.neg_prompt) body.neg_prompt = v.neg_prompt;
  if (v.cfg !== "" && v.cfg !== String(s.cfg ?? "")) body.cfg = Number(v.cfg);
  if (v.type !== s.type) body.type = v.type;
  if (!Object.keys(body).length) return; // pristine -- button is disabled anyway
  try {
    const updated = await api(dsApi(`/items/${item.id}`), {
      method: "PATCH", body,
    });
    Object.assign(item, updated);
    log(`Item ${item.id} saved (${Object.keys(body).join(", ")}).`, "success");
    await loadItems();
    // keep the editor on the same row if it survived the filter
    const still = items.findIndex((it) => it.id === item.id);
    if (still < 0) closeItemEditor();
    else fillEditor(items[still]);
    renderItems();
    if (body.type !== undefined) reloadDetail(); // bad count lives in stats
  } catch (err) {
    logError(err);
    showError("item-ed-error", errText(err));
  }
}

async function toggleType(item) {
  const next = item.type === "bad" ? "good" : "bad";
  try {
    const updated = await api(dsApi(`/items/${item.id}`), {
      method: "PATCH", body: { type: next },
    });
    Object.assign(item, updated);
    log(`Item ${item.id} marked ${next}.`, "success");
    renderItems();
    reloadDetail(); // bad count lives in stats
  } catch (err) {
    logError(err);
    showError("items-error", errText(err));
  }
}

async function discardItems(ids) {
  const what = ids.length === 1 ? `item ${ids[0]}` : `${ids.length} items`;
  if (!window.confirm(`Discard ${what}? Rows and preview files are deleted permanently.`)) return;
  const ed = editorItem();
  if (ed && ids.includes(ed.id)) closeItemEditor();
  try {
    const res = await api(dsApi("/items/discard"), {
      method: "POST", body: { item_ids: ids },
    });
    log(`Discarded ${res.deleted} item(s).`, "success");
    ids.forEach((id) => selected.delete(id));
    await loadItems();
    renderItems();
    reloadDetail();
  } catch (err) {
    logError(err);
    showError("items-error", errText(err));
  }
}

/* ---- multi-edit (M8e) ----------------------------------------------
   The panel appears with any selection: pick a field, pick a mode
   (text: replace/prepend/append), apply to every selected row. CFG
   and verdict replace outright; empty values are refused here rather
   than silently no-op'ing server-side. */

function renderBulkBar() {
  const bar = el("bulk-bar");
  bar.hidden = selected.size === 0;
  el("bulk-count").textContent = `${selected.size} selected`;
  el("bulk-apply-n").textContent = String(selected.size);
}

function setBulkField(field) {
  bulkField = field;
  for (const b of el("bulk-field").children) {
    b.classList.toggle("active", b.dataset.field === field);
  }
  el("bulk-row-text").hidden = !(field === "prompt" || field === "neg_prompt");
  el("bulk-row-cfg").hidden = field !== "cfg";
  el("bulk-row-type").hidden = field !== "type";
}

function setBulkType(value) {
  bulkType = value;
  for (const b of el("bulk-type").children) {
    b.classList.toggle("active", b.dataset.btype === value);
  }
}

async function applyBulk() {
  if (!selected.size) return;
  showError("items-error", "");
  const ids = [...selected];
  let body;
  if (bulkField === "prompt" || bulkField === "neg_prompt") {
    const value = el("bulk-text").value;
    if (!value) {
      showError("items-error", "Enter a value to apply to the selection.");
      el("bulk-text").focus();
      return;
    }
    const mode = el("bulk-text-mode").value;
    body = bulkField === "prompt"
      ? { item_ids: ids, prompt: value, prompt_mode: mode }
      : { item_ids: ids, neg_prompt: value, neg_prompt_mode: mode };
  } else if (bulkField === "cfg") {
    const raw = el("bulk-cfg").value.trim();
    if (raw === "") {
      showError("items-error", "Enter a CFG value to apply.");
      el("bulk-cfg").focus();
      return;
    }
    body = { item_ids: ids, cfg: Number(raw) };
  } else {
    if (!bulkType) {
      showError("items-error", "Pick good or bad first.");
      return;
    }
    body = { item_ids: ids, type: bulkType };
  }
  try {
    const res = await api(dsApi("/items"), { method: "PATCH", body });
    log(`${bulkField} applied to ${res.updated} item(s).`, "success");
    if (bulkField === "prompt" || bulkField === "neg_prompt") {
      el("bulk-text").value = "";
    } else if (bulkField === "cfg") {
      el("bulk-cfg").value = "";
    } else {
      setBulkType("");
    }
    await loadItems();
    renderItems();
    if (bulkField === "type") reloadDetail(); // bad count lives in stats
  } catch (err) {
    logError(err); // 422 invalid_query: mode/type/value guards
    showError("items-error", errText(err));
  }
}

async function commitToSet() {
  if (!selected.size) return;
  const name = el("set-name").value.trim();
  showError("items-error", "");
  if (!name) {
    showError("items-error", "Set name is required to commit.");
    el("set-name").focus();
    return;
  }
  try {
    const res = await api(dsApi("/sets"), {
      method: "POST",
      body: { item_ids: [...selected], name },
    });
    log(`Committed ${res.added} item(s) to set '${res.set_name}'.`, "success");
    el("set-name").value = "";
    selected.clear();
    renderItems();
    await reloadDetail();
    showTab("sets");
  } catch (err) {
    logError(err);
    showError("items-error", errText(err));
  }
}

/* ---- sets ---- */

function renderSets() {
  const list = el("sets-list");
  list.replaceChildren();
  const sets = detail ? detail.sets : [];
  if (!sets.length) {
    showState("sets-state",
      "No training sets yet -- select items on the Items tab and commit them.");
    return;
  }
  showState("sets-state", "");
  for (const set of sets) {
    list.appendChild(h("div", { class: "ds-set" },
      h("b", { text: set.name }),
      set.description ? h("span", { class: "text-dim", text: set.description }) : null,
      h("span", { class: "ds-card-meta", text: `${set.members} members` }),
      h("span", { class: "ds-card-meta", text: `created ${fmtTime(set.created_at)}` }),
    ));
  }
}

/* ---- tasks ---- */

async function loadTasks() {
  const res = await api(dsApi("/tasks"));
  tasks = res.tasks;
}

function renderTasks() {
  const list = el("tasks-list");
  list.replaceChildren();
  if (!tasks.length) {
    showState("tasks-state",
      "No dataset tasks yet -- use Add data to generate or import images.");
    return;
  }
  showState("tasks-state", "");
  for (const task of tasks) {
    const active = task.status === "pending" || task.status === "running";
    const pct = task.total > 0
      ? Math.min(100, Math.round((task.current / task.total) * 100)) : 0;
    list.appendChild(h("div", { class: "ds-task" },
      h("span", { class: "ds-task-id", text: `#${task.id}` }),
      h("span", {
        class: "ds-task-kind",
        text: TASK_KIND_LABEL[task.kind] || task.kind,
        title: task.kind,
      }),
      h("span", {
        class: `status-badge ${TASK_STATUS_CLASS[task.status] || "status-idle"}`,
        text: task.status,
      }),
      h("div", { class: "task-progress" },
        h("div", { class: "task-bar" },
          h("i", { style: `width: ${pct}%` })),
        h("span", { text: task.total > 0 ? `${task.current} / ${task.total}` : "…" }),
      ),
      h("span", { class: "ds-task-when", text: fmtTime(task.created_at) }),
      active
        ? h("button", {
            class: "btn btn-danger btn-small", text: "Stop",
            onclick: () => stopTask(task.id),
          })
        : null,
      task.error ? h("div", { class: "ds-task-error", text: task.error }) : null,
    ));
  }
}

async function stopTask(id) {
  try {
    await api(dsApi(`/tasks/${id}/stop`), { method: "POST" });
    log(`Sent stop to task #${id}.`, "success");
    await loadTasks();
    renderTasks();
  } catch (err) {
    logError(err); // 409 dataset_task_not_active: it finished first
  }
}

/* ---- add data dialog (M8e) ------------------------------------------
   One dialog, two generators, one launch endpoint. The client checks
   only what it can see fast (required fields, min <= max, counts);
   the server's use case is the authority and its envelope error lands
   in #add-error. */

const RESIZE_DESC = {
  fit: "Keep aspect ratio; wide/tall images split into crops so no " +
       "crop exceeds max aspect ratio (one item per crop).",
  center_crop: "Scale to cover the square, then cut the edges -- " +
               "always exactly latent_size², some content lost.",
  pad: "Keep aspect ratio, scale to fit, fill the edges with black " +
       "-- nothing lost, padded borders are real pixels.",
  resize: "Force the exact size -- stretches, aspect ratio not kept.",
};

function openAddDialog(name, tab) {
  addTarget = name;
  el("add-ds-name").textContent = name;
  showError("add-error", "");
  setAddTab(tab || addTab);
  updateAddHints();
  const dlg = el("add-dialog");
  if (!dlg.open) dlg.showModal();
  el("add-model").focus();
}

function setAddTab(tab) {
  addTab = tab;
  const gen = tab === "generate";
  el("add-tab-generate").classList.toggle("active", gen);
  el("add-tab-import").classList.toggle("active", !gen);
  el("add-tab-generate").setAttribute("aria-selected", String(gen));
  el("add-tab-import").setAttribute("aria-selected", String(!gen));
  el("add-panel-generate").hidden = !gen;
  el("add-panel-import").hidden = gen;
  el("add-model-hint").textContent = gen
    ? "Teacher checkpoint that samples the images."
    : "Checkpoint whose VAE encodes the imported images.";
  el("add-hint").textContent = gen
    ? "One task at a time; watch progress on the Tasks tab."
    : "Images are counted when the task starts; .txt sidecars become prompts.";
  updateAddHints();
}

function setAddPromptMode(mode) {
  for (const b of el("add-prompt-mode").children) {
    b.classList.toggle("active", b.dataset.pmode === mode);
  }
  el("add-prompts-list").hidden = mode !== "list";
  el("add-prompts-keywords").hidden = mode !== "keywords";
}

function setAddNegMode(mode) {
  for (const b of el("add-neg-mode").children) {
    b.classList.toggle("active", b.dataset.nmode === mode);
  }
  el("add-neg-list").hidden = mode !== "list";
  el("add-neg-keywords").hidden = mode !== "keywords";
}

function pxHint(inputId) {
  const v = Number(el(inputId).value || 0);
  return v >= 8 ? `→ ${v * 8}×${v * 8}px` : "";
}

function updateAddHints() {
  el("add-latent-px").textContent = pxHint("add-latent");
  el("add-import-latent-px").textContent = pxHint("add-import-latent");
  const mode = el("add-resize-mode").value;
  el("add-resize-desc").textContent = RESIZE_DESC[mode] || "";
  // the split knob only exists for the mode that splits
  el("add-max-aspect-row").hidden = mode !== "fit";
  if (addTab === "generate") {
    const n = Number(el("add-conditions").value || 0);
    const m = Number(el("add-samples").value || 0);
    el("add-total").textContent =
      n >= 1 && m >= 1 ? `→ ${n * m} images` : "";
  } else {
    el("add-total").textContent = "";
  }
}

function activeSegData(groupId, key) {
  const seg = el(groupId).querySelector(".seg.active");
  return seg ? seg.dataset[key] : null;
}

function addFieldError(message, inputId) {
  showError("add-error", message);
  if (inputId) el(inputId).focus();
}

function startAddTask() {
  const name = addTarget;
  if (!name) return;
  showError("add-error", "");
  const model = el("add-model").value.trim();
  if (!model) return addFieldError("Checkpoint is required.", "add-model");

  let body;
  if (addTab === "generate") {
    const promptMode = activeSegData("add-prompt-mode", "pmode") || "list";
    const prompts = el("add-prompts").value;
    const keywords = el("add-keywords").value;
    if (promptMode === "list" && !prompts.split("\n").some((l) => l.trim())) {
      return addFieldError("Prompt list needs at least one non-empty line.",
        "add-prompts");
    }
    if (promptMode === "keywords" && !keywords.split("\n").some((l) => l.trim())
        && !el("add-keywords-file").value.trim()) {
      return addFieldError(
        "Keyword mix needs keywords or a word-list file.", "add-keywords");
    }
    const cfgMin = Number(el("add-cfg-min").value);
    const cfgMax = Number(el("add-cfg-max").value);
    if (cfgMin > cfgMax) return addFieldError("CFG min must be <= max.");
    const stMin = Number(el("add-steps-min").value);
    const stMax = Number(el("add-steps-max").value);
    if (stMin > stMax) return addFieldError("Steps min must be <= max.");
    const tLow = Number(el("add-t-low").value);
    const tHigh = Number(el("add-t-high").value);
    if (tLow > tHigh) return addFieldError("T low must be <= high.");
    const negMode = activeSegData("add-neg-mode", "nmode") || "list";
    body = {
      kind: "generate_teacher",
      model,
      prompt_mode: promptMode,
      prompts,
      keywords,
      keywords_file: el("add-keywords-file").value.trim(),
      template: el("add-template").value,
      min_keywords: Number(el("add-min-kw").value),
      max_keywords: Number(el("add-max-kw").value),
      neg_mode: negMode,
      negative_prompt: el("add-negative").value,
      neg_keywords: el("add-neg-keywords-ta").value,
      neg_keywords_file: el("add-neg-keywords-file").value.trim(),
      neg_template: el("add-neg-template").value,
      neg_min_keywords: Number(el("add-neg-min-kw").value),
      neg_max_keywords: Number(el("add-neg-max-kw").value),
      cfg_min: cfgMin,
      cfg_max: cfgMax,
      steps_min: stMin,
      steps_max: stMax,
      t_mode: el("add-t-mode").value,
      t_low: tLow,
      t_high: tHigh,
      batch_size: Number(el("add-batch").value),
      seed: Number(el("add-seed").value || 42),
      n_conditions: Number(el("add-conditions").value),
      n_samples_per_cond: Number(el("add-samples").value),
      latent_size: Number(el("add-latent").value),
      model_type: el("add-model-type").value,
    };
  } else {
    const imageDir = el("add-image-dir").value.trim();
    if (!imageDir) {
      return addFieldError("Image dir is required (absolute, server-side).",
        "add-image-dir");
    }
    body = {
      kind: "ingest_lora",
      model,
      image_dir: imageDir,
      recursive: el("add-recursive").checked,
      resize_mode: el("add-resize-mode").value,
      latent_size: Number(el("add-import-latent").value),
      max_aspect_ratio: Number(el("add-max-aspect").value),
      model_type: el("add-import-model-type").value,
      neg_prompt: el("add-import-neg").value,
      seed: Number(el("add-import-seed").value || 42),
    };
  }

  const fromList = datasetName === null;
  api(`/datasets/${encodeURIComponent(name)}/tasks`, {
    method: "POST", body,
  }).then((task) => {
    log(`Task #${task.id} (${TASK_KIND_LABEL[task.kind] || task.kind}) ` +
        `started for '${name}' -- ${task.total} to process.`, "success");
    el("add-dialog").close();
    if (fromList) {
      location.href = `/datasets/${encodeURIComponent(name)}`;
    } else {
      showTab("tasks");
      loadTasks().then(renderTasks).catch(logError);
    }
  }).catch((err) => {
    logError(err); // 409 dataset_task_active, 422 invalid_query, ...
    showError("add-error", errText(err));
  });
}

/* Dataset tasks emit no events: poll while the detail view is open.
   When a sweep goes terminal, refresh items + stats too (rows landed). */
let lastActiveCount = 0;

async function pollTasks() {
  try {
    const prevActive = lastActiveCount;
    await loadTasks();
    lastActiveCount = tasks.filter(
      (t) => t.status === "pending" || t.status === "running").length;
    renderTasks();
    if (prevActive > 0 && lastActiveCount === 0) {
      log("A dataset task finished -- refreshing items.", "success");
      await loadItems();
      renderItems();
      reloadDetail();
    }
  } catch (err) {
    logError(err);
  }
}

function startPoll() {
  stopPoll();
  lastActiveCount = tasks.filter(
    (t) => t.status === "pending" || t.status === "running").length;
  pollTimer = setInterval(pollTasks, 4000);
}

function stopPoll() {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

/* ---- boot ---- */

function bindSeg(groupId, key, onPick) {
  el(groupId).addEventListener("click", (ev) => {
    const btn = ev.target.closest(".seg");
    if (!btn) return;
    for (const b of el(groupId).children) {
      b.classList.toggle("active", b === btn);
    }
    onPick(btn.dataset[key]);
  });
}

async function boot() {
  // global bindings
  el("btn-new-dataset").addEventListener("click", () => {
    el("create-card").hidden = false;
    el("new-name").focus();
  });
  el("btn-create").addEventListener("click", createDataset);
  el("new-name").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") createDataset();
  });
  for (const name of ["items", "sets", "tasks"]) {
    el(`tab-${name}`).addEventListener("click", () => showTab(name));
  }

  // toolbar: mode + filters + selection
  bindSeg("item-mode", "mode", setItemMode);
  el("item-filter").addEventListener("click", async (ev) => {
    const btn = ev.target.closest(".seg");
    if (!btn) return;
    itemFilter = btn.dataset.filter;
    for (const b of el("item-filter").children) {
      b.classList.toggle("active", b === btn);
    }
    try {
      await loadItems();
      renderItems();
    } catch (err) {
      logError(err);
      showError("items-error", errText(err));
    }
  });
  el("select-all").addEventListener("change", (ev) => {
    selected = ev.target.checked ? new Set(items.map((it) => it.id)) : new Set();
    renderItems();
  });

  // multi-edit panel
  bindSeg("bulk-field", "field", setBulkField);
  bindSeg("bulk-type", "btype", setBulkType);
  el("btn-bulk-apply").addEventListener("click", applyBulk);
  el("btn-clear-sel").addEventListener("click", () => {
    selected = new Set();
    renderItems();
  });
  el("btn-commit").addEventListener("click", commitToSet);
  el("btn-bulk-discard").addEventListener("click", () =>
    discardItems([...selected]));

  // add-data dialog
  el("btn-add-data").addEventListener("click", () =>
    openAddDialog(datasetName));
  el("btn-add-data-tab").addEventListener("click", () =>
    openAddDialog(datasetName));
  el("add-close").addEventListener("click", () => el("add-dialog").close());
  el("add-dialog").addEventListener("close", () => showError("add-error", ""));
  el("add-tab-generate").addEventListener("click", () => setAddTab("generate"));
  el("add-tab-import").addEventListener("click", () => setAddTab("import"));
  bindSeg("add-prompt-mode", "pmode", setAddPromptMode);
  bindSeg("add-neg-mode", "nmode", setAddNegMode);
  el("btn-add-start").addEventListener("click", startAddTask);
  for (const id of ["add-conditions", "add-samples", "add-latent",
                    "add-import-latent", "add-resize-mode"]) {
    el(id).addEventListener("input", updateAddHints);
    el(id).addEventListener("change", updateAddHints);
  }

  // item editor dialog
  el("item-ed-close").addEventListener("click", () => {
    if (editorDirty()
        && !window.confirm("Discard unsaved changes?")) return;
    closeItemEditor();
  });
  el("item-dialog").addEventListener("close", () => {
    editorIndex = -1;
    editorSnapshot = null;
  });
  el("item-ed-prev").addEventListener("click", () => editorNav(-1));
  el("item-ed-next").addEventListener("click", () => editorNav(1));
  el("item-ed-save").addEventListener("click", saveEditor);
  el("item-ed-revert").addEventListener("click", revertEditor);
  el("item-ed-discard").addEventListener("click", () => {
    const item = editorItem();
    if (item) discardItems([item.id]);
  });
  bindSeg("item-ed-type", "etype", (value) => {
    editorType = value;
    syncEditorDirty();
  });
  for (const id of ["item-ed-prompt", "item-ed-neg", "item-ed-cfg"]) {
    el(id).addEventListener("input", syncEditorDirty);
  }

  window.addEventListener("pagehide", stopPoll);

  // checkpoint catalog feeds the model datalist (suggestions only)
  try {
    const catalog = await api("/assets/checkpoint");
    const dl = el("dl-checkpoint");
    for (const opt of catalog.options || []) {
      dl.appendChild(h("option", { value: opt.value }));
    }
  } catch {
    // no suggestions is not an error
  }

  // item context menu (M8f): one option, closed by outside click / Escape
  el("item-menu-preview").addEventListener("click", setDatasetPreview);
  document.addEventListener("click", (ev) => {
    if (el("item-menu").hidden) return;
    if (ev.target.closest("#item-menu, .ds-item-menu")) return;
    closeItemMenu();
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && !el("item-menu").hidden) closeItemMenu();
  });

  // route: /datasets or /datasets/{name}
  const segs = location.pathname.split("/").filter(Boolean);
  const name = segs[0] === "datasets" && segs[1]
    ? decodeURIComponent(segs[1]) : null;
  if (name) await openDetail(name);
  else await showList();
}

boot().catch(logError);
