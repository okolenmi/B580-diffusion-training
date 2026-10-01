/* ---------------------------------------------------------------------------
   datasets.js -- Datasets page entry (M8c).

   Two views on one page, routed by the URL:
     /datasets         -- card list, create/delete
     /datasets/{name}  -- detail: stats chips + Items | Sets | Tasks tabs

   Items are the curation surface: filter by membership, edit prompts
   inline, toggle good/bad, discard, bulk-apply prompts, commit
   selections into training sets. Cache sweeps (ingest tasks) run one
   at a time per dataset (409 dataset_task_active) and are polled --
   dataset tasks publish no events, so polling is the source of truth
   here (4s while the detail view is open).

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
let selected = new Set(); // item ids marked in the grid
let tasks = [];           // all task rows for the dataset
let pollTimer = null;

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

const previewUrl = (previewPath) =>
  `/api/v1/datasets/${encodeURIComponent(datasetName)}/files/` +
  previewPath.split("/").map(encodeURIComponent).join("/");

/* ---- view switching ---- */

function showView(which) {
  const list = which === "list";
  el("view-list").hidden = !list;
  el("view-detail").hidden = list;
  el("topbar-actions").hidden = !list;
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

  const card = h("div", { class: "ds-card", onclick: (ev) => {
    // real link for normal/middle clicks; this handler covers the card body
    if (ev.target.closest("button, a")) return;
    location.href = nameLink.href;
  }},
    h("div", { class: "ds-card-head" }, nameLink,
      legacy ? h("span", { class: "ds-legacy", text: "legacy v1" }) : null),
    h("div", { class: "ds-card-desc", text: info.description || "" }),
    h("div", { class: "ds-card-meta", text: meta }),
    h("div", { class: "ds-card-meta", text: `created ${fmtTime(info.created_at)}` }),
    h("div", { class: "ds-card-foot" }, del),
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
    `${items.length} shown · ${selected.size} selected`;
  const all = items.length > 0 && selected.size === items.length;
  el("select-all").checked = all;
  const grid = el("items-grid");
  grid.replaceChildren();

  if (!items.length) {
    showState("items-state",
      itemFilter === "pending"
        ? "Nothing awaiting review -- every item is already used or bad."
        : itemFilter === "used"
          ? "No items are committed to training yet."
          : "This dataset has no items yet. Start a cache sweep on the Tasks tab.");
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

  const promptBox = h("div", {
    class: `item-prompt${item.prompt ? "" : " empty"}`,
    text: item.prompt || "(no prompt -- click to edit)",
    title: "Click to edit the prompt",
    onclick: () => editPrompt(item),
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
    },
    h("div", { class: "ds-thumb" }, thumbKids),
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

function editPrompt(item) {
  const card = el("items-grid")
    .querySelector(`[data-item-id="${item.id}"]`);
  if (!card) return;
  const promptBox = card.querySelector(".item-prompt");
  const area = h("textarea", { class: "item-prompt-edit", text: item.prompt });
  const save = h("button", {
    class: "btn btn-start btn-small", text: "Save",
    onclick: async () => {
      try {
        const updated = await api(dsApi(`/items/${item.id}`), {
          method: "PATCH", body: { prompt: area.value },
        });
        Object.assign(item, updated);
        log(`Item ${item.id} prompt saved.`, "success");
        renderItems();
      } catch (err) {
        logError(err);
        showError("items-error", errText(err));
      }
    },
  });
  const cancel = h("button", {
    class: "btn btn-secondary btn-small", text: "Cancel",
    onclick: () => renderItems(),
  });
  promptBox.replaceWith(area);
  area.insertAdjacentElement("afterend",
    h("div", { class: "item-edit-row" }, save, cancel));
  area.focus();
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

function renderBulkBar() {
  const bar = el("bulk-bar");
  bar.hidden = selected.size === 0;
  el("bulk-count").textContent = `${selected.size} selected`;
}

async function bulkPrompt() {
  if (!selected.size) return;
  const prompt = el("bulk-prompt").value;
  try {
    const res = await api(dsApi("/items"), {
      method: "PATCH",
      body: {
        item_ids: [...selected],
        prompt,
        prompt_mode: el("bulk-mode").value,
      },
    });
    log(`Prompt applied to ${res.updated} item(s).`, "success");
    el("bulk-prompt").value = "";
    await loadItems();
    renderItems();
  } catch (err) {
    logError(err);
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
    showState("tasks-state", "No cache sweeps recorded for this dataset yet.");
    return;
  }
  showState("tasks-state", "");
  for (const task of tasks) {
    const active = task.status === "pending" || task.status === "running";
    const pct = task.total > 0
      ? Math.min(100, Math.round((task.current / task.total) * 100)) : 0;
    list.appendChild(h("div", { class: "ds-task" },
      h("span", { class: "ds-task-id", text: `#${task.id}` }),
      h("span", { class: "ds-task-kind", text: task.kind }),
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

async function startTask() {
  const imageDir = el("task-image-dir").value.trim();
  const model = el("task-model").value.trim();
  showError("task-error", "");
  if (!imageDir || !model) {
    showError("task-error",
      "Image dir and checkpoint are required (absolute dir, catalog model).");
    return;
  }
  try {
    await api(dsApi("/tasks"), {
      method: "POST",
      body: {
        kind: "ingest_lora",
        image_dir: imageDir,
        model,
        recursive: el("task-recursive").checked,
        seed: Number(el("task-seed").value || 42),
      },
    });
    log("Cache sweep started.", "success");
    el("task-image-dir").value = "";
    await loadTasks();
    renderTasks();
  } catch (err) {
    logError(err); // 409 dataset_task_active: one sweep at a time
    showError("task-error", errText(err));
  }
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
      log("A cache sweep finished -- refreshing items.", "success");
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

/* ---- monitor hand-off (same as dashboard.js) ---- */

function openMonitor() {
  const id = el("monitor-id-input").value.trim();
  if (!id) {
    log("Paste a monitor id first (it is in the monitor page URL).", "warn");
    el("monitor-id-input").focus();
    return;
  }
  window.location.href = `/monitor/${encodeURIComponent(id)}`;
}

/* ---- boot ---- */

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
  el("btn-bulk-prompt").addEventListener("click", bulkPrompt);
  el("btn-commit").addEventListener("click", commitToSet);
  el("btn-bulk-discard").addEventListener("click", () =>
    discardItems([...selected]));
  el("btn-start-task").addEventListener("click", startTask);
  el("btn-open-monitor").addEventListener("click", openMonitor);
  el("monitor-id-input").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") openMonitor();
  });
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

  // route: /datasets or /datasets/{name}
  const segs = location.pathname.split("/").filter(Boolean);
  const name = segs[0] === "datasets" && segs[1]
    ? decodeURIComponent(segs[1]) : null;
  if (name) await openDetail(name);
  else await showList();
}

boot().catch(logError);
