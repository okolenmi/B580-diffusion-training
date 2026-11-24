/* The library of apps (/apps): the "new app" form (empty, or a copy of a saved graph) and the grid of existing apps. */

import { api } from "../api.js";
import { el, showMessage } from "../lib/dom.js";
import { errText } from "../lib/errors.js";

const enc = encodeURIComponent;

export async function showList() {
  el("view-list").hidden = false;
  el("view-page").hidden = true;
  try {
    const [apps, library] = await Promise.all([api("/apps"), api("/graphs/library")]);
    renderNewForm(library.graphs || [], new Set((apps.apps || []).map((t) => t.name)));
    renderCards(apps.apps || []);
  } catch (err) {
    const box = el("list-state");
    box.hidden = false;
    box.textContent = `Could not load apps: ${errText(err)}`;
  }
}

function renderNewForm(graphs, taken) {
  const select = el("new-graph");
  select.replaceChildren(Object.assign(document.createElement("option"), { value: "", textContent: "An empty app" }));
  for (const g of graphs) {
    const o = document.createElement("option");
    o.value = g.name;
    o.textContent = `A copy of ${g.name} (${g.node_count} nodes)`;
    select.append(o);
  }
  const create = el("btn-new");

  const go = () => {
    const nm = el("new-name").value.trim();
    if (!nm) return showMessage("new-error", "Give it a name.");
    if (taken.has(nm)) return showMessage("new-error", `"${nm}" already exists — pick another name or open the existing one.`);
    showMessage("new-error", "");
    location.href = `/apps/${enc(nm)}?new=1${select.value ? `&graph=${enc(select.value)}` : ""}`;
  };
  create.onclick = go;
  el("new-name").onkeydown = (e) => { if (e.key === "Enter") go(); };
}

function renderCards(items) {
  el("list-state").hidden = items.length > 0;
  el("list-state").textContent = "Nothing here yet — create an app above.";
  el("list-apps").hidden = items.length === 0;
  const grid = el("grid-apps");
  grid.replaceChildren();
  for (const t of items) {
    const card = document.createElement("a");
    card.className = "app-card";
    card.href = `/apps/${enc(t.name)}`;
    const name = document.createElement("div");
    name.className = "app-card-name";
    name.textContent = t.name;
    const desc = document.createElement("div");
    desc.className = "app-card-desc";
    desc.textContent = t.description || "";
    const meta = document.createElement("div");
    meta.className = "app-card-meta";
    meta.textContent = `v${t.version} · ${t.card_count} card${t.card_count === 1 ? "" : "s"} · ${t.node_count ? `${t.node_count} node${t.node_count === 1 ? "" : "s"}` : "no graph yet"} · ${new Date(t.updated_at).toLocaleDateString()}`;
    card.append(name, desc, meta);
    grid.append(card);
  }
}
