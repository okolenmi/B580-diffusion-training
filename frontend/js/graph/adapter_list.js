/* ---------------------------------------------------------------------------
   graph/adapter_list.js -- the control of a list of adapters (`ui.widget: "adapter_list"`): one row per adapter with an on/off box,
   its name, a strength and move/remove buttons, and an **Add** button that opens the files the server lists with a search box,
   so no file name is ever typed. The value is [{file, strength, on}] (lib/adapter_rows.js).
   --------------------------------------------------------------------------- */

import { addRow, moveBy, offered, patchAt, removeAt, rowsOf, splitName } from "../lib/adapter_rows.js";

const make = (tag, className, attrs = {}) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  Object.assign(node, attrs);
  return node;
};

function iconButton(text, title, onClick) {
  const b = make("button", "adl-btn", { type: "button", textContent: text, title });
  b.addEventListener("click", onClick);
  return b;
}

export function adapterListControl(spec, { value, onCommit, options, note }) {
  let rows = rowsOf(value);
  let files = null; // the server's list, fetched when Add is first opened
  const root = make("div", "adl");
  const list = make("div", "adl-rows");
  const add = make("button", "btn btn-secondary btn-small adl-add", { type: "button", textContent: "+ Add adapter" });
  const picker = make("div", "adl-picker");
  picker.hidden = true;
  const search = make("input", "cfg-input", { type: "search", placeholder: "Search the adapters folder…" });
  const found = make("div", "adl-found");
  picker.append(search, found);
  root.append(list, add, picker);

  const commit = (next) => {
    rows = next;
    onCommit(rows.length ? rows : undefined);
    draw();
  };

  function drawRow(r, i) {
    const on = make("input", "", { type: "checkbox", checked: r.on, title: "Use this adapter" });
    on.addEventListener("change", () => commit(patchAt(rows, i, { on: on.checked })));
    const { folder, base } = splitName(r.file);
    const name = make("span", "adl-name", { title: r.file });
    name.append(make("b", "", { textContent: base }), ...(folder ? [make("small", "", { textContent: ` ${folder}` })] : []));
    const strength = make("input", "cfg-input adl-strength", { type: "number", step: "0.05", value: String(r.strength), title: "Strength" });
    strength.addEventListener("change", () => {
      const n = parseFloat(strength.value);
      if (Number.isNaN(n)) { strength.value = String(r.strength); return note?.("Strength must be a number.", "warn"); }
      commit(patchAt(rows, i, { strength: n }));
    });
    const line = make("div", `adl-row${r.on ? "" : " is-off"}`);
    line.append(on, name, strength,
      iconButton("↑", "Move up", () => commit(moveBy(rows, i, -1))),
      iconButton("↓", "Move down", () => commit(moveBy(rows, i, 1))),
      iconButton("✕", "Remove", () => commit(removeAt(rows, i))));
    return line;
  }

  function draw() {
    list.replaceChildren(...(rows.length ? rows.map(drawRow) : [make("div", "adl-empty", { textContent: "No adapters" })]));
    if (!picker.hidden) drawFound();
  }

  function drawFound() {
    if (files === null) return found.replaceChildren(make("div", "adl-empty", { textContent: "Loading…" }));
    const shown = offered(files, search.value, rows);
    found.replaceChildren(...(shown.length ? shown.map((file) => {
      const { folder, base } = splitName(file);
      const item = make("button", "adl-item", { type: "button", title: file });
      item.append(make("b", "", { textContent: base }), ...(folder ? [make("small", "", { textContent: ` ${folder}` })] : []));
      item.addEventListener("click", () => { picker.hidden = true; search.value = ""; commit(addRow(rows, file)); });
      return item;
    }) : [make("div", "adl-empty", { textContent: files.length ? "Nothing matches" : "No adapter files found" })]));
  }

  add.addEventListener("click", async () => {
    picker.hidden = !picker.hidden;
    if (picker.hidden) return;
    drawFound();
    search.focus();
    try {
      files = await Promise.resolve(options ? options() : []);
    } catch {
      files = [];
      note?.("The adapter list is unavailable.", "warn");
    }
    drawFound();
  });
  search.addEventListener("input", drawFound);
  search.addEventListener("keydown", (e) => { if (e.key === "Escape") { picker.hidden = true; e.stopPropagation(); } });

  draw();
  return root;
}
