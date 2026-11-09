/* ---------------------------------------------------------------------------
   graph2/palette.js -- adding nodes: the sidebar list (grouped by category,
   searchable) and the quick-add popup (a double click on empty canvas, or a
   wire dropped on empty canvas, which lists only classes that can take or
   give that wire).
   --------------------------------------------------------------------------- */

import { hasSocket, wireProblem } from "./ports.js";

/**
 * The ports of a class that could be wired to `port` (an output when `fromOutput`, else an input), as `{info, port, possible}`.
 * A dynamic node with no outputs yet is offered through the outputs it may have (`possible`); it cannot be wired until it resolves.
 */
export function compatibleClasses(nodes, port, fromOutput) {
  const fits = (p) => (fromOutput ? wireProblem(port, p) === null : wireProblem(p, port) === null);
  const found = [];
  for (const info of nodes) {
    const side = fromOutput ? info.inputs.filter(hasSocket) : info.outputs;
    const match = side.find(fits);
    if (match) {
      found.push({ info, port: match, possible: false });
      continue;
    }
    const maybe = fromOutput ? null : (info.possible_outputs || []).find(fits);
    if (maybe) found.push({ info, port: maybe, possible: true });
  }
  return found;
}

/** Search over display name, class name and category; an empty query keeps everything. */
export function matches(info, query) {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return [info.display_name, info.class_name, info.category].some((s) => (s || "").toLowerCase().includes(q));
}

export function byCategory(nodes, query = "") {
  const groups = new Map();
  for (const info of nodes.filter((n) => matches(n, query))) {
    const key = info.category || "Other";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(info);
  }
  for (const list of groups.values()) list.sort((a, b) => a.display_name.localeCompare(b.display_name));
  return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b));
}

const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

/** The sidebar list: a click adds the node at the centre of the view, offset so successive nodes do not stack. */
export class Palette {
  constructor(root, search, { shapes, onAdd }) {
    this.root = root;
    this.search = search;
    this.shapes = shapes;
    this.onAdd = onAdd;
    this.search.addEventListener("input", () => this.render());
  }

  render() {
    this.root.replaceChildren();
    const catalog = this.shapes.catalog;
    if (!catalog) return;
    const errors = catalog.load_errors || [];
    if (errors.length) {
      const box = make("details", "ed-domain");
      box.append(make("summary", "", `Modules that failed to load (${errors.length})`));
      for (const error of errors) box.append(make("div", "console-line error", `${error.module}: ${error.error}`));
      this.root.append(box);
    }
    const query = this.search.value;
    for (const [category, list] of byCategory(catalog.nodes, query)) {
      const details = make("details", "ed-domain");
      details.open = Boolean(query) || list.length <= 6;
      details.append(make("summary", "", `${category} (${list.length})`));
      for (const info of list) {
        const item = make("div", "ed-palette-item", info.display_name);
        item.title = info.doc || info.class_name;
        item.addEventListener("click", () => this.onAdd(info.class_name));
        details.append(item);
      }
      this.root.append(details);
    }
  }
}

/** The floating list shown where a wire was dropped or the canvas was double-clicked. */
export class QuickAdd {
  constructor({ shapes, onPick }) {
    this.shapes = shapes;
    this.onPick = onPick;
    this.box = make("div", "g2-quick");
    this.box.hidden = true;
    document.body.append(this.box);
    document.addEventListener("pointerdown", (e) => { if (!this.box.contains(e.target)) this.close(); }, true);
  }

  close() {
    this.box.hidden = true;
  }

  /** @param {{x:number,y:number}} client where to show it; @param {Array<{info, port, possible}>|null} choices null lists every class */
  open(client, choices, title) {
    const all = choices || this.shapes.catalog.nodes.map((info) => ({ info, port: null }));
    this.box.replaceChildren(make("div", "g2-quick-title", title));
    const input = make("input", "cfg-input");
    input.placeholder = "type to filter…";
    const list = make("div", "g2-quick-list");
    const draw = () => {
      list.replaceChildren();
      for (const { info, port, possible } of all.filter((c) => matches(c.info, input.value)).slice(0, 40)) {
        const tag = port ? ` · ${port.label || port.name}${possible ? " (possible)" : ""}` : "";
        const row = make("div", "ed-palette-item", `${info.display_name}${tag}`);
        row.title = possible ? `${info.category} -- the output exists once the node is set up, so it is added unconnected` : info.category;
        row.addEventListener("click", () => { this.close(); this.onPick(info.class_name, port, possible); });
        list.append(row);
      }
      if (!list.children.length) list.append(make("div", "console-line info", "nothing fits"));
    };
    input.addEventListener("input", draw);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Escape") this.close();
      if (e.key === "Enter") list.firstElementChild?.click();
    });
    this.box.append(input, list);
    this.box.hidden = false;
    this.box.style.left = `${Math.max(0, Math.min(client.x, window.innerWidth - this.box.offsetWidth - 8))}px`;
    this.box.style.top = `${Math.max(0, Math.min(client.y, window.innerHeight - this.box.offsetHeight - 8))}px`;
    draw();
    input.focus();
  }
}
