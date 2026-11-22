/* The tray of everything the app's graph offers (edit mode), searchable, with groups that keep their open state. */

import { buildAvailable } from "../board/model.js";
import { el } from "../lib/dom.js";

export class Tray {
  /** `targetBoard()` is the board an "add" lands in. */
  constructor({ targetBoard }) {
    this.targetBoard = targetBoard;
  }

  onChange(callback) {
    el("tray-adv").onchange = callback;
    el("tray-search").oninput = callback;
  }

  render(modules, blocks) {
    const tray = el("tray-list");
    const needle = el("tray-search").value.trim().toLowerCase();
    const groups = buildAvailable(modules, blocks, { showAdvanced: el("tray-adv").checked })
      .map((g) => ({
        ...g,
        items: needle
          ? g.items.filter((i) => `${i.label} ${g.title} ${i.sub}`.toLowerCase().includes(needle))
          : g.items,
      }))
      .filter((g) => g.items.length);

    // Keep open/closed state of groups across redraws.
    const open = new Map([...tray.querySelectorAll("details")].map((d) => [d.dataset.key, d.open]));
    tray.replaceChildren();
    if (!groups.length) {
      const empty = document.createElement("div");
      empty.className = "bd-tray-empty";
      empty.textContent = needle ? "Nothing matches." : "Nothing more to add.";
      tray.append(empty);
      return;
    }
    for (const g of groups) {
      const d = document.createElement("details");
      d.dataset.key = g.key;
      d.open = open.has(g.key) ? open.get(g.key) : g.scope === "graph" || groups.length <= 3 || !!needle;
      const sum = document.createElement("summary");
      sum.textContent = g.title;
      d.append(sum);
      if (g.scope === "node" && g.items.some((i) => i.kind === "field" && !i.placed && !i.advanced)) {
        d.append(this.#sectionButton(g));
      }
      for (const kind of ["display", "field"]) {
        const items = g.items.filter((i) => i.kind === kind);
        if (!items.length) continue;
        const cap = document.createElement("div");
        cap.className = "bd-tray-cap";
        cap.textContent = kind === "display" ? "Live views" : "Settings";
        d.append(cap);
        for (const it of items) d.append(this.#item(it));
      }
      tray.append(d);
    }
  }

  #sectionButton(group) {
    const form = document.createElement("button");
    form.type = "button";
    form.className = "bd-tray-item bd-tray-form";
    form.title = "Add this node's settings as one section of cards";
    form.textContent = "All settings as a section";
    form.onclick = () => this.targetBoard().addNodeSection(group.key.slice("node:".length));
    return form;
  }

  #item(it) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "bd-tray-item";
    b.disabled = it.placed;
    b.title = it.sub || it.label;
    const label = document.createElement("span");
    label.textContent = it.label;
    const tag = document.createElement("span");
    tag.className = "bd-tray-tag";
    tag.textContent = it.placed ? "added" : it.advanced ? "advanced" : "+";
    b.append(label, tag);
    b.onclick = () => this.targetBoard().addRef(it.ref);
    return b;
  }
}
