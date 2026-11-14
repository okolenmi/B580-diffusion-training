/* ---------------------------------------------------------------------------
   editor/palette.js -- node catalog sidebar (M7).

   Renders /graphs/nodes (GraphCatalogOut: {count, domains, load_errors})
   as one <details> per domain; clicking an entry drops a node on the
   canvas center (slightly jittered per count so drops don't stack).
   load_errors surface as red rows -- a module that failed to import is
   information, not a blank palette.

   Search filters by display/class name, case-insensitively, and hides
   domains that end up empty.
   --------------------------------------------------------------------------- */

export class Palette {
  constructor(root, searchInput, { onAdd, onNote }) {
    this.root = root;
    this.search = searchInput;
    this.onAdd = onAdd;
    this.onNote = onNote;
    this.catalog = null;
    this._dropCount = 0;
    this.search.addEventListener("input", () => this._filter());
  }

  load(catalog) {
    this.catalog = catalog;
    this.render();
  }

  render() {
    this.root.replaceChildren();
    if (!this.catalog) return;

    for (const [domain, classes] of Object.entries(this.catalog.domains || {})) {
      if (!classes.length) continue;
      const details = document.createElement("details");
      details.className = "ed-domain";
      details.open = Object.keys(this.catalog.domains).length <= 3;
      const summary = document.createElement("summary");
      summary.textContent = `${domain} (${classes.length})`;
      details.appendChild(summary);

      for (const cls of classes) {
        const item = document.createElement("button");
        item.className = "ed-palette-item";
        item.textContent = cls.display_name || cls.class_name;
        item.title = cls.doc ? `${cls.class_name} -- ${cls.doc}` : cls.class_name;
        item.dataset.search = `${cls.display_name} ${cls.class_name}`.toLowerCase();
        item.addEventListener("click", () => this._add(cls));
        details.appendChild(item);
      }
      this.root.appendChild(details);
    }

    const errors = this.catalog.load_errors || [];
    for (const e of errors) {
      const row = document.createElement("div");
      row.className = "ed-load-error";
      row.textContent = `load failed: ${e.module} -- ${e.message}`;
      this.root.appendChild(row);
    }
    if (!this.catalog.count && !errors.length) {
      const line = document.createElement("div");
      line.className = "console-line warn";
      line.textContent = "Catalog is empty.";
      this.root.appendChild(line);
    }
    this._filter();
  }

  _add(cls) {
    const { x, y } = this.dropPosition();
    const node = this.onAdd(cls.class_name, x, y);
    this.onNote(`Added ${cls.display_name || cls.class_name}${node ? ` as ${node.id}` : ""}.`);
  }

  /**
   * Where the next dropped node lands: the viewport center of the canvas
   * scroller (passed in by the editor), offset a little each time so
   * consecutive drops stay visible instead of perfectly overlapping.
   * The editor owns this closure because only it can see the scroller.
   */
  dropPosition = () => ({ x: 80, y: 80 });

  _filter() {
    const q = this.search.value.trim().toLowerCase();
    for (const details of this.root.querySelectorAll(".ed-domain")) {
      let visible = 0;
      for (const item of details.querySelectorAll(".ed-palette-item")) {
        const hit = !q || item.dataset.search.includes(q);
        item.hidden = !hit;
        if (hit) visible++;
      }
      details.hidden = visible === 0;
      if (q && visible > 0) details.open = true;
    }
  }
}
