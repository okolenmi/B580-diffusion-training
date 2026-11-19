/* ---------------------------------------------------------------------------
   board/inspector.js -- the small form for the selected card (edit mode).

   Shows what a card's header cannot: its look (style), whether a note is a heading, where it lives (move it into another section or
   tab) and remove. It edits nothing itself; each control reports through the callbacks and the page applies the change to the
   document, then asks for `show` again.
   --------------------------------------------------------------------------- */

const STYLES = [
  ["plain", "Plain"], ["outlined", "Outlined"], ["accent", "Accent"], ["bare", "No frame"],
];

const h = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

function row(label, control) {
  const wrap = h("label", "bd-insp-row");
  wrap.append(h("span", "", label), control);
  return wrap;
}

export class Inspector {
  /**
   * @param {HTMLElement} host
   * @param {{patch: (id: string, patch: object) => void, move: (id: string, owner: string, tabId: string) => void,
   *          remove: (id: string) => void}} actions
   */
  constructor(host, actions) {
    this.host = host;
    this.actions = actions;
    this.hide();
  }

  hide() {
    this.host.hidden = true;
    this.host.replaceChildren();
  }

  /**
   * @param {object} info
   * @param {object} info.block          the selected card
   * @param {string} info.defaultTitle   what the title is when not renamed
   * @param {{owner: string, tabId: string, label: string}[]} info.targets  where it could move
   * @param {{owner: string, tabId: string}|null} info.here
   */
  show({ block, defaultTitle, targets, here }) {
    const { actions } = this;
    this.host.hidden = false;
    this.host.replaceChildren(h("strong", "bd-insp-head", "Selected card"));

    const title = h("input", "cfg-input");
    title.type = "text";
    title.maxLength = 120;
    title.value = block.title || "";
    title.placeholder = defaultTitle;
    title.setAttribute("aria-label", "Card title");
    title.addEventListener("change", () => actions.patch(block.id, { title: title.value.trim() || null }));
    this.host.append(row("Title", title));

    const style = h("select", "cfg-input");
    for (const [value, label] of STYLES) style.append(new Option(label, value));
    style.value = block.style || "plain";
    style.addEventListener("change", () => actions.patch(block.id, { style: style.value === "plain" ? undefined : style.value }));
    this.host.append(row("Look", style));

    if (block.ref && block.ref.scope === "note") {
      const look = h("select", "cfg-input");
      look.append(new Option("Text", "text"), new Option("Heading", "heading"));
      look.value = (block.options && block.options.look) || "text";
      look.addEventListener("change", () => actions.patch(block.id, { options: { ...(block.options || {}), look: look.value } }));
      this.host.append(row("Note is", look));
    }

    if (targets.length > 1) {
      const move = h("select", "cfg-input");
      for (const t of targets) move.append(new Option(t.label ? `Page › ${t.label}` : "Page", `${t.owner}\u0000${t.tabId}`));
      move.value = here ? `${here.owner}\u0000${here.tabId}` : "";
      move.addEventListener("change", () => {
        const [owner, tabId] = move.value.split("\u0000");
        actions.move(block.id, owner, tabId);
      });
      this.host.append(row("Place in", move));
    }

    const remove = h("button", "btn btn-kill btn-small", "Remove card");
    remove.type = "button";
    remove.addEventListener("click", () => actions.remove(block.id));
    this.host.append(remove);
  }
}
