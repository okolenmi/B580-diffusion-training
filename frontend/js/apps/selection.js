/* Which cards are selected in edit mode, and which board "add" lands in. */

export class CardSelection {
  constructor() {
    this.sel = null;          // the selected card id
    this.picked = new Set();  // every selected card (Ctrl/Shift-click adds); always holds `sel`
    this.target = null;       // the Board "add" lands in
  }

  get many() { return this.picked.size > 1; }

  /** The cards an action applies to: the picked ones, or the selected one. */
  ids() { return this.many ? [...this.picked] : [this.sel]; }

  select(id, target, extend) {
    // Cards are picked together only from one tab: a card on another board starts a new selection.
    if (extend && id && this.sel && this.target === target && this.target.alive) {
      if (this.picked.has(id) && this.picked.size > 1) this.picked.delete(id); else this.picked.add(id);
    } else {
      this.picked = new Set(id ? [id] : []);
    }
    this.sel = id && this.picked.has(id) ? id : [...this.picked][0] || null;
    this.target = target;
  }

  /** Select exactly one card (a new box). */
  only(id) {
    this.picked = new Set([id]);
    this.sel = id;
  }

  clear() {
    this.sel = null;
    this.picked = new Set();
  }

  /** Forget cards that no longer exist (`find` answers for an id). */
  prune(find) {
    this.picked = new Set([...this.picked].filter((id) => find(id)));
    if (this.sel && !this.picked.has(this.sel)) this.sel = [...this.picked][0] || null;
  }

  /** After the whole page was replaced: keep the selected card if it survived, drop the multi-selection. */
  reset(find) {
    if (this.sel && !find(this.sel)) this.sel = null;
    this.picked = new Set(this.sel ? [this.sel] : []);
  }
}
