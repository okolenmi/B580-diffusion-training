/* datasets/selection.js -- which images are ticked. Pure (no DOM), so the click rules are testable in plain node. */

export class Selection {
  constructor() {
    this.ids = new Set();
    this.anchor = null;
  }

  get size() { return this.ids.size; }
  has(id) { return this.ids.has(id); }
  list() { return [...this.ids]; }

  clear() {
    this.ids.clear();
    this.anchor = null;
  }

  /** Just this one image (a plain click). */
  only(id) {
    this.ids = new Set([id]);
    this.anchor = id;
  }

  toggle(id) {
    if (!this.ids.delete(id)) this.ids.add(id);
    this.anchor = id;
  }

  /** From the last clicked image to `id`, in the order shown (shift-click); without an anchor it is a plain click. */
  range(order, id) {
    const to = order.indexOf(id);
    if (to < 0) return;
    const from = this.anchor === null ? -1 : order.indexOf(this.anchor);
    if (from < 0) return this.only(id);
    const [a, b] = from < to ? [from, to] : [to, from];
    for (let i = a; i <= b; i++) this.ids.add(order[i]);
  }

  /** Everything given (the whole filtered view, loaded or not). */
  fill(ids) {
    this.ids = new Set(ids);
    this.anchor = ids[0] ?? null;
  }

  /** The one selected id, or null when there are none or several. */
  get single() { return this.ids.size === 1 ? [...this.ids][0] : null; }
}
