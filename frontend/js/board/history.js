/* ---------------------------------------------------------------------------
   board/history.js -- undo / redo over snapshots of a page's layout.

   A snapshot is any string (the page's JSON). `record` keeps one only when it differs from the current one, and a run of records
   marked `merge` (typing in a note) collapses into a single step. Values typed into field cards are not part of it: they live in the
   graph, not the layout.
   --------------------------------------------------------------------------- */

export class History {
  constructor(limit = 100) {
    this.limit = limit;
    this.steps = [];
    this.at = -1;
    this.mergeable = false;
  }

  /** Forget everything and start from `snapshot` (the page as loaded or saved). */
  reset(snapshot) {
    this.steps = [snapshot];
    this.at = 0;
    this.mergeable = false;
  }

  /** Make `snapshot` the current step; whatever could still be redone is dropped. Returns whether anything was recorded. */
  record(snapshot, { merge = false } = {}) {
    if (this.at >= 0 && this.steps[this.at] === snapshot) return false;
    if (merge && this.mergeable && this.at > 0) {
      this.steps[this.at] = snapshot;
      this.steps.length = this.at + 1;
      return true;
    }
    this.steps.length = this.at + 1;
    this.steps.push(snapshot);
    if (this.steps.length > this.limit) this.steps.shift();
    this.at = this.steps.length - 1;
    this.mergeable = merge;
    return true;
  }

  get canUndo() { return this.at > 0; }

  get canRedo() { return this.at >= 0 && this.at < this.steps.length - 1; }

  undo() {
    if (!this.canUndo) return null;
    this.mergeable = false;
    return this.steps[--this.at];
  }

  redo() {
    if (!this.canRedo) return null;
    this.mergeable = false;
    return this.steps[++this.at];
  }
}
