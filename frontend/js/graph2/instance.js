/* ---------------------------------------------------------------------------
   graph2/instance.js -- an instance: the frozen graph of one run, shown where the editor's graph usually is.

   The canvas, the inspector, the Monitor and the Views are all bound to one document, so looking at a run means
   putting the run's graph into that document and making it read-only; the user's own graph waits in memory
   (and the draft is not touched) until "Back to editor", or is replaced by "Copy to editor".
   --------------------------------------------------------------------------- */

export class InstanceView {
  /**
   * @param {{doc, canvas, saver, bar: HTMLElement, title: HTMLElement, copyBtn: HTMLElement, backBtn: HTMLElement,
   *   onNote: Function, onChange: (open: boolean) => void, onLeave: () => void, confirm?: (text: string) => boolean}} parts
   */
  constructor({ doc, canvas, saver, bar, title, copyBtn, backBtn, onNote, onChange = () => {}, onLeave = () => {}, confirm = (t) => globalThis.confirm(t) }) {
    Object.assign(this, { doc, canvas, saver, bar, title, onNote, onChange, onLeave, confirm });
    this.stash = null; // the user's own document while an instance is shown
    this.runId = null;
    copyBtn.addEventListener("click", () => this.copyToEditor());
    backBtn.addEventListener("click", () => this.back());
  }

  get open() { return this.runId !== null; }

  /** Show the graph a run was started with. A run that has none (an old format) is only reported. */
  enter(detail) {
    const graph = detail?.graph;
    if (!graph || !Array.isArray(graph.nodes) || !graph.nodes.length) {
      this.onNote(`Run #${detail?.execution_id} has no graph to show.`, "warn");
      return false;
    }
    const mine = this.open ? this.stash : this.doc.toJSON();
    try {
      this.doc.readOnly = false;
      this.doc.load(graph);
    } catch (err) {
      this.doc.readOnly = this.open;
      this.onNote(`The graph of run #${detail.execution_id} could not be shown: ${err.message || err}`, "error");
      return false;
    }
    this.stash = mine;
    this.runId = detail.execution_id;
    this.doc.readOnly = true;
    this.saver.hold(true);
    this.title.textContent = `Run #${this.runId}${detail.name ? " " + detail.name : ""} -- a frozen copy of its graph; it cannot be edited`;
    this.bar.hidden = false;
    this.canvas.fit();
    this.onChange(true);
    return true;
  }

  /** The user's own graph again. */
  back() {
    if (!this.open) return;
    this._close();
    this.doc.load(this.stash ?? { format: this.doc.toJSON().format, nodes: [], edges: [], event_edges: [], settings: {}, views: [] });
    this.stash = null;
    this.saver.hold(false);
    this.canvas.fit();
    this.onLeave();
  }

  /** Keep the run's graph as the editable one; the user's own canvas is given up (after asking, when it has content). */
  copyToEditor() {
    if (!this.open) return;
    if (this.stash?.nodes?.length && !this.confirm("Replace the canvas with this run's graph? Your current canvas is lost.")) return;
    this._close();
    this.stash = null;
    this.saver.hold(false);
    this.saver.flush();
    this.onNote("The run's graph is now the editable canvas.", "success");
    this.onLeave();
  }

  _close() {
    this.runId = null;
    this.doc.readOnly = false;
    this.bar.hidden = true;
    this.canvas.clearStatuses();
    this.onChange(false);
  }
}
