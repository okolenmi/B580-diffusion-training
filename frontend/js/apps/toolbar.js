/* The page's toolbar: buttons, the mode switch, undo/redo state, and what Save and Run say about the document. */

import { el } from "../lib/dom.js";

const SAVE_FIRST = "Save this app first.";
const NO_GRAPH = "This app has no nodes yet: use Edit graph to build its graph.";
const EXPOSURE = "Save the app first — it now shows fields the saved app does not.";

export class Toolbar {
  /** `on` maps button names to handlers: mode, save, run, queue, stop, remove, graph, undo, redo, note, heading, section, columns. */
  wire(on) {
    el("mode-use").onclick = () => on.mode("use");
    el("mode-edit").onclick = () => on.mode("edit");
    el("btn-save").onclick = on.save;
    el("btn-run").onclick = () => on.run("auto");
    el("btn-queue").onclick = () => on.run("queue");
    el("btn-stop").onclick = on.stop;
    el("btn-delete").onclick = on.remove;
    el("btn-graph").onclick = on.graph;
    el("add-auto").onclick = on.fill;
    el("btn-undo").onclick = on.undo;
    el("btn-redo").onclick = on.redo;
    el("btn-phone").onclick = () => {
      const phone = el("board").classList.toggle("bd-phone");
      el("btn-phone").setAttribute("aria-pressed", String(phone));
    };
    el("add-note").onclick = on.note;
    el("add-heading").onclick = on.heading;
    el("add-section").onclick = on.section;
    el("cols").onchange = () => on.columns(el("cols").value);
  }

  showMode(mode) {
    el("mode-use").classList.toggle("active", mode === "use");
    el("mode-edit").classList.toggle("active", mode === "edit");
    el("mode-use").setAttribute("aria-pressed", String(mode === "use"));
    el("mode-edit").setAttribute("aria-pressed", String(mode === "edit"));
    el("tray").hidden = mode !== "edit";
    document.body.classList.toggle("app-editing", mode === "edit");
  }

  showColumns(n) { el("cols").value = String(n); }

  showHistory(history) {
    el("btn-undo").disabled = !history.canUndo;
    el("btn-redo").disabled = !history.canRedo;
  }

  /** Describe the app (`doc`) and the run (`active`) on the Save / Run / Queue buttons and the state line. */
  render(doc, active) {
    const dirty = doc.isDirty();
    const save = el("btn-save");
    save.textContent = dirty ? "Save •" : "Save";
    save.classList.toggle("is-dirty", dirty);

    el("btn-stop").hidden = !active;
    const exposure = !doc.isNew && doc.exposureChanged();
    let why = "";
    if (doc.isNew) why = SAVE_FIRST;
    else if (!doc.hasNodes) why = NO_GRAPH;
    else if (exposure) why = EXPOSURE;
    else if (active) why = "A run is already in progress.";
    const run = el("btn-run");
    run.disabled = !!why;
    run.title = why || "Run the graph with the values on this page";
    const queueWhy = doc.isNew ? SAVE_FIRST : !doc.hasNodes ? NO_GRAPH : exposure ? EXPOSURE : "";
    const queue = el("btn-queue");
    queue.disabled = !!queueWhy;
    queue.title = queueWhy || "Run these values after whatever is running or waiting";

    const bits = [];
    if (doc.isNew) bits.push("not saved yet");
    else if (dirty) bits.push(doc.layoutDirty() && doc.valuesDirty() ? "unsaved layout and values" : doc.layoutDirty() ? "unsaved layout" : "unsaved values");
    el("bd-state").textContent = bits.join(" · ");
  }
}
