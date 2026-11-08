/* ---------------------------------------------------------------------------
   graph2/file_io.js -- a graph as a file: open one written by the command
   line (`python -m next.run sdxl-lora ... --dump graph.json`) or by an
   earlier export, and download the open one. Only format 3 is read; the
   document says why when it is not.
   --------------------------------------------------------------------------- */

import { errText } from "../lib/errors.js";

export class FileIO {
  constructor({ importBtn, exportBtn, input, nameInput }, doc, { onNote, onLoaded }) {
    this.doc = doc;
    this.onNote = onNote;
    this.onLoaded = onLoaded;
    this.nameInput = nameInput;
    importBtn.addEventListener("click", () => input.click());
    input.addEventListener("change", () => this.read(input.files[0]).finally(() => { input.value = ""; }));
    exportBtn.addEventListener("click", () => this.write());
  }

  async read(file) {
    if (!file) return;
    if (this.doc.nodes.size && !window.confirm(`Replace the current canvas with ${file.name}?`)) return;
    try {
      this.doc.load(JSON.parse(await file.text()));
      this.onNote(`Opened ${file.name} (${this.doc.nodes.size} nodes).`, "success");
      this.onLoaded?.();
    } catch (err) {
      this.onNote(`${file.name}: ${err instanceof SyntaxError ? "not JSON" : errText(err)}`, "error");
    }
  }

  write() {
    if (!this.doc.nodes.size) return this.onNote("Canvas is empty -- nothing to export.", "warn");
    const name = (this.nameInput.value.trim() || "graph").replace(/[^\w.-]+/g, "_");
    const link = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(new Blob([JSON.stringify(this.doc.toJSON(), null, 2)], { type: "application/json" })),
      download: `${name}.json`,
    });
    link.click();
    URL.revokeObjectURL(link.href);
  }
}
