/* datasets/import_dialog.js -- read a folder of pictures (and their caption files) into the open dataset. The server
   does it in the background; this follows the job, and a small pill in the page header keeps showing its progress
   after the dialog is closed, then says how it ended until it is clicked or a while has passed. */

import { h } from "../lib/dom.js";
import { errText } from "../lib/errors.js";
import { datasets, plural } from "./client.js";
import { icon } from "./icons.js";

const POLL_MS = 900;
const RESULT_MS = 20000;

export class ImportDialog {
  /** @param {import("./library_state.js").LibraryState} state */
  constructor(state) {
    this.state = state;
    this.timer = null;
    this.watching = null;
    this.folder = h("input", { type: "text", placeholder: "/home/me/pictures/portraits", spellcheck: false, autocomplete: "off", "aria-label": "Folder" });
    this.folder.addEventListener("keydown", (e) => { if (e.key === "Enter") this.begin(); });
    this.recursive = h("input", { type: "checkbox", checked: true });
    this.into = h("input", { type: "text", list: "ds-set-names", placeholder: "none", spellcheck: false, "aria-label": "Also add to set" });
    this.start = h("button", { class: "btn btn-start", onclick: () => this.begin() }, icon("import"), "Import");
    this.fill = h("i", { class: "meter-fill" });
    this.meter = h("div", { class: "meter", hidden: true }, this.fill);
    this.status = h("div", { class: "imp-status" });
    this.skipped = h("details", { class: "imp-skipped", hidden: true });
    this.dialog = h("dialog", { class: "dlg" },
      h("div", { class: "dlg-body" },
        h("h2", { class: "dlg-title", text: "Import images" }),
        h("p", { class: "text-dim dlg-text", text: "Pictures are read where they are, not copied. A caption file beside a picture (name.txt, name.tags.txt, name.desc.txt) is imported with it; a picture already in the dataset keeps its captions unless a caption file exists." }),
        h("label", { class: "field" }, h("span", { text: "Folder on this machine" }), this.folder),
        h("div", { class: "imp-opts" },
          h("label", { class: "check" }, this.recursive, "Include subfolders"),
          h("label", { class: "field imp-into" }, h("span", { text: "Also add to set" }), this.into)),
        this.meter, this.status, this.skipped),
      h("div", { class: "dlg-foot" }, h("span", { class: "dlg-gap" }), h("button", { class: "btn btn-secondary", text: "Close", onclick: () => this.dialog.close() }), this.start));
    this.pill = h("button", { class: "imp-pill", hidden: true, title: "Show the import", onclick: () => { this.pill.hidden = true; this.open(); } });
    this.fading = null;
    document.body.append(this.dialog);
    this.follow();
  }

  open() {
    if (!this.dialog.open) this.dialog.showModal();
    this.folder.focus();
  }

  async begin() {
    const folder = this.folder.value.trim();
    if (!folder) { this.folder.focus(); return; }
    try {
      const job = await datasets.startImport(this.state.name, folder, this.recursive.checked, this.into.value.trim() || "all");
      this.watching = job.id;
      this.status.className = "imp-status";
      this.follow();
    } catch (err) {
      this.status.className = "imp-status bad";
      this.status.textContent = errText(err);
    }
  }

  /** Follow the newest job to its end; also picks up one that was already running when the page opened. */
  async follow() {
    clearTimeout(this.timer);
    let job;
    try { job = (await datasets.imports(this.state.name))[0]; } catch { return; }
    if (!job) return;
    const running = job.state === "running";
    this.start.disabled = running;
    this.meter.hidden = !running;
    if (running) { clearTimeout(this.fading); this.pill.className = "imp-pill"; this.pill.hidden = false; } else if (this.watching !== job.id) this.pill.hidden = true;
    const percent = job.total ? Math.round((100 * job.done) / job.total) : 0;
    this.fill.style.width = `${percent}%`;
    if (running) {
      this.watching = job.id;
      const text = job.total ? `Reading ${job.done.toLocaleString("en")} of ${job.total.toLocaleString("en")} pictures…` : "Looking for pictures…";
      this.status.className = "imp-status";
      this.status.textContent = text;
      this.pill.replaceChildren(icon("import", 13), job.total ? `${percent}%` : "…");
      this.timer = setTimeout(() => this.follow(), POLL_MS);
      return;
    }
    this.drawResult(job);
    if (this.watching === job.id) {
      this.watching = null;
      this.pill.className = `imp-pill ${job.state === "done" ? "ok" : "bad"}`;
      this.pill.replaceChildren(icon(job.state === "done" ? "check" : "x", 13), job.state === "done" ? `${job.added} added` : "Import failed");
      this.pill.hidden = false;
      this.fading = setTimeout(() => { this.pill.hidden = true; }, RESULT_MS);
      this.state.note(job.state === "done" ? `Import finished: ${job.added} added, ${job.known} already known.` : `Import failed: ${job.error}`, job.state === "done" ? "success" : "error");
      this.state.refresh();
    }
  }

  drawResult(job) {
    this.skipped.hidden = !(job.skipped?.length);
    if (job.state === "failed") {
      this.status.className = "imp-status bad";
      this.status.textContent = `The last import failed: ${job.error}`;
      return;
    }
    this.status.className = "imp-status ok";
    this.status.textContent = `Last import: ${plural(job.added, "picture")} added, ${job.known} already in the dataset.`;
    if (job.skipped?.length) {
      this.skipped.replaceChildren(h("summary", { text: `${plural(job.skipped.length, "file")} skipped` }),
        ...job.skipped.slice(0, 8).map(([path, why]) => h("div", { class: "imp-skip text-dim" }, h("code", { text: path.split("/").pop() }), ` — ${why}`)));
    }
  }
}
