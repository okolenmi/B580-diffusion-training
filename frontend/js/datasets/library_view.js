/* datasets/library_view.js -- one dataset as a workbench: set tabs and a toolbar over the image mosaic, an inspector
   beside it. This class only assembles the parts and answers the few requests that span them. */

import { h } from "../lib/dom.js";
import { errText } from "../lib/errors.js";
import { datasets, fmtBytes, plural } from "./client.js";
import { confirmDialog, Lightbox } from "./dialogs.js";
import { ImportDialog } from "./import_dialog.js";
import { icon } from "./icons.js";
import { Inspector } from "./inspector.js";
import { LibraryState } from "./library_state.js";
import { Mosaic } from "./mosaic.js";
import { SetTabs } from "./set_tabs.js";
import { Toolbar } from "./toolbar.js";

export class LibraryView {
  /** @param {HTMLElement} root  @param {{name: string, note: Function, sub: HTMLElement, actions: HTMLElement}} page */
  constructor(root, { name, note, sub, actions }) {
    this.state = new LibraryState(name, note);
    this.sub = sub;
    const lightbox = new Lightbox({ name, url: (id) => datasets.imageUrl(name, id), channel: () => this.state.channel });
    this.importer = new ImportDialog(this.state);
    const hooks = { lightbox, onRemove: () => this.removeSelected(), onImport: () => this.importer.open() };

    const toolbarRoot = h("div", { class: "ds-toolbar" });
    const setsRoot = h("div", { class: "settabs", role: "tablist", "aria-label": "Sets" });
    const stage = h("div", { class: "ds-stage" });
    const inspectorRoot = h("aside", { class: "ds-inspector", "aria-label": "Inspector" });
    root.append(h("div", { class: "ds-work" }, h("div", { class: "ds-main" }, toolbarRoot, setsRoot, stage), inspectorRoot));

    this.mosaic = new Mosaic(stage, this.state, hooks);
    this.toolbar = new Toolbar(toolbarRoot, this.state, this.mosaic);
    this.tabs = new SetTabs(setsRoot, this.state);
    this.inspector = new Inspector(inspectorRoot, this.state, hooks);
    actions.append(this.importer.pill, h("button", { class: "btn btn-start btn-small", onclick: hooks.onImport }, icon("import", 14), "Import images"));
    this.state.addEventListener("summary", () => this.drawSub());
    this.state.start().then(() => this.mosaic.scroller.focus({ preventScroll: true }), (err) => this.fail(err));
  }

  drawSub() {
    const me = this.state.summary;
    const sets = Object.keys(me.sets).length - 1;
    this.sub.textContent = [plural(me.images, "image"), fmtBytes(me.bytes), sets ? plural(sets, "set") : ""].filter(Boolean).join(" · ");
  }

  fail(err) {
    this.state.note(errText(err), "error");
    this.mosaic.empty.hidden = false;
    this.mosaic.empty.replaceChildren(h("h3", { text: "This dataset could not be opened" }), h("p", { class: "text-dim", text: errText(err) }),
      h("a", { class: "btn btn-secondary", href: "/datasets", text: "Back to datasets" }));
  }

  async removeSelected() {
    const ids = this.state.selection.list();
    if (!ids.length) return;
    const ok = await confirmDialog({
      title: `Remove ${plural(ids.length, "image")} from the dataset?`,
      text: "Their captions and set memberships are deleted. The picture files on disk are not touched, and importing the folder again brings them back (without these captions).",
      ok: "Remove",
    });
    if (ok) await this.state.removeImages(ids);
  }
}
