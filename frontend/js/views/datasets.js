/* ---------------------------------------------------------------------------
   datasets.js -- the Datasets page: image libraries (pictures, captions, sets).

     /datasets          the datasets, create and delete
     /datasets/{name}   one dataset: browse, caption, group into sets, import

   A dataset keeps pictures where they are; latent caches are built by runs, one per VAE.
   --------------------------------------------------------------------------- */

import { el } from "../lib/dom.js";
import { log } from "../lib/log.js";
import { LibraryList } from "../datasets/library_list.js";
import { LibraryView } from "../datasets/library_view.js";

const note = (text, kind = "info") => log(text, kind);
const name = decodeURIComponent(location.pathname.replace(/^\/datasets\/?/, "")) || null;
const body = el("page-content");
const sub = el("ds-sub");

if (name) {
  document.body.classList.add("ds-open");
  document.title = `${name} · Datasets`;
  el("ds-title").textContent = name;
  el("btn-back").hidden = false;
  new LibraryView(body, { name, note, sub, actions: el("topbar-actions") });
} else {
  new LibraryList(body, { note, sub }).refresh();
}
