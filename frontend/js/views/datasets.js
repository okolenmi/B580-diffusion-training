/* ---------------------------------------------------------------------------
   datasets.js -- the Datasets page: image libraries (images + captions + sets).

     /datasets          the libraries, create and delete
     /datasets/{name}   one library: import, browse, caption, group into sets

   A library keeps images where they are; latent caches are built by runs, one per VAE.
   --------------------------------------------------------------------------- */

import { el } from "../lib/dom.js";
import { log } from "../lib/log.js";
import { LibraryDetail } from "../datasets/library_detail.js";
import { LibraryList } from "../datasets/library_list.js";

const note = (text, kind = "info") => log(text, kind);
const name = decodeURIComponent(location.pathname.replace(/^\/datasets\/?/, "")) || null;

const body = el("page-content");
if (name) {
  el("ds-title").textContent = name;
  el("ds-sub").textContent = "Images, captions and sets";
  el("btn-back").hidden = false;
  new LibraryDetail(body, { name, note });
} else {
  new LibraryList(body, { note }).refresh();
}
