/* ---------------------------------------------------------------------------
   editor/assets.js -- server-side asset catalogs for path params (path_kind).

   One cached GET per kind (`/assets/{kind}`), so the picker on a node and
   the picker in the inspector share one round trip and repeated canvas
   re-renders never re-fetch. The cache is invalidated after an upload so
   the next build shows the new file.

   Security: this module never builds a filesystem path. It sends file
   *names* to the API, which sandboxes every path server-side (the graph
   route is meant to be reachable from another device -- see
   infrastructure/file_asset_store.py). Client-side we still strip any
   directory component a browser might hand over, so an upload lands at
   the top of the kind's folder.
   --------------------------------------------------------------------------- */

import { api } from "../api.js";

/** Port path_kind -> asset kind (Save-As writes into the lora folder). */
export function assetKindFor(pathKind) {
  return pathKind === "lora_output" ? "lora" : pathKind;
}

const CATALOGS = new Map(); // kind -> Promise<catalog payload>

/** Cached catalog: {kind, base_dir, files[], options[{value,label}], upload_supported, browse_supported}. */
export function assetCatalog(kind) {
  if (!CATALOGS.has(kind)) {
    const pending = api(`/assets/${encodeURIComponent(kind)}`).catch((err) => {
      CATALOGS.delete(kind); // transient failure: retry on the next build
      throw err;
    });
    CATALOGS.set(kind, pending);
  }
  return CATALOGS.get(kind);
}

export function invalidateAssetCatalog(kind) {
  CATALOGS.delete(kind);
}

/**
 * Upload a File into the server's folder for `path_kind`.
 * `nameOverride` (Save-As) sends the typed relative path instead of the
 * file's own name; a plain upload strips any directory the browser may
 * hand over so it lands at the top of the folder. Returns the relative
 * path the server saved it under (the value the param should hold).
 * Rejects with ApiError (413 asset_too_large, 422 invalid_query for a
 * bad name/kind, ...).
 */
export async function uploadAsset(pathKind, file, nameOverride) {
  const kind = assetKindFor(pathKind);
  const name = nameOverride || file.name.split(/[\\/]/).pop();
  const encoded = name.split("/").map(encodeURIComponent).join("/");
  const res = await api(`/assets/${encodeURIComponent(kind)}/files/${encoded}`, {
    method: "PUT",
    rawBody: file,
    headers: { "Content-Type": "application/octet-stream" },
  });
  invalidateAssetCatalog(kind);
  return res.relative_path;
}
