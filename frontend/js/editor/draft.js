/* ---------------------------------------------------------------------------
   editor/draft.js -- the canvas survives a reload.

   The graph lives in memory (GraphDoc); only "Save to library" wrote it
   anywhere, so a reload (or a crashed tab) opened an empty canvas and
   everything not saved by name was gone. This keeps ONE working draft in
   this browser's localStorage: written (debounced) after every change,
   flushed when the page is hidden, read back at boot.

   It is a per-browser convenience, not the graph's home: the library is
   still where graphs are kept, shared and run from. Storage can be
   missing, full or corrupt (private window, quota, hand-edited), so every
   access is guarded and the editor works the same without it.

   Raw params are stored (GraphDoc.toDraft), not the run payload: the run
   payload drops a value whose input is fed by a wire, and removing that
   wire later must give the typed value back.
   --------------------------------------------------------------------------- */

export const DRAFT_KEY = "ed.draft.v1";

function defaultStorage() {
  try { return globalThis.localStorage || null; } catch { return null; }
}

/** The stored draft, or null when there is none / it cannot be trusted. */
export function readDraft(storage = defaultStorage()) {
  if (!storage) return null;
  let raw;
  try { raw = storage.getItem(DRAFT_KEY); } catch { return null; }
  if (!raw) return null;
  try {
    const draft = JSON.parse(raw);
    if (!draft || typeof draft !== "object" || !Array.isArray(draft.nodes)) return null;
    return draft.nodes.length ? draft : null;
  } catch {
    return null;
  }
}

/** Store the document; an empty canvas removes the draft (Clear must stick). */
export function writeDraft(doc, storage = defaultStorage()) {
  if (!storage) return false;
  try {
    if (!doc.size) storage.removeItem(DRAFT_KEY);
    else storage.setItem(DRAFT_KEY, JSON.stringify(doc.toDraft()));
    return true;
  } catch {
    return false; // quota / denied: the draft is a convenience
  }
}

/**
 * Start saving. Returns {notify, flush}: `notify()` after any change
 * schedules a write, `flush()` writes now (page hide / unload). Install it
 * only AFTER the stored draft has been restored, or the empty canvas at
 * boot would erase the thing being restored.
 */
export function createDraftSaver(doc, { delay = 400, storage, timers = globalThis } = {}) {
  let timer = null;
  const flush = () => {
    if (timer !== null) { timers.clearTimeout(timer); timer = null; }
    return writeDraft(doc, storage);
  };
  const notify = () => {
    if (timer !== null) timers.clearTimeout(timer);
    timer = timers.setTimeout(flush, delay);
  };
  return { notify, flush };
}
