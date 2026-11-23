/* ---------------------------------------------------------------------------
   lib/adapter_rows.js -- the value of an adapter list: [{file, strength, on}], and the pure operations the list control uses.
   A string (an older saved graph kept one adapter per line, `file=strength`) is read as the same list.
   --------------------------------------------------------------------------- */

const row = (file, strength = 1, on = true) => ({ file, strength, on });

/** The rows a stored value stands for; anything that is not a row is left out. */
export function rowsOf(value) {
  if (typeof value === "string") {
    const rows = [];
    for (const line of value.split("\n")) {
      const text = line.trim();
      if (!text || text.startsWith("#")) continue;
      const cut = text.lastIndexOf("=");
      const file = (cut < 0 ? text : text.slice(0, cut)).trim();
      const tail = cut < 0 ? "1" : text.slice(cut + 1).trim().toLowerCase();
      const strength = tail === "off" ? 1 : Number(tail);
      if (file && Number.isFinite(strength)) rows.push(row(file, strength, tail !== "off"));
    }
    return rows;
  }
  if (!Array.isArray(value)) return [];
  return value.filter((r) => r && typeof r.file === "string" && r.file)
    .map((r) => row(r.file, Number.isFinite(Number(r.strength)) ? Number(r.strength) : 1, r.on !== false));
}

/** The rows with ``file`` added at the end; the same rows when it is already listed. */
export function addRow(rows, file) {
  return rows.some((r) => r.file === file) ? rows : [...rows, row(file)];
}

export const removeAt = (rows, i) => rows.filter((_, k) => k !== i);

export function patchAt(rows, i, changes) {
  return rows.map((r, k) => (k === i ? { ...r, ...changes } : r));
}

/** The rows with the one at ``i`` moved by ``delta`` places (stays put at either end). */
export function moveBy(rows, i, delta) {
  const j = i + delta;
  if (j < 0 || j >= rows.length) return rows;
  const next = [...rows];
  [next[i], next[j]] = [next[j], next[i]];
  return next;
}

/** A file path as its folder and its short name: `styles/ink/line.safetensors` -> {folder: "styles/ink/", base: "line"}. */
export function splitName(file) {
  const cut = Math.max(file.lastIndexOf("/"), file.lastIndexOf("\\"));
  const base = file.slice(cut + 1).replace(/\.(safetensors|ckpt|pt)$/i, "");
  return { folder: file.slice(0, cut + 1), base };
}

/** The files to offer: those not listed yet that contain every word of ``query`` (any case), at most ``limit``. */
export function offered(files, query, rows, limit = 200) {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const taken = new Set(rows.map((r) => r.file));
  return files.filter((f) => !taken.has(f) && words.every((w) => f.toLowerCase().includes(w))).slice(0, limit);
}
