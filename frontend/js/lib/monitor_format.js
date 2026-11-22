/* How the monitor page words numbers and exports its session. Pure: no DOM or clock. */

export function fmt(v) {
  if (v === undefined || v === null) return "\u2014";
  return v >= 1 ? v.toFixed(4) : v >= 0.001 ? v.toFixed(5) : v.toExponential(2);
}

export function fmtMB(v) {
  if (v === undefined || v === null) return "\u2014";
  return v >= 1024 ? (v / 1024).toFixed(1) + " GB" : Math.round(v) + " MB";
}

export function fmtDuration(seconds) {
  if (!isFinite(seconds) || seconds < 0) return "\u2014";
  const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = Math.floor(seconds % 60);
  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`;
}

/** Rows of reports as CSV: columns = union of keys in first-seen order, missing keys empty -- never imputed. */
export function toCsv(rows) {
  const cols = [];
  for (const r of rows) {
    for (const k of Object.keys(r)) if (!cols.includes(k)) cols.push(k);
  }
  const esc = (v) => {
    if (v === undefined || v === null) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const lines = [cols.join(",")];
  for (const r of rows) lines.push(cols.map((c) => esc(r[c])).join(","));
  return lines.join("\n");
}
