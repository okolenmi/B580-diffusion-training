/* ---------------------------------------------------------------------------
   board/series.js -- live data for display blocks, as pure functions.

   A *stream* is one monitor-bus id (`node:<id>` or `graph:run`). Several
   blocks may watch the same stream (a node's loss chart, learning-rate chart
   and progress bar all read `node:trainer`), so the data lives in one
   `StreamState` per id and each block *derives* what it draws from it --
   nothing is plotted twice and nothing is kept per block.

   Frame contract (same as the old monitor page, backend/presentation/api/
   monitor.py): `{"type":"connected"}`, raw step reports (a dict with `step`,
   or `t` for graph-scope streams), `{"type":"clear"}`, `{"type":"run_end",
   step, cancelled}`. A frame with some other `type`, or with no x at all, is
   not telemetry and is ignored rather than plotted as a hole.

   Two rules carried over from the dashboard because they are the ones that
   stop a chart lying:
     * a key absent from a report is a **gap**, never zero;
     * a non-finite number (the server sends null + a marker) is a gap too.

   No DOM, no EventSource, no real clock: `applyFrame` takes the arrival time
   so tests drive it, and `decimate`/`progressOf` are plain functions.
   --------------------------------------------------------------------------- */

/** Series colours for keys discovered at run time (stable per key). */
export const PALETTE = [
  "#6c8cff", "#4caf50", "#ff9800", "#e0577d", "#26c6da",
  "#ab47bc", "#d4e157", "#8d6e63", "#78909c", "#ffca28",
];

const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);

/* ---- one stream's state -------------------------------------------------- */

export class StreamState {
  /** @param {{maxRows?: number}} [opts] rows kept (oldest dropped past it) */
  constructor({ maxRows = 50000 } = {}) {
    this.maxRows = maxRows;
    this.reset();
    this.conn = "connecting"; // connecting | live | disconnected
    this.bad = 0; // frames that were not JSON
  }

  /** What a `clear` frame does: forget the run, keep the connection. */
  reset() {
    this.rows = [];      // [{frame, at}]  at = arrival ms
    this.dropped = 0;    // rows discarded to stay under maxRows
    this.last = {};      // key -> {value, x}: latest present value (gap-proof)
    this.keys = new Set();  // numeric keys seen, in first-seen order
    this.t0 = null;      // first `t` after a clear: x=t is shown relative to it
    this.runEnd = null;  // null while running, {step, cancelled} after run_end
    this.latestFrame = null;
  }

  /** Parse and apply one SSE `data` string. Never throws. */
  applyRaw(text, at = 0) {
    let frame;
    try { frame = JSON.parse(text); } catch { this.bad++; return { kind: "bad" }; }
    return this.applyFrame(frame, at);
  }

  applyFrame(frame, at = 0) {
    if (frame === null || typeof frame !== "object" || Array.isArray(frame)) {
      this.bad++;
      return { kind: "bad" };
    }
    if (frame.type === "connected") {
      // The server sends this at the start of EVERY (re)connect and then
      // replays the stream's history (state frames included), so whatever
      // is held now would be delivered a second time: start from empty.
      this.reset();
      this.conn = "live";
      return { kind: "connected" };
    }
    if (frame.type === "clear") { this.reset(); return { kind: "clear" }; }
    if (frame.type === "run_end") {
      this.runEnd = { step: frame.step ?? null, cancelled: !!frame.cancelled };
      return { kind: "run_end" };
    }
    if (frame.type !== undefined) return { kind: "ignored" };
    if (num(frame.step) === null && num(frame.t) === null) return { kind: "ignored" };

    if (this.runEnd) this.runEnd = null; // a new report after run_end: new run
    if (this.t0 === null && num(frame.t) !== null) this.t0 = frame.t;
    this.rows.push({ frame, at });
    if (this.rows.length > this.maxRows) {
      const cut = this.rows.length - this.maxRows;
      this.rows.splice(0, cut);
      this.dropped += cut;
    }
    this.latestFrame = frame;
    const x = { step: frame.step, t: this.t0 === null ? null : frame.t - this.t0 };
    for (const [key, value] of Object.entries(frame)) {
      const n = num(value);
      if (n !== null) {
        this.keys.add(key);
        this.last[key] = { value: n, x };
      } else if (typeof value === "string") {
        this.last[key] = { value, x };
      }
    }
    return { kind: "point" };
  }

  setConn(state) { this.conn = state; }
}

/* ---- x axis -------------------------------------------------------------- */

/** x of a frame for a module: the step, or seconds since the run began. */
export function xOf(state, frame, xKey) {
  if (xKey === "t") {
    const t = num(frame.t);
    return t === null || state.t0 === null ? null : t - state.t0;
  }
  return num(frame.step);
}

/* ---- series discovery ---------------------------------------------------- */

/** Shell-style glob (`*`, `?`, `[abc]`) as the backend's `pattern` means it
    (Python `fnmatch`). Case-sensitive. */
export function globToRegExp(pattern) {
  let out = "";
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i];
    if (c === "*") out += ".*";
    else if (c === "?") out += ".";
    else if (c === "[") {
      const end = pattern.indexOf("]", i + 2);
      if (end === -1) out += "\\[";
      else {
        let body = pattern.slice(i + 1, end);
        if (body[0] === "!") body = "^" + body.slice(1);
        out += `[${body.replace(/\\/g, "\\\\")}]`;
        i = end;
      }
    } else out += c.replace(/[.+^${}()|\\/]/g, "\\$&");
  }
  return new RegExp(`^${out}$`);
}

/** Pretty label for a discovered key (`fwd_ms` -> `fwd`). */
export function labelFromKey(key) {
  return key.replace(/_ms$/, "").replace(/_mb$/, "").replace(/_/g, " ");
}

/**
 * The series a module draws right now: its declared ones, plus -- when it
 * has a `pattern` -- every numeric key the stream has shown that matches and
 * is not excluded. Discovered colours come from `PALETTE` by the key's
 * position in the stream's first-seen order, so a series keeps its colour
 * for the whole run however many appear later.
 */
export function seriesFor(module, state) {
  const out = (module.series || []).map((s) => ({ ...s, label: s.label || s.key }));
  const have = new Set(out.map((s) => s.key));
  if (module.pattern) {
    const re = globToRegExp(module.pattern);
    const excluded = new Set(module.exclude || []);
    let i = 0;
    for (const key of state.keys) {
      if (re.test(key) && !excluded.has(key) && !have.has(key)) {
        out.push({
          key, label: labelFromKey(key),
          color: PALETTE[(i + out.length) % PALETTE.length], unit: "",
        });
        have.add(key);
      }
      i++;
    }
  }
  return out.map((s, idx) => ({ ...s, color: s.color || PALETTE[idx % PALETTE.length] }));
}

/* ---- chart rows ---------------------------------------------------------- */

/** `[{x, values: {key: number|null}}]` for the chart, one per frame that has
    an x. `keys` limits which series are copied. */
export function chartRows(state, module, keys) {
  const rows = [];
  for (const { frame } of state.rows) {
    const x = xOf(state, frame, module.x || "step");
    if (x === null) continue;
    const values = {};
    for (const k of keys) values[k] = num(frame[k]);
    rows.push({ x, values });
  }
  return rows;
}

/**
 * Cut `rows` to about `maxRows` for drawing while keeping what a reader
 * looks for: in each bucket of consecutive rows, the row holding each
 * series' minimum and the row holding its maximum survive, so a one-step
 * loss spike is still in the picture; the first and last rows always do
 * (the newest point is where the eye goes). Kept rows are *original* rows,
 * in order -- no averaging, no invented values.
 */
export function decimate(rows, maxRows, keys) {
  if (rows.length <= maxRows || maxRows < 4) return rows;
  const k = Math.max(1, keys.length);
  const buckets = Math.max(1, Math.floor((maxRows - 2) / (2 * k)));
  const size = rows.length / buckets;
  const keep = new Set([0, rows.length - 1]);
  for (let b = 0; b < buckets; b++) {
    const from = Math.floor(b * size);
    const to = Math.min(rows.length, Math.floor((b + 1) * size));
    for (const key of keys) {
      let lo = -1, hi = -1;
      for (let i = from; i < to; i++) {
        const v = rows[i].values[key];
        if (v === null || v === undefined) continue;
        if (lo < 0 || v < rows[lo].values[key]) lo = i;
        if (hi < 0 || v > rows[hi].values[key]) hi = i;
      }
      if (lo >= 0) { keep.add(lo); keep.add(hi); }
    }
  }
  return [...keep].sort((a, b) => a - b).map((i) => rows[i]);
}

/** Every record the stream holds as CSV (what the old Monitor page exported):
    one row per report, a column per key in first-seen order, a missing or
    non-finite value left EMPTY -- a gap stays a gap, it is never written as 0. */
export function rowsToCsv(rows) {
  const keys = [];
  const seen = new Set();
  for (const { frame } of rows) {
    for (const k of Object.keys(frame)) {
      if (!seen.has(k)) { seen.add(k); keys.push(k); }
    }
  }
  const cell = (v) => {
    if (typeof v === "number") return Number.isFinite(v) ? String(v) : "";
    if (typeof v === "string") return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
    return "";
  };
  const lines = [keys.join(",")];
  for (const { frame } of rows) lines.push(keys.map((k) => cell(frame[k])).join(","));
  return lines.join("\n") + "\n";
}

/** Reference lines: the latest value of `reference_key`, if the stream has
    shown one. */
export function referenceLines(module, state) {
  if (!module.reference_key) return [];
  const last = state.last[module.reference_key];
  if (!last || typeof last.value !== "number") return [];
  return [{ value: last.value, label: labelFromKey(module.reference_key), color: "#ff9800" }];
}

/* ---- latest-value kinds -------------------------------------------------- */

/** scalar: the newest present value of the module's first series, with how
    old it is -- `stale` when the newest frame did not carry it. */
export function scalarOf(module, state) {
  const series = seriesFor(module, state)[0];
  if (!series) return null;
  const last = state.last[series.key];
  if (!last || typeof last.value !== "number") return null;
  const newest = state.latestFrame;
  const stale = !newest || num(newest[series.key]) === null;
  return { series, value: last.value, x: last.x, stale };
}

/** bars: every series' newest present value, in series order. */
export function barsOf(module, state) {
  return seriesFor(module, state)
    .map((s) => {
      const last = state.last[s.key];
      return last && typeof last.value === "number" ? { ...s, value: last.value } : null;
    })
    .filter(Boolean);
}

export function textOf(module, state) {
  const series = (module.series || [])[0];
  if (!series) return null;
  const last = state.last[series.key];
  return last === undefined ? null : String(last.value);
}

/**
 * progress: `value` of `total` with a rate and an ETA from the last few
 * reports' arrival times (`window` rows). Rate is steps per second of
 * *wall-clock*, because that is what the person waits in; the ETA is null
 * until there are two reports at least a second apart, or when the run is
 * not advancing -- never a guess from one sample.
 */
export function progressOf(module, state, { window = 20 } = {}) {
  const vk = module.value_key, tk = module.total_key;
  const v = state.last[vk], t = state.last[tk];
  if (!v || typeof v.value !== "number") return null;
  const total = t && typeof t.value === "number" && t.value > 0 ? t.value : null;
  const out = {
    value: v.value, total,
    fraction: total ? Math.min(1, v.value / total) : null,
    rate: null, eta: null,
    done: !!state.runEnd,
  };
  const recent = state.rows
    .filter((r) => num(r.frame[vk]) !== null)
    .slice(-window);
  if (recent.length >= 2) {
    const a = recent[0], b = recent[recent.length - 1];
    const dt = (b.at - a.at) / 1000;
    const dv = b.frame[vk] - a.frame[vk];
    // Under a second apart means the reports arrived in a burst -- a page
    // opened late is replayed the whole history at once -- and a rate from
    // that would be nonsense ("19000/s"), so there is none.
    if (dt >= 1 && dv > 0) {
      out.rate = dv / dt;
      if (total && !out.done) out.eta = Math.max(0, (total - v.value) / out.rate);
    }
  }
  return out;
}

/** "1h 02m", "4m 05s", "12s" -- for an ETA or an elapsed time. */
export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(r).padStart(2, "0")}s`;
  return `${r}s`;
}

/** A readable number for a scalar: integers plain, big values with
    separators, small ones to 4 significant digits. */
export function formatValue(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  if (a >= 100) return v.toFixed(1);
  if (a >= 10) return v.toFixed(2);
  if (a >= 1) return v.toFixed(3);
  if (a === 0) return "0";
  return v.toPrecision(3);
}
