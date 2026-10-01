/* ---------------------------------------------------------------------------
   editor/widgets.js -- the param controls, shared by the inspector and
   the node body.

   A value must look and behave identically wherever it is edited, so
   this module is the ONLY place a catalog port becomes a control:
     path_kind -> picker over the server's asset folder (+ upload),
     choices   -> select, bool -> checkbox, int/float -> number,
     list/dict/tuple -> JSON textarea, str/Path -> text.

   Commit contract:
     - commit(value) writes node.params (the key is deleted for empty,
       so the declared default applies server-side) and calls
       doc.changed("params", origin) with origin "inspector" |
       "canvas" -- editor.js uses it to decide who re-renders;
     - onRebuild() asks the CALLER to rebuild itself, fired for
       structural commits only: a visible_when gate moved, or a path
       picker must repopulate from a fresh catalog. The canvas always
       re-renders on any change, so only the inspector passes a real
       callback here.

   visible_when (Node.Port contract): [gate_param, value | [values]].
   An unset gate falls back to its declared default, so rows that ship
   visible stay visible. Value is preserved while a row is hidden.
   --------------------------------------------------------------------------- */

import { assetCatalog, assetKindFor, uploadAsset } from "./assets.js";

const JSON_TYPES = new Set(["list", "dict", "tuple"]);

/** Is this port's row shown for the node's current params? `siblings`
    is the class's input list (the gate's default lives on its port). */
export function rowVisible(node, port, siblings) {
  if (!port.visible_when) return true;
  const [gate, accepted] = port.visible_when;
  let cur = node.params[gate];
  if (cur === undefined || cur === null || cur === "") {
    const gatePort = (siblings || []).find((p) => p.name === gate);
    cur = gatePort ? gatePort.default : undefined;
  }
  return Array.isArray(accepted) ? accepted.includes(cur) : cur === accepted;
}

/** Can this commit move another row's visible_when gate (or require a
    picker to repopulate)? Then the caller must rebuild its form. */
function isStructural(doc, node, port) {
  if (port.path_kind) return true;
  const cls = doc.classOf(node);
  if (!cls) return false;
  return cls.inputs.some((p) => p.visible_when && p.visible_when[0] === port.name);
}

/**
 * Build the control for one port.
 * opts: {doc, node, port, origin, onNote, onRebuild}
 */
export function buildWidget({ doc, node, port, origin, onNote, onRebuild }) {
  const commit = (value, rebuild = false) => {
    if (value === undefined || value === null || value === "") delete node.params[port.name];
    else node.params[port.name] = value;
    doc.changed("params", origin);
    if ((rebuild || isStructural(doc, node, port)) && onRebuild) onRebuild();
  };
  const current = () => {
    const v = node.params[port.name];
    return v === undefined || v === null ? "" : String(v);
  };
  const note = onNote || (() => {});

  if (port.path_kind) return pathWidget({ port, current, commit, note });

  if (port.choices && port.choices.length) {
    const sel = document.createElement("select");
    sel.className = "cfg-input";
    const def = document.createElement("option");
    def.value = "";
    def.textContent =
      port.default !== null && port.default !== undefined && port.default !== ""
        ? `(default: ${port.default})`
        : "(default)";
    sel.appendChild(def);
    for (const c of port.choices) {
      const o = document.createElement("option");
      o.value = c;
      o.textContent = c;
      sel.appendChild(o);
    }
    sel.value = current();
    sel.title = port.doc || port.name;
    sel.addEventListener("change", () => commit(sel.value || undefined, true));
    return sel;
  }

  if (port.type === "bool") {
    const wrap = document.createElement("label");
    wrap.className = "param-label";
    wrap.style.justifySelf = "start";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = node.params[port.name] === true;
    box.addEventListener("change", () => {
      // store explicit false too: unchecking must override an earlier true
      node.params[port.name] = box.checked;
      doc.changed("params", origin);
      if (onRebuild) onRebuild(); // the true/false text (and any gate) moved
    });
    wrap.append(box, document.createTextNode(" " + (box.checked ? "true" : "false")));
    return wrap;
  }

  if (port.type === "int" || port.type === "float") {
    const input = document.createElement("input");
    input.className = "cfg-input";
    input.type = "number";
    if (port.type === "float") input.step = "any";
    input.value = current();
    if (port.default !== null && port.default !== undefined && input.value === "") {
      input.placeholder = String(port.default);
    }
    input.title = port.doc || port.name;
    input.addEventListener("change", () => {
      if (input.value === "") return commit(undefined);
      const n = port.type === "int" ? parseInt(input.value, 10) : parseFloat(input.value);
      if (Number.isNaN(n)) {
        input.value = "";
        note(`${port.name}: not a ${port.type}, value cleared.`, "warn");
        return commit(undefined);
      }
      commit(n);
    });
    return input;
  }

  if (JSON_TYPES.has(port.type)) {
    const input = document.createElement("textarea");
    input.className = "cfg-input";
    input.rows = 2;
    const raw0 = current();
    input.value = node.params[port.name] !== undefined && typeof node.params[port.name] === "object"
      ? JSON.stringify(node.params[port.name])
      : raw0;
    input.title = port.doc || `${port.name} (JSON)`;
    input.addEventListener("change", () => {
      const raw = input.value.trim();
      if (raw === "") return commit(undefined);
      try {
        commit(JSON.parse(raw), true);
      } catch {
        note(`${port.name}: invalid JSON -- keeping previous value.`, "warn");
        const v = node.params[port.name]; // untouched: show what was kept
        input.value = v === undefined || v === null
          ? ""
          : typeof v === "object" ? JSON.stringify(v) : String(v);
      }
    });
    return input;
  }

  // str / Path / anything else: text
  const input = document.createElement("input");
  input.className = "cfg-input";
  input.type = "text";
  input.value = current();
  if (port.default_repr !== null && port.default_repr !== undefined && input.value === "") {
    input.placeholder = port.default_repr;
  }
  input.title = port.doc || port.name;
  input.addEventListener("change", () => commit(input.value.trim() || undefined));
  return input;
}

/**
 * path_kind widget: choose from the server's folder (a select fed from
 * the cached /assets/{kind} catalog) with an upload button, or -- for
 * the Save-As kind (lora_output) -- a typed target path, since the file
 * does not exist yet. The server sandboxes every path; this side only
 * ever sends names.
 */
function pathWidget({ port, current, commit, note }) {
  const kind = assetKindFor(port.path_kind);
  const row = document.createElement("div");
  row.className = "prow";

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.hidden = true;
  const upload = document.createElement("button");
  upload.type = "button";
  upload.className = "pupload";
  upload.textContent = "\u2191"; // up arrow
  upload.hidden = true; // revealed when the catalog reports upload_supported
  upload.title = `Upload a file from this computer into the server's ${kind} folder`;

  let typed = null;
  let sel = null;
  if (port.path_kind === "lora_output") {
    // Save-As: the target may not exist yet, so free text + upload
    typed = document.createElement("input");
    typed.className = "cfg-input";
    typed.type = "text";
    typed.value = current();
    if (port.default_repr !== null && port.default_repr !== undefined && typed.value === "") {
      typed.placeholder = port.default_repr;
    }
    typed.title = port.doc || `${port.name} (save path, e.g. sub/name.safetensors)`;
    typed.addEventListener("change", () => commit(typed.value.trim() || undefined));
    row.appendChild(typed);
  } else {
    sel = document.createElement("select");
    sel.className = "cfg-input";
    const ph = document.createElement("option");
    ph.value = "";
    ph.textContent =
      port.default !== null && port.default !== undefined && port.default !== ""
        ? `(default: ${port.default})`
        : "(choose\u2026)";
    sel.appendChild(ph);
    sel.value = current();
    sel.title = port.doc || port.name;
    sel.addEventListener("change", () => commit(sel.value || undefined));
    row.appendChild(sel);
  }

  // capabilities come from the catalog -- both branches: the picker is
  // populated from it, and the upload button is revealed only when the
  // server says this kind accepts uploads (dataset: never)
  assetCatalog(kind)
    .then((cat) => {
      if (sel && sel.isConnected) {
        for (const o of cat.options || []) {
          const opt = document.createElement("option");
          opt.value = o.value;
          opt.textContent = o.label || o.value;
          sel.appendChild(opt);
        }
        const cur = current();
        if (cur && !cat.options.some((o) => o.value === cur)) {
          // the stored value isn't on the server: show it truthfully
          const opt = document.createElement("option");
          opt.value = cur;
          opt.textContent = `${cur} (not on server)`;
          sel.appendChild(opt);
        }
        sel.value = cur;
        if (cat.base_dir) sel.title = `${port.name} under ${cat.base_dir}`;
      }
      if (cat.upload_supported) upload.hidden = false;
    })
    .catch(() => {
      if (sel && sel.isConnected) {
        sel.disabled = true;
        sel.title = "asset list unavailable";
      }
    });

  upload.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const f = fileInput.files && fileInput.files[0];
    if (!f) return;
    const target = typed ? typed.value.trim() : ""; // Save-As honours the typed path
    upload.disabled = true;
    note(`Uploading ${f.name}\u2026`);
    uploadAsset(port.path_kind, f, target || undefined)
      .then((rel) => {
        note(`Uploaded ${f.name} \u2192 ${rel}`);
        commit(rel, true); // picker repopulates from the fresh catalog
      })
      .catch((err) => {
        // 409 is not a crash: the file is already there and we did not
        // touch it. Say what happened and how to proceed, rather than
        // showing the raw envelope code (docs 08 N-14).
        const hint = err && err.code === "asset_exists"
          ? " -- it already exists; use Save-As for a different name, or " +
            "delete it first to replace it"
          : "";
        note(
          `Upload failed: ${err && err.message ? err.message : err}${hint}`,
          "error",
        );
      })
      .finally(() => {
        upload.disabled = false;
        fileInput.value = "";
      });
  });
  row.append(upload, fileInput);
  return row;
}
