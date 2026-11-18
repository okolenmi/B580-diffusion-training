/* ---------------------------------------------------------------------------
   views/settings.js -- Settings: folders, the Python environment (with a live check), interface, housekeeping and the server.

   Each section is a class that owns its panel and its own requests, so a failing source breaks only its panel. The side menu
   follows the scroll position, and `/settings#environment` opens at that section (the start page links there).
   --------------------------------------------------------------------------- */

import { api } from "../api.js";
import { h, mount } from "../lib/dom.js";
import { checkEnvironment, venvFolder } from "../lib/environment_check.js";
import { errText } from "../lib/errors.js";
import { fmtRel } from "../lib/format.js";
import { nodeHealth } from "../lib/home_model.js";
import { logError } from "../lib/log.js";
import { followScroll } from "../lib/side_nav.js";
import { SettingsForm } from "../lib/settings_form.js";

const CONSOLE_KEY = "shell.console.v1";
const quiet = (promise) => promise.catch((err) => { logError(err); return null; });

function panel(id, title, desc, body, foot = null) {
  return h("section", { class: "panel", id },
    h("div", { class: "panel-head" }, h("div", {}, h("div", { class: "panel-title" }, title), desc ? h("div", { class: "panel-desc" }, desc) : null)),
    h("div", { class: "panel-body" }, body),
    foot ? h("div", { class: "panel-foot" }, foot) : null);
}

const row = (key, value, action = null, note = null) => h("div", { class: "kv-row" },
  h("span", { class: "kv-key" }, key), h("span", { class: "kv-value" }, value), action ? h("span", {}, action) : h("span", {}), note ? h("span", { class: "kv-note" }, note) : null);

/* ---- Folders ------------------------------------------------------------- */

class FoldersSection {
  constructor() {
    this.form = new SettingsForm([
      { key: "models_dir", label: "Models folder", hint: "Checkpoints are read from <folder>/checkpoints and trained LoRAs are saved to <folder>/loras. A ComfyUI `models` folder works as it is." },
      { key: "checkpoints_dir", label: "Checkpoints folder", hint: "Only when checkpoints live somewhere other than <models folder>/checkpoints." },
      { key: "loras_dir", label: "LoRA folder", hint: "Only when trained LoRAs should go somewhere other than <models folder>/loras." },
      { key: "comfy_dir", label: "ComfyUI folder", hint: "Optional. Lets the app reuse a ComfyUI installation's models; nothing needs ComfyUI to run." },
    ], () => this.refreshCounts());
    this.counts = h("div", { class: "kv" });
    this.view = null;
  }

  async refreshCounts() {
    const [checkpoints, loras] = await Promise.all([quiet(api("/catalog/models/checkpoints")), quiet(api("/catalog/models/loras"))]);
    const resolved = this.view ? this.view.resolved : {};
    const found = (key, list, noun) => {
      if (!resolved[key]) return h("span", { class: "chip chip-bad" }, "folder missing: saving creates it");
      return list ? h("span", { class: `chip ${list.files.length ? "chip-ok" : ""}` }, `${list.files.length} ${noun}${list.files.length === 1 ? "" : "s"} found`) : null;
    };
    mount(this.counts,
      row("Checkpoints are read from", h("span", { class: "mono" }, resolved.checkpoints_dir || "not set"), found("checkpoints_dir", checkpoints, "checkpoint")),
      row("LoRAs are saved to", h("span", { class: "mono" }, resolved.loras_dir || "not set"), found("loras_dir", loras, "LoRA")));
  }

  load(view) {
    this.view = view;
    this.form.fill(view);
    this.refreshCounts();
  }

  render() {
    const f = this.form;
    return panel("folders", "Folders", "Where models are read from and trained LoRAs are written. Leave a field empty to detect it.",
      [f.field("models_dir"), this.counts,
        h("details", { class: "advanced" }, h("summary", {}, "Use different folders for checkpoints, LoRAs or ComfyUI"),
          f.field("checkpoints_dir"), f.field("loras_dir"), f.field("comfy_dir"))],
      [f.saveButton, f.revertButton, f.status]);
  }
}

/* ---- Environment --------------------------------------------------------- */

class EnvironmentSection {
  constructor() {
    this.form = new SettingsForm([
      { key: "venv_python", label: "Python interpreter", hint: "The python file of the virtualenv that trains (venv/bin/python, venv\\Scripts\\python.exe on Windows). Training runs use it at once; the server itself moves into it the next time it starts." },
    ], (view) => { this.settings = view; this.check(); });
    this.settings = null;
    this.report = h("div", { class: "kv" });
    this.checkButton = h("button", { class: "btn btn-secondary", type: "button", onclick: () => this.check() }, "Check again");
    this.tokenizers = null;
    this.health = null;
    this.catalog = null;
  }

  load(view) {
    this.settings = view;
    this.form.fill(view);
    return this.check();
  }

  async check() {
    const python = this.settings ? this.settings.resolved.venv_python : null;
    this.checkButton.disabled = true;
    mount(this.report, row("Checking", h("span", { class: "say" }, "PyTorch is being asked in that environment…"), h("span", { class: "dot dot-busy" })));
    const [env, health, catalog, tokenizers] = await Promise.all([
      checkEnvironment(python), quiet(api("/health")), quiet(api("/catalog")), quiet(api("/installer/tokenizers"))]);
    Object.assign(this, { health, catalog, tokenizers });
    this.paint(python, env);
    this.checkButton.disabled = false;
  }

  paint(python, env) {
    const nodes = nodeHealth(this.catalog);
    const serverFolder = this.health ? venvFolder(this.health.interpreter) : null;
    const same = serverFolder && env.folder ? serverFolder === env.folder : null;
    const serverRow = this.health
      ? row("Server runs in", h("span", { class: "mono" }, this.health.interpreter),
        same == null ? null : h("span", { class: `chip ${same ? "chip-ok" : "chip-bad"}` }, same ? "same environment" : "different environment"),
        same === false ? "The nodes load PyTorch inside the server, so they cannot until the server runs in the chosen environment. Stop it and start it with the launcher." : null)
      : row("Server runs in", "unknown");
    mount(this.report,
      row("Interpreter in use", h("span", { class: "mono" }, python || "none"),
        this.settings && this.settings.exists.venv_python === false ? h("span", { class: "chip chip-bad" }, "file does not exist") : null, env.folder ? null : env.reason),
      serverRow,
      row("PyTorch", env.torchVersion || "not found", h("span", { class: `chip ${env.torchVersion ? "chip-ok" : "chip-bad"}` }, env.torchVersion ? "found" : "missing"), env.torchVersion ? null : env.reason),
      row("Graphics card", env.present ? `${env.name}, ${Math.round(env.totalMemoryMb).toLocaleString()} MB` : "not seen by PyTorch",
        h("span", { class: `chip ${env.present ? "chip-ok" : "chip-bad"}` }, env.present ? "ready" : "not ready")),
      row("Node library", nodes.state === "ok" ? "all modules loaded" : nodes.state === "bad" ? `${nodes.failed} modules failed to load` : "unknown",
        h("span", { class: `chip ${nodes.state === "ok" ? "chip-ok" : nodes.state === "bad" ? "chip-bad" : ""}` }, nodes.state), nodes.reason),
      (this.tokenizers || []).map((set) => this.tokenizerRow(set)));
  }

  tokenizerRow(set) {
    const button = set.complete ? null : h("button", { class: "btn btn-secondary btn-small", type: "button", onclick: async (e) => {
      e.target.disabled = true; e.target.textContent = "Downloading…";
      try { await api(`/installer/tokenizers/${set.key}`, { method: "POST" }); } catch (err) { logError(err); }
      this.check();
    } }, "Download");
    return row(set.label, `for ${set.needed_for}`, button || h("span", { class: "chip chip-ok" }, "present"), null);
  }

  render() {
    const f = this.form;
    return panel("environment", "Python environment", "Training runs in a virtualenv with the Intel XPU build of PyTorch.",
      [f.field("venv_python"), h("div", { class: "panel-sub-title" }, ""), this.report,
        h("div", { class: "callout", style: "margin-top:14px" },
          "To rebuild a broken environment, delete its folder: setup opens again by itself and rebuilds it. ", h("a", { href: "/help#reinstall" }, "How that works"))],
      [f.saveButton, f.revertButton, f.status, h("span", { class: "spacer" }), this.checkButton]);
  }
}

/* ---- Interface ----------------------------------------------------------- */

class InterfaceSection {
  render() {
    const say = h("span", { class: "say" });
    const reset = h("button", { class: "btn btn-secondary", type: "button", onclick: () => {
      try { localStorage.removeItem(CONSOLE_KEY); say.textContent = "Done. The console returns to its default place when a page opens."; } catch { say.textContent = "This browser does not let the page store anything."; }
    } }, "Reset the console window");
    return panel("interface", "Interface", "How the app looks in this browser.",
      [h("div", { class: "field-label", style: "margin-bottom:8px" }, "Theme"),
        h("div", { class: "theme-options", role: "radiogroup", "aria-label": "Design theme" },
          h("label", { class: "theme-option" }, h("input", { type: "radio", name: "theme", value: "dark", checked: true }),
            h("span", { class: "theme-swatch theme-swatch-dark", "aria-hidden": "true" }), h("span", { class: "theme-name" }, "Dark"), h("span", { class: "chip chip-accent" }, "current")),
          h("label", { class: "theme-option theme-option-disabled" }, h("input", { type: "radio", name: "theme", value: "light", disabled: true }),
            h("span", { class: "theme-swatch theme-swatch-light", "aria-hidden": "true" }), h("span", { class: "theme-name" }, "Light"), h("span", { class: "chip" }, "planned"))),
        h("p", { class: "field-hint", style: "margin-top:10px" }, "Every color comes from shared tokens, so Light is a token set, not a redesign. It is not built yet."),
        h("div", { class: "kv", style: "margin-top:12px" },
          row("Console window", "The log window in the corner remembers its position and size in this browser.", reset)),
        h("div", { style: "margin-top:8px" }, say)]);
  }
}

/* ---- Maintenance --------------------------------------------------------- */

class MaintenanceSection {
  constructor() {
    this.body = h("div", { class: "kv" });
    this.note = h("div", { style: "margin-top:10px" });
    this.armed = false;
  }

  async load() {
    const [runs, graphs, datasets] = await Promise.all([
      quiet(api("/graphs/executions?limit=500")), quiet(api("/graphs/library")), quiet(api("/datasets"))]);
    const active = runs ? runs.executions.filter((r) => ["planned", "queued", "running"].includes(r.status)).length : 0;
    const say = h("span", { class: "say" });
    const wipe = h("button", { class: `btn ${this.armed ? "btn-danger" : "btn-secondary"}`, type: "button", disabled: !runs || !runs.count || active > 0, onclick: async () => {
      if (!this.armed) { this.armed = true; wipe.textContent = `Really delete ${runs.count} runs?`; wipe.classList.replace("btn-secondary", "btn-danger"); return; }
      try { const done = await api("/graphs/executions", { method: "DELETE" }); this.armed = false; say.textContent = `Deleted ${done.deleted} runs.`; } catch (err) { say.textContent = errText(err); }
      this.load();
    } }, this.armed && runs ? `Really delete ${runs.count} runs?` : "Delete run history");
    mount(this.body,
      row("Run history", runs ? `${runs.count} run${runs.count === 1 ? "" : "s"}${active ? `, ${active} still active` : ""}` : "could not be read", wipe,
        active ? "Stop the active runs first; deleting history would stop them." : "Removes every finished run with its charts. Trained LoRAs are files in your LoRA folder and stay."),
      row("Saved graphs", graphs ? `${graphs.count}` : "could not be read", h("a", { class: "btn btn-secondary btn-link", href: "/graph" }, "Open editor")),
      row("Datasets", datasets ? `${datasets.datasets.length}` : "could not be read", h("a", { class: "btn btn-secondary btn-link", href: "/datasets" }, "Open datasets")),
      );
    mount(this.note, say);
  }

  render() {
    return panel("maintenance", "Housekeeping", "What the app keeps for you, and how to clear it.", [this.body, this.note]);
  }
}

/* ---- Server -------------------------------------------------------------- */

class ServerSection {
  constructor() { this.body = h("div", { class: "kv" }); }

  async load() {
    const health = await quiet(api("/health"));
    if (!health) { mount(this.body, h("div", { class: "callout callout-bad" }, "The server did not answer.")); return; }
    mount(this.body,
      row("Version", health.version),
      row("Started", health.started_at ? `${new Date(health.started_at).toLocaleString()} (${fmtRel(Date.parse(health.started_at))})` : "unknown"),
      row("Python", `${health.python || "?"}`, null),
      row("Address", h("span", { class: "mono" }, window.location.host)),
      row("Start id", h("span", { class: "mono" }, (health.boot_id || "").slice(0, 8)), null, "Changes every time the server restarts."));
  }

  render() {
    return panel("server", "Server", "The process serving this page.", [this.body,
      h("p", { class: "field-hint", style: "margin-top:12px" }, "To restart the server, stop it and start it with the launcher (run_server_linux.sh, or run_server_windows.bat).")]);
  }
}

/* ---- page ---------------------------------------------------------------- */

const SECTIONS = [["folders", "Folders"], ["environment", "Python environment"], ["interface", "Interface"], ["maintenance", "Housekeeping"], ["server", "Server"]];

async function boot() {
  const folders = new FoldersSection();
  const environment = new EnvironmentSection();
  const maintenance = new MaintenanceSection();
  const server = new ServerSection();
  const nav = document.getElementById("settings-nav");
  mount(nav, h("div", { class: "side-nav-title" }, "Sections"), SECTIONS.map(([id, label]) => h("a", { href: `#${id}` }, label)));
  mount(document.getElementById("settings-root"), folders.render(), environment.render(), new InterfaceSection().render(), maintenance.render(), server.render());
  followScroll(nav, SECTIONS.map(([id]) => id));
  if (window.location.hash) document.getElementById(window.location.hash.slice(1))?.scrollIntoView();

  const view = await api("/settings");
  folders.load(view);
  maintenance.load();
  server.load();
  await environment.load(view);
}

boot().catch((err) => {
  logError(err);
  mount(document.getElementById("settings-root"), h("div", { class: "error-box" }, errText(err)));
});
