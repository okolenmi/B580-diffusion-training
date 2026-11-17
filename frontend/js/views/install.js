/* ---------------------------------------------------------------------------
   install.js -- first-run setup (/setup).

   Five steps, each answering one question the next one depends on:

     1. Graphics card   which Intel GPU the system lists (torch may not exist yet)
     2. Environment     a new virtualenv in the project, or an existing one (its folder);
                        the packages found there are shown before anything is changed
     3. Install         fills the environment; pip output streams into a log that can be scrolled
     4. Prompt files    tokenizer vocabulary files, downloaded into the project's assets folder
     5. Models folder   default ./models; a ComfyUI `models` folder works as is (same layout)

   ComfyUI is never required. The page only appears while setup is unfinished; the server refuses installs and
   applies after that (`installer_not_allowed`), so a stale tab cannot re-point anything.
   --------------------------------------------------------------------------- */

import { api, ApiError } from "../api.js";
import { el, h, mount, showMessage } from "../lib/dom.js";
import { errText } from "../lib/errors.js";
import { log, logError } from "../lib/log.js";

const STEPS = ["Graphics card", "Environment", "Install", "Prompt files", "Models folder"];

const wizard = {
  step: 1,
  state: null,       // GET /installer/state
  devices: null,     // GET /installer/devices
  target: "new",     // "new" | "existing"
  venvDir: "",
  report: null,      // POST /installer/environment
  python: null,      // the interpreter the trainer will use, once known
  verify: null,      // POST /installer/environment/verify
  tokenizers: null,  // GET /installer/tokenizers
  torch: "tested",    // "tested" | "latest" | "custom"
  torchCustom: "",
};

const size = (mb) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`);
const root = () => el("setup-root");

function frame(title, ...body) {
  mount(root(),
    h("ol", { class: "setup-steps" }, STEPS.map((name, i) =>
      h("li", { class: i + 1 === wizard.step ? "is-current" : i + 1 < wizard.step ? "is-done" : "" }, name))),
    h("h2", { class: "setup-step-title" }, title),
    ...body);
}

function actions(...buttons) {
  return h("div", { class: "setup-actions" }, ...buttons);
}

function button(label, onclick, { primary = false, disabled = false } = {}) {
  return h("button", { class: primary ? "btn btn-primary" : "btn", type: "button", onclick, disabled }, label);
}

const message = () => h("p", { class: "setup-error", id: "setup-msg", hidden: true });

function show(n) {
  wizard.step = n;
  saveWizard();
  const render = [null, stepDevices, stepEnvironment, null, stepTokenizers, stepModels][n];
  if (render) render();
}

/* Setup was reopened because the environment chosen earlier no longer exists; the rest of the installation is already configured. */
const repairing = () => Boolean(wizard.state && wizard.state.missing_environment);

function finishRepair() {
  try { localStorage.removeItem(SAVED); } catch { /* only a convenience */ }
  window.location.replace("/help");
}

/* ---- 1. graphics card ---------------------------------------------------- */

async function stepDevices() {
  if (!wizard.devices) {
    frame("Graphics card", h("p", { class: "setup-note text-dim" }, "Looking for graphics cards…"));
    try {
      wizard.devices = await api("/installer/devices");
    } catch (err) {
      wizard.devices = { enumerated: false, backend: "xpu", devices: [], reason: errText(err) };
    }
  }
  const { devices, enumerated, reason } = wizard.devices;
  const body = !enumerated || !devices.length
    ? h("div", { class: "setup-device is-unknown" },
        h("div", {},
          h("div", {}, "The system did not list an Intel graphics card."),
          h("div", { class: "text-dim" }, "Setup can continue; the card is checked again with PyTorch once the environment is installed."),
          reason ? h("div", { class: "setup-device-detail text-dim" }, reason) : null))
    : h("div", {}, devices.map((d) => h("div", { class: "setup-device is-ok" },
        h("span", { class: "setup-device-state", "aria-hidden": "true" }, "✓"),
        h("div", {},
          h("div", {}, d.name || `Device ${d.index}`, d.total_memory_mb ? ` — ${Math.round(d.total_memory_mb).toLocaleString()} MB` : ""),
          d.note ? h("div", { class: "text-dim" }, d.note) : null))));
  frame("Graphics card",
    repairing() ? h("p", { class: "setup-error" },
      `The environment chosen in the earlier setup is gone: ${wizard.state.missing_environment}. Rebuild it here; your folders and settings are kept.`) : null,
    h("p", { class: "setup-note text-dim" },
      "Training uses an Intel GPU through PyTorch's XPU build. This is what the system lists; the next steps install and verify PyTorch."),
    body,
    actions(button("Continue", () => show(2), { primary: true })));
}

/* ---- 2. environment ------------------------------------------------------ */

/* The choices that survive a reload, so a refreshed tab resumes instead of starting over. The install itself is the server's. */
const SAVED = "setup.wizard.v1";

function loadWizard() {
  try {
    const saved = JSON.parse(localStorage.getItem(SAVED) || "{}");
    for (const key of ["target", "venvDir", "torch", "torchCustom", "python"]) if (typeof saved[key] === "string") wizard[key] = saved[key];
    if (Number.isInteger(saved.step)) wizard.step = saved.step;
  } catch (err) {
    logError("installer: saved choices unreadable", err);
  }
}

function saveWizard() {
  try {
    const { target, venvDir, torch, torchCustom, python, step } = wizard;
    localStorage.setItem(SAVED, JSON.stringify({ target, venvDir, torch, torchCustom, python, step }));
  } catch (err) {
    logError("installer: could not save choices", err);
  }
}

function packageTable(report) {
  return h("div", { class: "setup-packages" }, report.packages.map((p) => h("div", { class: `setup-pkg ${p.installed_version ? "is-ok" : "is-missing"}` },
    h("span", { class: "setup-pkg-state", "aria-hidden": "true" }, p.installed_version ? "✓" : "○"),
    h("span", { class: "setup-pkg-name" }, p.name),
    h("span", { class: "setup-pkg-version text-dim" }, p.installed_version || (p.approx_mb ? `will install, ~${size(p.approx_mb)} on disk` : "will install")),
    h("span", { class: "setup-pkg-why text-dim" }, p.why))));
}

/* Which torch to install, only when the environment has none. The tested release is the default because it is the one the
   trainer's speed and memory numbers come from; the others exist for people who need a different build. */
function torchChoice(report) {
  if (!report.to_install.includes("torch")) return null;
  const option = (id, title, blurb) => h("label", { class: `setup-option${wizard.torch === id ? " is-picked" : ""}` },
    h("input", { type: "radio", name: "setup-torch", checked: wizard.torch === id, onchange: () => { wizard.torch = id; saveWizard(); stepEnvironment(false); } }),
    h("span", { class: "setup-option-body" },
      h("span", { class: "setup-option-title" }, title),
      h("span", { class: "setup-option-blurb" }, blurb)));
  return h("fieldset", { class: "setup-options" },
    h("legend", {}, "Which PyTorch (Intel XPU build)?"),
    option("tested", `${report.tested_torch} (tested)`, "The release the trainer is tested with on the Arc B580. Recommended."),
    option("latest", "The newest on the Intel XPU index", "Not tested with this trainer; may change speed or memory use."),
    option("custom", "A specific version",
      "For example 2.11.0. The install fails with pip's message if the Intel XPU index does not have it."),
    wizard.torch === "custom"
      ? h("input", { class: "setup-input", type: "text", id: "setup-torch-version", value: wizard.torchCustom, placeholder: report.tested_torch,
          oninput: (e) => { wizard.torchCustom = e.target.value.trim(); saveWizard(); } })
      : null);
}

function torchValue() {
  return wizard.torch === "custom" ? (wizard.torchCustom || "tested") : wizard.torch;
}

function reportSummary(report) {
  if (report.error) return h("p", { class: "setup-error" }, report.error);
  const missing = report.to_install.length;
  return h("div", {},
    report.python_version ? h("p", { class: "setup-note text-dim" }, `Python ${report.python_version} at ${report.interpreter}; ${report.installed_count} packages installed.`) : null,
    report.problems.map((text) => h("p", { class: "setup-error" }, text)),
    report.discards.map((what) => h("p", { class: "setup-note" }, `Note: ${what}.`)),
    packageTable(report),
    h("p", { class: "setup-note text-dim" }, missing
      ? `${missing} package${missing === 1 ? "" : "s"} to install (about ${size(report.download_mb)} on disk). Everything already in the environment is pinned and left as it is.`
      : "Everything this project needs is already there."));
}

async function inspect() {
  showMessage("setup-msg", "");
  try {
    wizard.report = await api("/installer/environment", { method: "POST", body: { target: wizard.target, venv_dir: wizard.venvDir || null } });
  } catch (err) {
    wizard.report = null;
    showMessage("setup-msg", errText(err));
    return;
  }
  stepEnvironment(false);
}

async function stepEnvironment(fresh = true) {
  if (fresh && wizard.target === "new") {
    wizard.report = null;
    try { wizard.report = await api("/installer/environment", { method: "POST", body: { target: "new" } }); } catch (err) { logError("installer: environment", err); }
  }
  const pick = (target, title, blurb) => h("label", { class: `setup-option${wizard.target === target ? " is-picked" : ""}` },
    h("input", { type: "radio", name: "setup-target", checked: wizard.target === target, onchange: () => { wizard.target = target; wizard.report = null; saveWizard(); stepEnvironment(true); } }),
    h("span", { class: "setup-option-body" },
      h("span", { class: "setup-option-title" }, title),
      h("span", { class: "setup-option-blurb" }, blurb)));
  const report = wizard.report;
  const ready = report && !report.error && report.usable;
  const nothing = ready && !report.to_install.length;
  frame("Python environment",
    h("p", { class: "setup-note text-dim" }, "Training runs in a Python virtualenv. Choose one; nothing is changed until you press Install."),
    h("fieldset", { class: "setup-options" },
      pick("new", "A new virtualenv in the project folder", "Created at ./venv with the packages below; a venv already there is completed. Delete the folder to undo it."),
      pick("existing", "An existing virtualenv", "Give the folder of a virtualenv you already have (ComfyUI's works). Its packages are checked first and never changed.")),
    wizard.target === "existing"
      ? h("div", { class: "setup-field" },
          h("span", { class: "setup-field-label" }, "Virtualenv folder"),
          h("input", { class: "setup-input", type: "text", id: "setup-venv", value: wizard.venvDir, placeholder: "/path/to/venv",
            oninput: (e) => { wizard.venvDir = e.target.value; saveWizard(); },
            onkeydown: (e) => { if (e.key === "Enter") inspect(); } }),
          h("span", { class: "setup-field-hint text-dim" }, "The folder that holds bin/python (Scripts\\python.exe on Windows)."),
          actions(button("Check packages", inspect)))
      : null,
    message(),
    report ? reportSummary(report) : null,
    report && !report.error ? torchChoice(report) : null,
    actions(
      button("Back", () => show(1)),
      nothing
        ? button("Use this environment", () => useEnvironment(report), { primary: true })
        : button(ready && report.to_install.length ? `Install ${report.to_install.length} packages` : "Install", startInstall,
            { primary: true, disabled: !ready })));
}

function useEnvironment(report) {
  adoptEnvironment(report.interpreter);
}

/* Store the chosen interpreter and move the server into it. After an install the server restarts even when it already runs in that
   environment: a running process keeps what it imported before the packages existed, which is how "Installed" came to sit beside a
   server that could not see torch. */
async function adoptEnvironment(python, { installed = false } = {}) {
  wizard.python = python;
  wizard.restartNote = null;
  frame("Using that environment…", h("p", { class: "setup-note text-dim" }, "Saving the choice."));
  try {
    const before = await bootId();
    const result = await api("/installer/apply", { method: "POST", body: { venv_python: python, restart: installed } });
    if (result.restarting) await waitForRestart(before);
    else if (result.restart_note) wizard.restartNote = result.restart_note;
  } catch (err) {
    wizard.restartNote = errText(err);
    logError("installer: could not move the server into the environment", err);
  }
  await stepInstalled();
}

async function restartNow() {
  wizard.restartNote = null;
  try {
    const before = await bootId();
    const result = await api("/installer/restart", { method: "POST" });
    if (result.restarting) await waitForRestart(before);
    else wizard.restartNote = result.restart_note || "The server did not restart.";
  } catch (err) {
    wizard.restartNote = errText(err);
    logError("installer: restart failed", err);
  }
  await stepInstalled();
}

/* ---- 3. install ---------------------------------------------------------- */

/* The log box is created once and updated in place: redrawing it every poll reset its scroll position, so a reader
   could not look back. It follows new output only while the reader is at the bottom. */
function logView() {
  const box = h("pre", { class: "setup-log", id: "setup-log" });
  return {
    box,
    set(lines) {
      const text = lines.join("\n");
      if (box.textContent === text) return;
      const follow = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
      box.textContent = text;
      if (follow) box.scrollTop = box.scrollHeight;
    },
  };
}

async function startInstall() {
  showMessage("setup-msg", "");
  let job;
  try {
    job = await api("/installer/install", { method: "POST", body: { target: wizard.target, venv_dir: wizard.venvDir || null, torch: torchValue() } });
  } catch (err) {
    showMessage("setup-msg", errText(err));
    logError("installer: could not start the install", err);
    return;
  }
  log(`installer: started job ${job.id}`);
  await watchInstall(job);
}

async function watchInstall(job) {
  wizard.step = 3;
  saveWizard();
  const title = h("h2", { class: "setup-step-title" }, "Installing…");
  const note = h("p", { class: "setup-note text-dim" });
  const failure = h("p", { class: "setup-error", hidden: true });
  const view = logView();
  const footer = actions();
  mount(root(),
    h("ol", { class: "setup-steps" }, STEPS.map((name, i) => h("li", { class: i + 1 === 3 ? "is-current" : i < 2 ? "is-done" : "" }, name))),
    title, note, failure, view.box, footer);

  for (;;) {
    try {
      job = await api(`/installer/install/${job.id}`);
    } catch (err) {
      // Lost to a server restart: the install itself may have finished, so say that instead of claiming either outcome.
      title.textContent = "Lost track of the install.";
      failure.textContent = errText(err);
      failure.hidden = false;
      mount(footer, button("Back", () => show(2)));
      return;
    }
    note.textContent = `${job.packages.length} packages into ${job.target_label}`;
    view.set(job.log);
    if (job.terminal) break;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  if (job.state === "succeeded") {
    log(`installer: job ${job.id} succeeded`);
    await adoptEnvironment(job.target_python, { installed: true });
  } else {
    title.textContent = "The install did not finish.";
    failure.textContent = job.error || job.state;
    failure.hidden = false;
    logError(`installer: job ${job.id} ended ${job.state}`, job.error);
    mount(footer, button("Back", () => show(2)), button("Try again", startInstall, { primary: true }));
  }
}

/* Whether the card is visible to PyTorch in the environment that will be used: the real answer, now that torch exists there. */
async function stepInstalled() {
  wizard.step = 3;
  saveWizard();
  frame("Checking the graphics card in that environment…", h("p", { class: "setup-note text-dim" }, "Importing PyTorch takes a few seconds."));
  try {
    wizard.verify = await api("/installer/environment/verify", { method: "POST", body: { target: wizard.target, venv_dir: wizard.venvDir || null } });
  } catch (err) {
    wizard.verify = { present: false, reason: errText(err) };
  }
  const v = wizard.verify;
  frame(v.present ? "Ready." : "Installed, but the card is not visible yet.",
    h("div", { class: `setup-device ${v.present ? "is-ok" : "is-missing"}` },
      h("span", { class: "setup-device-state", "aria-hidden": "true" }, v.present ? "✓" : "!"),
      h("span", {}, v.present ? `${v.name} — ${Number(v.total_memory_mb).toLocaleString()} MB, PyTorch ${v.torch_version}` : (v.reason || "PyTorch sees no Intel GPU.")),
      v.present ? null : h("div", { class: "text-dim" }, "Training will not start until PyTorch sees the card. You can finish setup and fix the driver afterwards.")),
    v.server_in_environment === false
      ? h("p", { class: "setup-error" }, "The server is not running inside that environment, so its nodes cannot load PyTorch. Press Restart the server.")
      : null,
    wizard.restartNote ? h("p", { class: "setup-error" }, wizard.restartNote) : null,
    actions(v.present && v.server_in_environment !== false ? null : button("Check again", stepInstalled),
      v.present && v.server_in_environment !== false ? null : button("Restart the server", restartNow),
      repairing() ? button("Finish", finishRepair, { primary: true }) : button("Continue", () => show(4), { primary: true })));
}

/* ---- 4. prompt files ----------------------------------------------------- */

async function downloadTokenizer(key, node) {
  node.disabled = true;
  node.textContent = "Downloading…";
  showMessage("setup-msg", "");
  try {
    await api(`/installer/tokenizers/${key}`, { method: "POST" });
  } catch (err) {
    showMessage("setup-msg", errText(err));
  }
  await stepTokenizers();
}

async function stepTokenizers() {
  try { wizard.tokenizers = await api("/installer/tokenizers"); } catch (err) { wizard.tokenizers = []; logError("installer: tokenizers", err); }
  wizard.step = 4;
  saveWizard();
  frame("Prompt files",
    h("p", { class: "setup-note text-dim" },
      "Prompts are tokenized with published vocabulary files. They are downloaded once into the project's assets folder; ComfyUI is not needed. "
      + "You can skip this and download them later."),
    wizard.tokenizers.map((t) => {
      const action = h("button", { class: "btn", type: "button" }, "Download");
      action.addEventListener("click", () => downloadTokenizer(t.key, action));
      return h("div", { class: `setup-pkg ${t.complete ? "is-ok" : "is-missing"}` },
        h("span", { class: "setup-pkg-state", "aria-hidden": "true" }, t.complete ? "✓" : "○"),
        h("span", { class: "setup-pkg-name" }, t.label),
        h("span", { class: "setup-pkg-version text-dim" }, `for ${t.needed_for}`),
        t.complete ? null : action);
    }),
    message(),
    actions(button("Back", () => show(2)), button("Continue", () => show(5), { primary: true })));
}

/* ---- 5. models folder ---------------------------------------------------- */

function stepModels() {
  wizard.step = 5;
  const initial = wizard.state.resolved_models_dir || wizard.state.default_models_dir;
  frame("Models folder",
    h("p", { class: "setup-note text-dim" },
      "Checkpoints are read from <folder>/checkpoints and LoRAs are saved to <folder>/loras, the same layout ComfyUI uses. "
      + "To share models with ComfyUI, choose its `models` folder."),
    h("label", { class: "setup-field", for: "setup-models" },
      h("span", { class: "setup-field-label" }, "Models folder"),
      h("input", { class: "setup-input", type: "text", id: "setup-models", value: initial }),
      h("span", { class: "setup-field-hint text-dim" }, "Missing folders are created. The default sits next to the project.")),
    message(),
    actions(button("Back", () => show(4)), button("Finish setup", finish, { primary: true })));
}

async function finish(event) {
  const finishButton = event.currentTarget;
  const models = el("setup-models").value.trim();
  if (!models) {
    showMessage("setup-msg", "Choose a models folder.");
    return;
  }
  finishButton.disabled = true;
  try {
    const body = { models_dir: models };
    if (wizard.python) body.venv_python = wizard.python;
    const before = await bootId();
    const result = await api("/installer/apply", { method: "POST", body });
    log("installer: setup finished");
    try { localStorage.removeItem(SAVED); } catch { /* only a convenience */ }
    if (result.restarting) await waitForRestart(before);
    window.location.replace("/help");
    return;
  } catch (err) {
    if (err instanceof ApiError && err.code === "installer_not_allowed") {
      showMessage("setup-msg", "Setup was already completed. Reloading…");
      setTimeout(() => window.location.reload(), 800);
      return;
    }
    showMessage("setup-msg", errText(err));
    logError("installer: apply failed", err);
    finishButton.disabled = false;
  }
}

async function bootId() {
  return (await api("/health")).boot_id;
}

/* Wait for a server whose boot id differs from ``before``: the new process, not the old one on its way out. */
async function waitForRestart(before) {
  frame("Restarting the server…", h("p", { class: "setup-note text-dim" }, "It restarts inside the environment you chose. This takes a few seconds."));
  const started = Date.now();
  while (Date.now() - started < 180000) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    try {
      if ((await bootId()) !== before) return;
    } catch (err) {
      /* down: it is restarting */
    }
  }
  throw new Error("The server did not come back within three minutes. Stop it in the terminal and start run_server_linux.sh again; setup continues from this step.");
}

/* ---- start --------------------------------------------------------------- */

async function boot() {
  loadWizard();
  wizard.state = await api("/installer/state");
  if (!wizard.state.setup_open) {
    window.location.replace("/help");
    return;
  }
  if (wizard.state.install_id) {
    const job = await api(`/installer/install/${wizard.state.install_id}`).catch(() => null);
    if (job && !job.terminal) return watchInstall(job);
  }
  if (wizard.step === 3 && wizard.python) return stepInstalled();
  show(wizard.step === 3 ? 2 : Math.min(Math.max(wizard.step, 1), 5));
}

boot().catch((err) => {
  logError("installer: could not load", err);
  const node = document.getElementById("setup-root");
  if (node) {
    mount(node, h("h2", { class: "setup-step-title" }, "Setup could not be loaded."), h("p", { class: "setup-error" }, errText(err)));
  }
});
