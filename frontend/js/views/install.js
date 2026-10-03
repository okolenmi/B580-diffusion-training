/* ---------------------------------------------------------------------------
   install.js -- the first-run installer (/setup).

   Two screens, in the order the questions actually have to be answered:

     1. Readiness -- can this machine start a training run? Eight packages,
        the accelerator, one verdict. This screen is *informational*: it
        installs nothing, because nothing here is installable yet (the
        opt-in download is a later phase, and a screen that promises a
        button which does not exist is worse than one that says "not yet").
     2. Paths -- where is ComfyUI, and where do model files live? Defaults
        pre-filled from ComfyUI's own layout, because the resolution policy
        in path_tiers already prefers it and a default that disagreed with
        that policy would surface later as "I set it to the default and it
        went somewhere else".

   The wizard only *appears* while the installation is unconfigured, and
   the server refuses the write after it is configured (`installer_not_
   allowed`, 409). This page checks that state on entry and steps aside if
   it has already been done -- so a stale tab cannot re-point the model
   directories out from under a run.

   Everything network-shaped goes through api.js, so the error envelope is
   decoded in one place. A 409 from apply is *expected* here and is shown
   as a closed-wizard message rather than as an error to retry.
   --------------------------------------------------------------------------- */

import { api, ApiError } from "../api.js";
import { el, showMessage } from "../lib/dom.js";
import { errText } from "../lib/errors.js";
import { log, logError } from "../lib/log.js";


/* tiny DOM builder, as in the other views: attrs {class, text, ...} and
   no innerHTML anywhere server or user data reaches. */
function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "value") node.value = value;
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}


/* Append children, skipping nullish ones. `Node.append` renders a null
   argument as the text "null", and several pieces of both screens are
   conditional -- so the skipping has to happen here rather than being
   remembered at each call site. */
function mount(root, ...children) {
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    root.append(child);
  }
}


/* ---- state ------------------------------------------------------------- */

let state = null;       // GET /installer/state
let readiness = null;   // GET /installer/readiness
let manifest = null;    // GET /installer/manifest
let step = 1;           // 1 = readiness, 2 = paths

const TIER_LABEL = {
  required: "server",
  training: "training",
  optional: "optional",
  comfy_provided: "from ComfyUI",
};


/* ---- step 1: readiness -------------------------------------------------- */

function tierGroup(tier) {
  const rows = readiness.packages.filter((p) => p.tier === tier);
  if (!rows.length) return null;
  return h(
    "div",
    { class: "setup-tier" },
    h("h3", { class: "setup-tier-title" }, TIER_LABEL[tier] || tier),
    ...rows.map(packageRow),
  );
}

function packageRow(pkg) {
  const ok = pkg.installed;
  return h(
    "div",
    { class: `setup-pkg ${ok ? "is-ok" : pkg.blocking ? "is-missing" : "is-optional-missing"}` },
    h(
      "span",
      { class: "setup-pkg-state", "aria-hidden": "true" },
      ok ? "✓" : pkg.blocking ? "✕" : "–",
    ),
    h(
      "span",
      { class: "setup-pkg-name" },
      pkg.name,
      ok && pkg.installed_version
        ? h("span", { class: "setup-pkg-version text-dim" }, ` ${pkg.installed_version}`)
        : null,
    ),
    h("span", { class: "setup-pkg-why text-dim" }, pkg.why),
    pkg.approx_mb && !ok
      ? h("span", { class: "setup-pkg-size text-dim" }, `~${pkg.approx_mb} MB`)
      : null,
  );
}

function deviceRow() {
  if (!readiness.device_checked) {
    return h(
      "div",
      { class: "setup-device is-unknown" },
      h("span", { class: "setup-device-state" }, "?"),
      h(
        "span",
        {},
        "Graphics card not checked yet -- the training stack above is " +
          "incomplete, so the answer could not change the verdict.",
      ),
    );
  }
  if (!readiness.device_present) {
    return h(
      "div",
      { class: "setup-device is-missing" },
      h("span", { class: "setup-device-state", "aria-hidden": "true" }, "✕"),
      h(
        "span",
        {},
        readiness.device_reason || "No supported graphics device was found.",
        readiness.device_detail
          ? h("span", { class: "text-dim setup-device-detail" }, ` ${readiness.device_detail}`)
          : null,
      ),
    );
  }
  const mb = readiness.device_total_memory_mb;
  return h(
    "div",
    { class: "setup-device is-ok" },
    h("span", { class: "setup-device-state", "aria-hidden": "true" }, "✓"),
    h("span", {}, readiness.device_name || "Supported device"),
    mb ? h("span", { class: "text-dim" }, ` — ${mb.toLocaleString()} MB VRAM`) : null,
  );
}

function renderReadiness(root) {
  const verdict = h(
    "div",
    { class: `setup-verdict ${readiness.ready ? "is-ok" : "is-warn"}` },
    h("strong", {}, readiness.ready ? "Ready to train." : "Not ready to train yet."),
    " ",
    readiness.ready
      ? "Every package this project needs is installed and the card is visible."
      : `${readiness.blocking_missing_count ?? readiness.missing.length} of ` +
        `${readiness.packages.length} required packages are missing.`,
  );

  mount(root,
    h("h2", { class: "setup-step-title" }, "Can this machine run a training step?"),
    verdict,
    h(
      "p",
      { class: "setup-note text-dim" },
      "This page reports only. Installing packages is a separate, explicit " +
        "step that does not exist yet -- nothing here has been downloaded or " +
        "changed.",
    ),
    deviceRow(),
    tierGroup("required"),
    tierGroup("training"),
    // Conditional pieces go through `mount`, not `root.append`: the native
    // append stringifies a null argument into a text node, which is how a
    // "render only when the manifest loaded" conditional put a literal
    // `null` under the package list. `h()` skips nullish children;
    // `mount` exists so the top level cannot forget to.
    manifest && manifest.comfy_additions
      ? h(
          "p",
          { class: "setup-note text-dim" },
          `${manifest.comfy_additions.length} server packages ` +
            "(fastapi, uvicorn, python-multipart, tomli_w) are the only ones " +
            "safe to install into ComfyUI's own environment. torch and the " +
            "accelerator stack are what ComfyUI already has and pins, so " +
            "this project will not install them there.",
        )
      : null,
    h(
      "div",
      { class: "setup-actions" },
      h(
        "button",
        { class: "btn btn-primary", type: "button", onclick: () => showStep(2) },
        "Continue",
      ),
      h(
        "button",
        {
          class: "btn",
          type: "button",
          onclick: () => refreshReadiness(root, { announce: true }),
        },
        "Re-check",
      ),
    ),
  );
}


/* ---- step 2: paths ------------------------------------------------------ */

function pathField(id, label, value, hint) {
  return h(
    "label",
    { class: "setup-field", for: id },
    h("span", { class: "setup-field-label" }, label),
    h("input", { class: "setup-input", type: "text", id, name: id, value: value || "" }),
    hint ? h("span", { class: "setup-field-hint text-dim" }, hint) : null,
  );
}

function renderPaths(root, { comfyDir = "" } = {}) {
  // The model default is *derived*, not read: renderPaths builds the very
  // ComfyUI field it was about to read, so `el("setup-comfy")` threw
  // "no element with id setup-comfy" and Continue silently did nothing --
  // the one step of the wizard that cannot fail loudly, because a throw
  // inside a click handler has no caller to reach.
  const derived = comfyDir ? `${comfyDir}/models/checkpoints` : "";

  mount(root,
    h("h2", { class: "setup-step-title" }, "Where do model files live?"),
    h(
      "p",
      { class: "setup-note text-dim" },
      "Defaults follow ComfyUI's own layout, which is what this project " +
        "already resolves when nothing is configured.",
    ),
    pathField(
      "setup-comfy",
      "ComfyUI directory",
      comfyDir,
      "The folder containing ComfyUI's models/ directory.",
    ),
    pathField(
      "setup-models",
      "Models directory (optional)",
      derived ? `${derived}checkpoints` : "",
      "Leave empty to use ComfyUI's own models/checkpoints.",
    ),
    h(
      "p",
      { class: "setup-error", id: "setup-msg", hidden: true },
    ),
    h(
      "div",
      { class: "setup-actions" },
      h(
        "button",
        { class: "btn", type: "button", onclick: () => showStep(1) },
        "Back",
      ),
      h(
        "button",
        { class: "btn btn-primary", type: "button", onclick: applyPaths },
        "Finish setup",
      ),
    ),
  );
}

async function applyPaths() {
  const button = document.querySelector(".setup-actions .btn-primary");
  const comfy = el("setup-comfy").value.trim();
  const models = el("setup-models").value.trim();

  if (!comfy) {
    showMessage("setup-msg", "ComfyUI directory is required.");
    return;
  }
  button.disabled = true;
  try {
    const body = { comfy_dir: comfy };
    // Only sent when filled: an absent key means "leave it alone", so an
    // untouched default does not become an explicit override of the
    // resolution policy.
    if (models) body.checkpoints_dir = models;
    const applied = await api("/installer/apply", { method: "POST", body });
    log(`installer: configured, resolved comfy_dir=${applied.state.resolved_comfy_dir}`);
    await boot();
  } catch (err) {
    if (err instanceof ApiError && err.code === "installer_not_allowed") {
      // Expected once another tab has finished. Not an error to retry.
      showMessage("setup-msg", "Setup was already completed. Reloading…");
      log("installer: already configured elsewhere, reloading");
      setTimeout(() => window.location.reload(), 800);
      return;
    }
    showMessage("setup-msg", errText(err));
    logError("installer: apply failed", err);
  } finally {
    button.disabled = false;
  }
}


/* ---- shell ------------------------------------------------------------- */

function showStep(n) {
  step = n;
  const root = el("setup-root");
  root.replaceChildren();
  if (n === 1) renderReadiness(root);
  // What resolution found, if anything: a machine where ComfyUI was
  // auto-detected gets its own directory pre-filled rather than being
  // asked to type what the server just told it.
  else renderPaths(root, { comfyDir: (state && state.resolved_comfy_dir) || "" });
}

async function refreshReadiness(root, { announce = false } = {}) {
  if (announce) {
    showMessage("setup-msg", "");
    log("installer: re-checking this machine");
  }
  readiness = await api("/installer/readiness");
  showStep(1);
}

async function boot() {
  const root = el("setup-root");
  // Order matters only for cost: /state is cheap and decides whether this
  // page exists at all, and readiness costs a torch-importing subprocess.
  state = await api("/installer/state");

  if (state.configured) {
    document.title = "Setup already complete";
    root.replaceChildren(
      h("h2", { class: "setup-step-title" }, "Setup is already complete."),
      h(
        "p",
        { class: "setup-note text-dim" },
        `ComfyUI: ${state.resolved_comfy_dir}`,
      ),
      h(
        "p",
        {},
        h(
          "a",
          { class: "setup-link", href: "/settings" },
          "Open Settings",
        ),
        " to change any of these.",
      ),
    );
    return;
  }

  const [, ready, man] = await Promise.all([
    api("/installer/manifest"),
    api("/installer/readiness"),
    Promise.resolve(null),
  ]);
  manifest = ready;
  readiness = ready;
  void man;

  // The manifest is fetched for the wording about the ComfyUI additions
  // list; it is small and static, and failing to get it must not stop the
  // screen that matters.
  showStep(step);
}

boot().catch((err) => {
  logError("installer: could not load", err);
  const root = document.getElementById("setup-root");
  if (root) {
    root.replaceChildren(
      h("h2", { class: "setup-step-title" }, "Setup could not be loaded."),
      h("p", { class: "setup-error" }, errText(err)),
    );
  }
});