/* ---------------------------------------------------------------------------
   install.js -- the first-run installer (/setup).

   Three screens, in the order the questions actually have to be answered:

     1. Readiness -- can this machine start a training run? Nine packages,
        the accelerator, one verdict. Informational: it installs nothing.
     2. Where packages go, and which GPU -- two questions on one screen,
        because asking about the virtualenv before the user knows what is
        in it wastes their answer. The size is on this screen because the
        choice is a *disk* decision and "~2.5 GB" is what makes it
        informed. Selecting ComfyUI's venv runs the conflict check
        immediately, before anything is written.
     3. Paths -- where is ComfyUI, and where do model files live? Defaults
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
let devices = null;     // GET /installer/devices -- lazy, it costs ~1.8s
let conflicts = null;   // GET /installer/conflicts -- only when reuse is picked
let step = 1;           // 1 = readiness, 2 = install target, 3 = paths

const TIER_LABEL = {
  required: "server",
  training: "training",
  optional: "optional",
  comfy_provided: "from ComfyUI",
};

/* ---- step 2: install target and GPU ------------------------------------ */

/* The two targets, with the numbers that make the choice informed.
   `installable_only` is deliberate: a package this project would never
   install into the chosen venv is not part of its size, and quoting the
   full figure for "reuse ComfyUI's venv" would be a lie in the user's
   favour by 2.5 GB. */
function installOptions(root) {
  // The two sizes come from two different places on purpose. The new-venv
  // figure is the manifest's `full_install_approx_mb` -- what a from-scratch
  // install costs, which is what that option actually promises. The reuse
  // figure is the readiness report's `approx_download_mb`, which counts only
  // installable absences -- torch is never installable, so quoting the full
  // figure next to "reuse" would overstate that option by 2.5 GB in the
  // wizard's favour.
  const full = (manifest && manifest.full_install_approx_mb) || 0;
  const additions = (readiness && readiness.approx_download_mb) || 0;
  const picked = root.dataset.target || "new";

  const size = (mb) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`);

  const option = (id, title, blurb, meta) =>
    h(
      "label",
      { class: `setup-option${picked === id ? " is-picked" : ""}`, for: `setup-target-${id}` },
      h("input", {
        type: "radio",
        name: "setup-target",
        id: `setup-target-${id}`,
        value: id,
        checked: picked === id,
        onchange: () => {
          root.dataset.target = id;
          // Repaint immediately so the picked border and the swap of the
          // size figure are instant, then fetch the conflict check *only*
          // for the reuse option -- it is a subprocess in another
          // interpreter, and nobody asked for it until this click.
          renderInstallTarget(el("setup-root"));
          if (id === "comfy") {
            conflicts = null;
            enterInstallTarget(el("setup-root"));
          } else {
            conflicts = null;
          }
        },
      }),
      h("span", { class: "setup-option-body" },
        h("span", { class: "setup-option-title" }, title),
        h("span", { class: "setup-option-blurb" }, blurb),
        meta ? h("span", { class: "setup-option-meta" }, meta) : null),
    );

  return h(
    "fieldset",
    { class: "setup-options" },
    h("legend", {}, "Where should this project's packages go?"),
    option("new", "A new virtualenv for this project",
      "Nothing else on this machine is touched. Delete the directory to undo it.",
      size(full)),
    option("comfy", "Reuse ComfyUI's virtualenv",
      "Nothing to download for the training stack -- ComfyUI already has it. "
        + "We check for conflicts first and will not change a version ComfyUI declares.",
      size(additions)),
  );
}

/* The conflict check's three outcomes, rendered as three things rather than
   one verdict. The distinction is the point: "safe" and "unknown" both
   permit the install and both *mean* something different, and "unknown"
   is what a real venv is mostly made of -- 150 of 185 packages on this
   machine. Collapsing them into a boolean would hide the number that most
   justifies pinning everything. */
function conflictPanel() {
  if (!conflicts) return null;

  if (!conflicts.checked) {
    return h("div", { class: "setup-conflict is-refused" },
      h("h3", {}, "Could not check"),
      h("p", {}, conflicts.refusal_reason
        || "ComfyUI's environment could not be read, so safety cannot be shown."),
      h("p", { class: "text-dim" },
        "Choose a separate environment above, or fix the path and check again."));
  }

  const c = conflicts.counts || {};
  const rows = conflicts.findings.filter((f) => f.outcome !== "safe");
  const sample = rows.slice(0, 6);
  const rest = rows.length - sample.length;

  return h("div", { class: `setup-conflict ${conflicts.safe ? "is-safe" : "is-refused"}` },
    h("h3", {}, conflicts.safe
      ? `No conflicts — ${c.total} packages checked`
      : "ComfyUI's environment is inconsistent"),
    h("p", { class: "text-dim" },
      `${c.conflict} conflict${c.conflict === 1 ? "" : "s"}, `
      + `${c.safe} declared and matching, ${c.unknown} installed but not declared.`),
    conflicts.safe
      ? h("p", { class: "text-dim" },
        `All ${conflicts.constraints.length} packages are pinned to their exact `
        + "installed version, so nothing already in that environment can change.")
      : h("p", {}, conflicts.refusal_reason),
    sample.length
      ? h("ul", { class: "setup-conflict-list" },
        ...sample.map((f) => h("li", { class: `is-${f.outcome}` }, f.description)),
        rest > 0 ? h("li", { class: "text-dim" },
          `…and ${rest} more. All ${conflicts.constraints.length} are pinned exactly.`)
          : null)
      : null,
  );
}

/* CUDA is a *future* feature and is deliberately not a peer option. A
   radio button next to XPU invites a user to pick it, and picking it
   installs 3 GB of CUDA torch onto a card that will refuse to train --
   failing at run time, hours later, with a stack trace rather than a
   choice. So it sits below a rule, under its own heading, and says it is
   not available (ADR 0004). */
function futureBackends() {
  return h("div", { class: "setup-future" },
    h("hr", {}),
    h("h3", {}, "Not supported yet"),
    h("div", { class: "setup-option is-disabled" },
      h("span", { class: "setup-option-body" },
        h("span", { class: "setup-option-title" }, "NVIDIA CUDA"),
        h("span", { class: "setup-option-blurb" },
          "Not supported yet. This project targets Intel Arc, so a CUDA "
          + "build of torch would install ~3 GB and then refuse to train."),
        h("span", { class: "setup-option-meta" }, "~3 GB — unavailable"))),
  );
}

function deviceChoice() {
  if (!devices) {
    return h("p", { class: "text-dim" }, "Looking for graphics cards…");
  }
  if (!devices.enumerated) {
    return h("div", { class: "setup-conflict is-refused" },
      h("p", {}, devices.reason
        || "The graphics cards could not be listed."),
      h("p", { class: "text-dim" },
        "Screen 1 already reported whether a device is usable."));
  }
  if (!devices.devices.length) {
    return h("div", { class: "setup-conflict is-refused" },
      h("p", {}, "No supported graphics device was found."));
  }
  // One card is a fact, not a choice: rendering a radio for it would
  // invite the reader to think the wheel depends on an answer, and any
  // answer to that question is the same answer.
  if (devices.devices.length === 1) {
    const d = devices.devices[0];
    return h("div", { class: "setup-device is-ok" },
      h("span", { class: "setup-device-state", "aria-hidden": "true" }, "✓"),
      h("span", {}, d.name || "Graphics card"),
      d.total_memory_mb
        ? h("span", { class: "text-dim" }, ` — ${d.total_memory_mb.toLocaleString()} MB VRAM`)
        : null,
      h("span", { class: "text-dim" }, " — the only one, so the choice is made."));
  }
  return h("fieldset", { class: "setup-options" },
    h("legend", {}, `Which graphics card? (${devices.devices.length} found)`),
    ...devices.devices.map((d, i) => h(
      "label",
      { class: `setup-option${i === 0 ? " is-picked" : ""}` },
      h("input", { type: "radio", name: "setup-gpu", checked: i === 0 }),
      h("span", { class: "setup-option-body" },
        h("span", { class: "setup-option-title" }, d.name || `Device ${d.index}`),
        h("span", { class: "setup-option-meta" },
          d.total_memory_mb ? `${d.total_memory_mb.toLocaleString()} MB VRAM` : "size unknown")),
    )));
}

function renderInstallTarget(root) {
  // Read once, here, and pass it down: the radio's onchange re-enters this
  // function, and a component that re-read `root.dataset.target` each time
  // it asked is a component whose two views of the same value can disagree.
  const comfy = comfyFieldValue(root);
  const target = root.dataset.target || "new";
  // Replace, not append. This is re-entered from three places -- showStep,
  // the radio's onchange, and the two lazy fetches -- and only showStep
  // cleared the root. So every repaint stacked another copy of the whole
  // screen, which is how one click produced two screens and why the page
  // read as duplicated with one of them stuck on the loading line.
  root.replaceChildren();
  mount(root,
    h("h2", { class: "setup-step-title" }, "Where do packages go, and which GPU?"),
    h("p", { class: "setup-note text-dim" },
      "This choice decides how much disk the install needs and whether "
        + "anything already on this machine can be affected."),

    // ComfyUI's location comes before the virtualenv question, because the
    // second question cannot be asked without the first. See
    // comfyFieldValue() for why it cannot be read from settings.
    h(
      "label",
      { class: "setup-field", for: "setup-comfy-dir" },
      h("span", { class: "setup-field-label" }, "ComfyUI directory"),
      h("input", {
        class: "setup-input", type: "text", id: "setup-comfy-dir",
        name: "setup-comfy-dir", value: comfy,
        placeholder: "/path/to/ComfyUI",
        onchange: () => {
          rememberComfy(el("setup-root"));
          // Re-check against what was typed, but only if the reuse option
          // is the one being considered -- otherwise the check's answer
          // would appear under an option the user did not pick.
          if ((el("setup-root").dataset.target || "new") === "comfy") {
            conflicts = null;
            renderInstallTarget(el("setup-root"));
            loadConflicts(el("setup-root"));
          } else {
            rememberComfy(el("setup-root"));
          }
        },
      }),
      h("span", { class: "setup-field-hint text-dim" },
        "The folder containing ComfyUI's models/ directory. Needed to check "
        + "whether its virtualenv can be reused; leave it empty to skip that "
        + "option."),
    ),
    comfy
      ? null
      : h("p", { class: "setup-note text-dim" },
        "Without a path here, reusing ComfyUI's environment cannot be checked "
        + "— and an install that cannot be checked is one this project will "
        + "not make."),

    installOptions(root),
    h("h3", { class: "setup-subtitle" }, "Graphics card"),
    deviceChoice(),
    futureBackends(),
    target === "comfy" ? conflictPanel() : null,
    h("p", { class: "setup-error", id: "setup-msg", hidden: true }),
    h("div", { class: "setup-actions" },
      h("button", { class: "btn", type: "button", onclick: () => showStep(1) }, "Back"),
      h("button", { class: "btn btn-primary", type: "button", onclick: startInstall }, "Install"),
    ),
  );
}

/* Two lazy fetches, and the reason for each being lazy:
   - devices costs ~1.8s (it imports torch in a subprocess). Asked when
     this screen is *shown*, not before, and not on every render.
   - conflicts is asked only when the user picks ComfyUI's venv, because
     the answer is 185 rows long and nobody asked for it yet. */
async function enterInstallTarget(root) {
  const wantConflicts = (root.dataset.target || "new") === "comfy";
  const pending = devices
    ? Promise.resolve()
    : api("/installer/devices")
        .then((d) => { devices = d; })
        .catch((err) => {
          devices = {
            enumerated: false, backend: "xpu", devices: [], reason: errText(err),
          };
        });

  if (!wantConflicts) {
    conflicts = null;
    // Re-render once the cards are known. Without this the screen kept
    // saying "Looking for graphics cards…" for ever: the fetch resolved,
    // `devices` was set, and nothing ever asked for a repaint. The
    // placeholder was only ever replaced by navigating away and back.
    await pending;
    renderInstallTarget(el("setup-root"));
    return;
  }
  await pending;
  await loadConflicts(root);
}

/* The ComfyUI directory this screen checks, from the field on this screen
   and not from settings.

   Which is the whole reason the field is here. Configuring comfy_dir makes
   the server report `configured: true`, and the wizard then refuses to
   show at all ("Setup is already complete"). So a first-run machine --
   the only machine that sees this screen -- cannot have comfy_dir in
   settings, and asking about "ComfyUI's virtualenv" before its location is
   known made the reuse option unselectable in practice. A browser check of
   this screen is what found it: setting comfy_dir mid-test closed the
   wizard out from under the test.

   Asking here fixes the ordering the design intended -- "where is ComfyUI"
   first, then "which virtualenv" -- and screen 3 keeps only the model
   directories, which genuinely do follow from the answer. */
function comfyFieldValue(root) {
  const field = root.querySelector("#setup-comfy-dir");
  const typed = field ? field.value.trim() : "";
  const resolved = (state && state.resolved_comfy_dir) || "";
  return typed || resolved || "";
}

/* Remember what the user typed, so screen 3 persists the same value.

   Stored on the root rather than in a module variable because the root is
   replaced on every step change, and a module-level value survives a
   `boot()` that reset everything else. */
function rememberComfy(root) {
  const field = root.querySelector("#setup-comfy-dir");
  if (field) root.dataset.comfy = field.value.trim();
}

async function loadConflicts(root) {
  const comfy = comfyFieldValue(root);
  const query = comfy ? `?comfy_dir=${encodeURIComponent(comfy)}` : "";
  try {
    conflicts = await api(`/installer/conflicts${query}`);
  } catch (err) {
    // A failed check is a refusal, not an absence: rendering this as
    // "no conflicts" would be the one wrong answer available here.
    conflicts = {
      checked: false, safe: false, findings: [], additions: [],
      constraints: [], counts: {},
      refusal_reason: errText(err),
    };
  }
  renderInstallTarget(el("setup-root"));
}


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
      "This screen reports only -- nothing has been downloaded or changed. " +
        "The next screen is where you choose where packages should go.",
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
            `(${manifest.comfy_additions.join(", ")}) are the only ones ` +
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


/* ---- the install itself ------------------------------------------------- */

/* What to install, and into which target.

   The split is target-dependent and comes from the manifest, because the
   two options are not the same operation:

   - **A new venv**: this project owns it, so everything missing goes in --
     including torch, which is most of the 2.5 GB.
   - **ComfyUI's venv**: only what is safe there. `never_install` marks the
     accelerator stack as ComfyUI's to pin, not ours to install, and torch
     is what that flag exists to keep out.

   A first version filtered on `tier !== "comfy_provided"` instead, and
   that was wrong in the dangerous direction: no requirement actually uses
   that tier, so the filter matched nothing and the wizard offered to
   install torch into ComfyUI's virtualenv -- the one operation the whole
   conflict check exists to prevent. The tier constant is a real
   classification that the current manifest happens not to exercise; the
   flag on each row is the one that is load-bearing.

   `readiness.missing` is the authority on what is absent, so the two
   lists cannot disagree about it. */
function packagesToInstall(target) {
  if (!manifest || !readiness) return [];
  const missing = new Set(readiness.missing || []);
  return manifest.requirements
    .filter((r) => missing.has(r.name))
    .filter((r) => target !== "comfy" || !r.never_install)
    .map((r) => r.name);
}

async function startInstall() {
  const root = el("setup-root");
  const target = root.dataset.target || "new";
  const packages = packagesToInstall(target);
  const button = root.querySelector(".btn-primary");

  if (!packages.length) {
    showMessage("setup-msg",
      "Everything this project needs is already installed.");
    showStep(3);
    return;
  }

  let body = { target, packages };

  if (target === "comfy") {
    // Installing into a venv we have not successfully checked is the one
    // option here that could change a version ComfyUI declares, so it is
    // refused rather than attempted without pins. The user is sent to a
    // separate environment, which is the only other thing on this screen.
    if (!conflicts || !conflicts.checked) {
      showMessage("setup-msg",
        "ComfyUI's environment has not been checked, so this install cannot "
        + "be shown to be safe. Choose a separate virtualenv, or run the "
        + "check first.");
      return;
    }
    const settings = await api("/settings");
    body.constraints = conflicts.constraints;
    body.comfy_venv_python = settings.resolved.venv_python;
  } else {
    body.constraints = [];
  }

  button.disabled = true;
  let job;
  try {
    job = await api("/installer/install", { method: "POST", body });
    log(`installer: started job ${job.id} (${job.state})`);
  } catch (err) {
    showMessage("setup-msg", errText(err));
    logError("installer: could not start the install", err);
    button.disabled = false;
    return;
  }
  await pollInstall(job.id);
}

/* Polling rather than a streaming endpoint: the job id survives a reload,
   every state stays fetchable afterwards, and it is the shape the graph
   execution endpoints in this app already use. */
async function pollInstall(jobId) {
  const root = el("setup-root");
  let seen = 0;

  for (;;) {
    let job;
    try {
      job = await api(`/installer/install/${jobId}`);
    } catch (err) {
      // A job this server does not know was almost certainly lost to a
      // restart. Not reported as a failure of the install -- the readiness
      // report is the source of truth for whether the packages are there --
      // and certainly not reported as success.
      renderInstallStatus(root, {
        state: "unknown",
        packages: [],
        constraints: [],
        command: [],
        log: [],
        error: errText(err),
      });
      return;
    }
    renderInstallStatus(root, job, job.log.length - seen);
    seen = job.log.length;

    if (job.terminal) {
      if (job.state === "succeeded") {
        log(`installer: job ${jobId} succeeded`);
        showStep(3);
      } else {
        logError("installer: job " + jobId + " ended " + job.state, job.error);
      }
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

function renderInstallStatus(root, job, freshLines = 0) {
  const done = job.state === "succeeded";
  const bad = job.state === "failed" || job.state === "unknown";
  mount(root,
    h("h2", { class: "setup-step-title" },
      done ? "Installed."
        : bad ? "The install did not finish."
        : job.state === "queued" ? "Waiting to start…"
        : "Installing…"),
    h("p", { class: "setup-note text-dim" },
      `${(job.packages || []).length} packages into ${job.target_label || "the chosen environment"}`
      + (job.constraints && job.constraints.length
        ? `, with ${job.constraints.length} existing packages pinned to their exact version`
        : "")
      + "."),
    bad && job.error ? h("p", { class: "setup-error" }, job.error) : null,
    // The command, because "nothing already installed can change" is a claim
    // about a specific pip invocation, and a claim is worth more when the
    // invocation implementing it is readable.
    job.command && job.command.length
      ? h("pre", { class: "setup-cmd text-dim" }, job.command.join(" \\\n  "))
      : null,
    h("pre", { class: `setup-log${freshLines ? " is-new" : ""}` },
      (job.log || []).slice(-200).join("\n")),
    h("div", { class: "setup-actions" },
      done || bad
        ? h("button", { class: "btn", type: "button", onclick: () => showStep(2) }, "Back")
        : null,
      done
        ? h("button", { class: "btn btn-primary", type: "button", onclick: () => showStep(3) }, "Continue")
        : bad
          ? h("button", { class: "btn btn-primary", type: "button", onclick: startInstall }, "Try again")
          : null),
  );
}

/* ---- step 3: paths ------------------------------------------------------ */

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
        "already resolves when nothing is configured. The ComfyUI " +
        "directory itself was given on the previous screen, because the " +
        "install-target question cannot be asked without it.",
    ),
    h("p", { class: "setup-note text-dim" }, `ComfyUI: ${comfyDir || "not set"}`),
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
        { class: "btn", type: "button", onclick: () => showStep(2) },
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
  // Collected on screen 2, not here: screen 2 needs it to check the reuse
  // option, and a field that exists on the screen that needs it and is
  // then asked for again on the next one can disagree. `dataset.comfy` is
  // the single value both screens read.
  const comfy = (el("setup-root").dataset.comfy || "").trim();
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
  if (n === 1) {
    renderReadiness(root);
    return;
  }
  if (n === 2) {
    // `dataset.target` is read by installOptions before anything is
    // rendered, so it has to be initialised here rather than lazily inside
    // the render -- the first version left it undefined, and the fallback
    // to "new" silently overwrote a "comfy" the user had already chosen
    // when they went Back and returned.
    if (root.dataset.target !== "comfy") root.dataset.target = "new";
    renderInstallTarget(root);
    rememberComfy(root);
    enterInstallTarget(root);
    return;
  }
  // What resolution found, if anything: a machine where ComfyUI was
  // auto-detected gets its own directory pre-filled rather than being
  // asked to type what the server just told it.
  renderPaths(root, { comfyDir: root.dataset.comfy || (state && state.resolved_comfy_dir) || "" });
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

  // Destructure in the same order as the array. This was
  // `const [, ready, man] = await Promise.all([manifest, readiness, null])`
  // followed by `manifest = ready` -- so the manifest variable was holding
  // the *readiness* payload and `full_install_approx_mb` was permanently
  // undefined. That is why the design doc recorded the install size as
  // "unrendered": it was a wiring bug, not a missing feature, and screen 2
  // cannot size a disk decision without it.
  //
  // Both are needed: readiness is what screen 1 renders and what sizes the
  // "reuse" option, the manifest is what sizes the "new venv" option and
  // names the packages safe to add to ComfyUI's environment.
  const [man, ready] = await Promise.all([
    api("/installer/manifest"),
    api("/installer/readiness"),
  ]);
  manifest = man;
  readiness = ready;

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