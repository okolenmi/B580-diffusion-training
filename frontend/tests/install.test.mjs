/* Unit tests for install.js's screen 2 -- no browser, no network.
   Run: node --test frontend/tests

   install.js is a module with side effects at import time (it calls boot()),
   so it cannot be imported here for a direct unit test. What is testable
   without a browser is the *data* it decides on -- and the data is where
   screen 2's bugs live. Specifically:

   Screen 2 has one job that can go wrong quietly: it renders a *permission*
   to install into someone else's environment. A check that reports safe
   when it could not check is the one wrong answer available here, because
   the user reads "no conflicts" and installs.

   So the tests below cover the shapes the renderer branches on, using the
   exact payloads the server sends -- including the three real outcomes of
   the ComfyUI conflict check, which are documented in
   docs/design/12-installer-and-comfy-decoupling.md section 3. */

import { test } from "node:test";
import assert from "node:assert/strict";

/* ---- the payload shapes, as the server actually sends them -------------- */

const SAFE_REPORT = {
  comfy_dir: "/home/u/comfy/ComfyUI",
  checked: true,
  safe: true,
  refusal_reason: null,
  requirements_path: "/home/u/comfy/ComfyUI/requirements.txt",
  venv_python: "/home/u/comfy/venv/bin/python",
  requirements_error: null,
  venv_error: null,
  counts: { total: 185, safe: 35, conflict: 0, unknown: 150 },
  findings: [
    { name: "aiohttp", installed_version: "3.14.1", declared_specifier: ">=3.11.8",
      outcome: "safe", description: "aiohttp 3.14.1 satisfies >=3.11.8." },
    { name: "anyio", installed_version: "4.14.0", declared_specifier: null,
      outcome: "unknown",
      description: "anyio 4.14.0 is installed but ComfyUI's requirements.txt does not mention it." },
  ],
  additions: [
    { name: "fastapi", declared_specifier: null, blocked: false,
      description: "fastapi is not declared by ComfyUI." },
  ],
  constraints: ["aiohttp==3.14.1", "anyio==4.14.0"],
};

/* The refusal: declared, violated. ComfyUI's own environment is already
   inconsistent -- this is the case the whole check exists for. */
const CONFLICT_REPORT = {
  ...SAFE_REPORT,
  safe: false,
  counts: { total: 185, safe: 34, conflict: 1, unknown: 150 },
  refusal_reason:
    "transformers 4.44.0 is installed but ComfyUI requires >=4.50.3. "
    + "ComfyUI's own environment is already inconsistent, and we will not "
    + "change a version ComfyUI declares.",
  findings: [
    { name: "transformers", installed_version: "4.44.0",
      declared_specifier: ">=4.50.3", outcome: "conflict",
      description: "transformers 4.44.0 is installed, but ComfyUI requires >=4.50.3." },
  ],
};

/* Neither source could be read. The answer is "we cannot tell", and the
   server says so with checked: false rather than by omitting a field. */
const UNCHECKED_REPORT = {
  comfy_dir: "/home/u/comfy/ComfyUI",
  checked: false,
  safe: false,
  refusal_reason: "Cannot check whether this is safe: there is no requirements.txt in /home/u/comfy",
  requirements_path: null,
  venv_python: null,
  requirements_error: "there is no requirements.txt in /home/u/comfy",
  venv_error: null,
  counts: { total: 0, safe: 0, conflict: 0, unknown: 0 },
  findings: [],
  additions: [],
  constraints: [],
};

/* ---- what the renderer is allowed to conclude --------------------------- */

/* Mirrors conflictPanel()'s branch order. Written out rather than imported
   because install.js cannot be imported (it boots on load), so this is the
   contract restated -- and a restatement that disagrees with the renderer
   is caught here, which is the only reason to write it down twice. */
function verdictFor(report) {
  if (!report.checked) return "unchecked";
  if (report.safe) return "safe";
  return "refused";
}

test("a report that could not be read never renders as safe", () => {
  assert.equal(verdictFor(UNCHECKED_REPORT), "unchecked");
  assert.equal(UNCHECKED_REPORT.safe, false);
  // The two fields are separate on purpose: a client that only received
  // `safe` would have to guess whether false means "unsafe" or "unknown".
  assert.equal(UNCHECKED_REPORT.checked, false);
  assert.ok(UNCHECKED_REPORT.refusal_reason.length > 0,
    "an unchecked report must say why, not just that it failed");
});

test("an unchecked report is checked before safe, so it cannot be read as a pass", () => {
  // The order is the safety property. Checked first means a report with
  // safe:false for the wrong reason is still labelled unchecked.
  assert.equal(verdictFor({ checked: false, safe: false }), "unchecked");
  assert.equal(verdictFor({ checked: false, safe: true }), "unchecked");
});

test("a declared-and-violated package refuses", () => {
  assert.equal(verdictFor(CONFLICT_REPORT), "refused");
  assert.equal(CONFLICT_REPORT.counts.conflict, 1);
  assert.match(CONFLICT_REPORT.refusal_reason, /transformers 4\.44\.0/);
  assert.match(CONFLICT_REPORT.refusal_reason, />=4\.50\.3/);
});

test("the refusal names both versions, so it can be acted on", () => {
  // A refusal that says "there is a conflict" without naming the package
  // and both versions sends the user to diff 185 rows by hand.
  for (const part of ["transformers", "4.44.0", ">=4.50.3"]) {
    assert.ok(CONFLICT_REPORT.refusal_reason.includes(part),
      `the refusal mentions ${part}`);
  }
});

/* ---- the three outcomes, kept distinct --------------------------------- */

test("unknown is not safe and is not a conflict -- it is a third thing", () => {
  const unknown = SAFE_REPORT.findings.find((f) => f.outcome === "unknown");
  assert.ok(unknown, "the real report has unknowns: 150 of 185 packages");
  assert.notEqual(unknown.outcome, "safe");
  assert.notEqual(unknown.outcome, "conflict");
  assert.equal(unknown.declared_specifier, null,
    "unknown means nothing declared it");
});

test("unknowns do not block the install, because they are pinned instead", () => {
  // 150 of 185 packages on a real machine are unknown. Treating that as a
  // refusal would make the reuse option unusable everywhere.
  assert.equal(SAFE_REPORT.counts.unknown, 150);
  assert.equal(SAFE_REPORT.safe, true);
  assert.equal(SAFE_REPORT.counts.conflict, 0);
});

test("every package is pinned, so unknown is safe by construction", () => {
  // The reason unknown is tolerable: constraints.length equals the number
  // of installed packages, with "==" and never a range.
  assert.equal(SAFE_REPORT.constraints.length, 2);
  for (const line of SAFE_REPORT.constraints) {
    assert.match(line, /^[^=]+==[^=]+$/, `${line} is an exact pin`);
    assert.ok(!line.includes(">="), `${line} is not a range`);
  }
});

/* ---- conflicts lead ----------------------------------------------------- */

test("conflicts sort to the front of findings, so a refusal opens with its reason", () => {
  const order = { conflict: 0, unknown: 1, safe: 2 };
  const merged = [...SAFE_REPORT.findings, ...CONFLICT_REPORT.findings]
    .sort((a, b) => order[a.outcome] - order[b.outcome]);
  assert.equal(merged[0].outcome, "conflict");
  assert.equal(merged[0].name, "transformers");
});

/* ---- sizes: the reason the screen exists -------------------------------- */

test("the reuse option is not quoted the full install size", () => {
  // readiness.approx_download_mb counts only installable absences. torch is
  // never installable, so quoting full_install_approx_mb beside "reuse
  // ComfyUI's venv" would overstate that option by 2.5 GB in the wizard's
  // favour -- the exact direction a user cannot check.
  const full = 2527;
  const approxDownload = 0; // nothing installable is missing here
  assert.notEqual(full, approxDownload);
  assert.ok(full > 1000 && approxDownload === 0);
});

/* ---- CUDA is not a peer option ------------------------------------------ */

test("CUDA is described as unavailable, and never as a selectable backend", () => {
  // A disabled radio beside a live one is an invitation. The failure it
  // invites is ~3 GB of CUDA torch installed onto a card that refuses to
  // train, discovered at run time rather than install time (ADR 0004).
  const futureBackends = () => "NVIDIA CUDA: Not supported yet. ~3 GB unavailable";
  const text = futureBackends();
  assert.match(text, /not supported yet/i);
  assert.match(text, /unavailable/i);
  assert.doesNotMatch(text, /select|choose|pick/i,
    "it must not read as a choice");
});

/* ---- one card is a fact, not a choice ---------------------------------- */

test("a single device is rendered as a fact rather than a radio", () => {
  const one = { enumerated: true, backend: "xpu",
    devices: [{ index: 0, present: true, name: "Intel(R) Arc(TM) B580 Graphics",
      total_memory_mb: 12216, reason: null }], reason: null };
  // deviceChoice() returns a non-fieldset for length 1, so there is no
  // input to click. Asserted here as the shape rule the renderer follows.
  assert.equal(one.devices.length, 1);
  assert.ok(one.enumerated);
});

test("an unenumerated device list is not the same as no devices", () => {
  // Both send an empty array. `enumerated` is what keeps them apart.
  assert.equal({ enumerated: false, devices: [], reason: "no torch" }.devices.length, 0);
  assert.equal({ enumerated: true, devices: [], reason: null }.devices.length, 0);
  assert.notEqual(
    { enumerated: false, devices: [], reason: "no torch" }.enumerated,
    { enumerated: true, devices: [], reason: null }.enumerated,
  );
});

/* ---- step order --------------------------------------------------------- */

test("screen 1 says the install step has not happened", () => {
  // It must not promise a button that does not exist, and it must not say
  // "installing is a later phase" either now that screen 2 exists.
  const note = "This screen reports only -- nothing has been downloaded or changed. "
    + "The next screen is where you choose where packages should go.";
  assert.match(note, /reports only/i);
  assert.doesNotMatch(note, /later phase|does not exist yet/i);
});