# 12. Installer design, and the cost of not needing ComfyUI

*[← design index](README.md)*

**Status: §2 is built; §3 onward is design.** This is the reasoning behind a
rebuild of the first-run installer, plus a measured answer to a question
that came up while building the first version: *could this project drop its
dependency on ComfyUI?* The answer is below, with numbers.

The §2 preflight exists as `backend/first_run.py`, called by
`run_server.sh`; §3's third screen does not. The ComfyUI separation
(§7) has not been started.

Supersedes the phasing in
[`11-first-run-and-installer.md`](11-first-run-and-installer.md) for
anything about install order, and corrects
[ADR 0005](../decisions/0005-installer-as-browser-surface.md) on one point.
The built parts are Phase A (the requirement manifest), Phase D (first-run
detection) and the read-only wizard; this document describes what replaces
them and why the ordering in ADR 0005 was wrong.

---

## 1. What was built, and what was wrong with it

The first version is a readiness report plus a two-field path form, served
from inside the running server at `/setup`. It reports; it installs nothing.
ADR 0005 argued that was the right shape, because the server has to be
running for a browser page to exist.

**That reasoning is correct and the conclusion is wrong**, and the gap is
ordering rather than principle. A wizard that lives inside the server can
only do things the server can already do — but the *first* thing that is
missing is usually the thing that stops the server starting. So the version
that ships cannot help the person who needs it most.

The specific case, which is the project's own: `requirements.txt` names four
packages (`fastapi`, `uvicorn`, `python-multipart`, `tomli_w`) and the
trainer imports four more (`torch`, `numpy`, `safetensors`, `pillow`) that
appear in no requirements file, because they are expected from ComfyUI's
venv. A user who installs `requirements.txt` into a fresh venv gets a server
that starts and a training stack that is silently absent — a
`ModuleNotFoundError` at the moment they press "start training", from code
they did not write, about a package they were never told about.

---

## 2. What can install before the server exists

Not much, and that is the design constraint. Split by whether the server
process is needed:

| Package | Needs the server? | Why |
|---|---|---|
| fastapi, uvicorn, python-multipart, tomli_w | **no** | four small pure-Python packages, nothing project-specific |
| torch, numpy, safetensors, pillow | no, but needs a *decision* | the decision needs the readiness report and the GPU choice |

The four server packages are the whole of "make the server start". So:

```
run_server.sh
  └─ python -m backend.first_run          # stdlib only, ~4s of work
       ├─ imports the four?  yes → exit 0, print nothing
       ├─ creates <tempdir>/distillation-bootstrap-<pid>/
       ├─ pip installs the four into it
       ├─ prints the URL, tries to open a browser
       └─ execve run_server.sh, with VENV_PYTHON = that interpreter
  └─ exec python -m backend.cli --host 0.0.0.0 "$@"
```

**Measured, not estimated:** `python -m venv` 2.1 s, the four packages
5.1 s, 32 MB total. `scripts/test_bootstrap_install.sh` runs the whole
thing end to end in a venv with no site-packages and asserts the server
answers afterwards.

### What changed from the first draft, and why

The version above replaced a plan for a **preflight web server** on port
8767 with **no server at all**. The page it would have served had one job
— say "installing" — and the terminal already says that, with pip's real
output attached. A second HTTP server on loopback would also have been a
new thing to bind, to secure, and to get wrong on a machine whose only
working server is the one that cannot start yet. So the shape is now the
simplest one that works: install, print, re-exec.

Three consequences, stated because they are the non-obvious parts:

* **The re-exec replaces the process, so `run_server.sh` runs twice.**
  That is fine and is the point: the second pass finds the packages present
  and returns 0 immediately, so `.env`, the interpreter precedence and the
  user's arguments are applied by the one script that owns them rather than
  reimplemented in Python. Ctrl-C reaches the server directly, because there
  is no parent left to intercept it.
* **`--host` and `--port` must be forwarded from the original `argv`**, not
  from `parse_known_args`'s unrecognised remainder. They are known to the
  bootstrap — it needs them to print a link — so the remainder is empty and
  the server came up on the default port while the printed link named the
  requested one. Found by the install test, because the unit test called
  `main()` with an explicit list and never went near the path `__main__`
  takes. Both that and a second one (forwarding `argv` before resolving
  `None`, which unpacked a `TypeError`) are now covered by
  `backend/tests/test_bootstrap.py`, and both mutations were checked to
  turn it red.
* **`DISTILLATION_NO_BROWSER=1` suppresses the tab**, and the reason is
  not politeness: the install test drove this path repeatedly and opened a
  real tab on the desktop every run. A bootstrap that reaches for the
  user's browser unattended is doing something surprising whether or not the
  install was wanted.

`--check` reports what is missing and installs nothing. It exists because
there was otherwise no way to ask the question without starting the answer,
which is also what made the first version of the install test hang.

---

## 3. The wizard, once the server is up

Three screens, in this order. The order is the argument of this document.

### Screen 1 — readiness (already built, keep)

Per-package presence with versions, the device, one verdict. Costs about
1.7 s because the device probe imports torch in a subprocess; skipped
entirely when the package check has already answered the question.

Unchanged from what ships. It is the one screen that exists for its own
sake, and it is what sizes screen 2.

### Screen 2 — where do packages go, and which GPU

This merges what were two separate questions, because asking about the
venv before the user knows what is in it wastes their answer:

> **Where should packages be installed?**
>
> This project needs 8 packages, about 2.5 GB — mostly torch, which is
> backend-specific (see §5).
>
> ○ **A new virtualenv for this project** — ~2.5 GB. Nothing else is
>   touched. You can delete the directory to undo it.
>
> ○ **Reuse ComfyUI's virtualenv** — nothing to download. We check for
>   conflicts with ComfyUI's `requirements.txt` first, and install only the
>   four server packages under a constraints file.

The size figure is not decoration: the venv choice is a *disk* decision, and
`full_install_approx_mb` is already computed and exposed. It is currently
unrendered, which is the clearest sign that the first version answered the
wrong question first.

Selecting "reuse" runs the conflict check immediately, before anything is
written, and reports the specific clash:

> `transformers 4.44` is installed; ComfyUI requires `>=4.50.3`.
> We will not change ComfyUI's copy. Choose a separate environment, or fix
> it yourself and re-check.

**A conflict is not a failure of the install — it is the install working
correctly.** The constraints file makes it unreachable for pip to
downgrade anything already present, so the only possible outcome is refusal
and a message. There is no third path and no resolver.

**Both sources are consulted, because they answer different questions.**
ComfyUI's `requirements.txt` says what ComfyUI *needs*; its venv says what
is *actually there*. Those diverge — a user upgrades one and not the other,
patches a file, or installs something by hand — and the divergence is the
interesting case rather than an edge case. So:

| Installed | In `requirements.txt` | Treatment |
|---|---|---|
| matches | yes | known-safe; proceed |
| differs | yes | a **conflict**: ComfyUI's own declaration is broken. Name it, refuse. |
| any | **no** | **unknown**: installed but undeclared, so nothing vouches for it. |

The third row is where the user's point lands. An undeclared dependency is
not evidence of a working setup — it is the absence of evidence, and the
safe reading is to require an exact version match rather than a range. A
declared requirement is a contract ComfyUI publishes; an installed package
is just a fact about the machine, and the two do not deserve the same
confidence.

So the check reports three outcomes, not two: *safe*, *conflict* (declared
and violated) and *unknown* (undeclared, so pinned strictly). Treating
unknown as safe is how a soft install quietly becomes the hard one.

### Screen 3 — the paths, as now

ComfyUI directory and model locations, pre-filled from resolution. Smaller
than it looks: screen 2 already decided the venv, so this screen is
mostly confirming a value the server already knows.

---

## 4. ComfyUI: required today, on maintenance grounds

Asked during the first build, answered by reading the source.

**Five modules of 92 node files import comfy's own code**, and they are the
load-bearing ones:

| File | Imports | Needed for |
|---|---|---|
| `nodes/model/unet_wrapper.py` | `comfy.ldm.modules.diffusionmodules.openaimodel` → `UNetModel` | constructing the UNet |
| `nodes/model/clip_encoder.py` | `comfy.sdxl_clip` → `SDXLClipModel`, `SDXLTokenizer` | the text encoder |
| `nodes/model/vae_decode.py` | `comfy.ldm.models.autoencoder` → `AutoencoderKL` | the VAE |
| `nodes/model/attention_checkpointing.py` | `comfy.ldm.modules.attention` | patch target (wraps, does not own) |
| `nodes/model/gradient_checkpointing.py` | `comfy.ldm.modules.diffusionmodules.util` | patch target (wraps, does not own) |

Without those, there is no model to train. The rest is genuinely
independent: node discovery is this project's own tree, the trainer's cwd
is the project root rather than ComfyUI's directory, and `comfy_dir` is
optional throughout — `ProjectLayout.from_paths_module()` catches the
failure and substitutes the project root.

So ComfyUI supplies **the model architecture code**, and nothing else in
this project duplicates it.

### The vendoring question, measured

Two modules in this project already *look* like vendored copies, and it is
worth being precise about what they are:

* `gradient_checkpointing.py` — a ~50-line `torch.autograd.Function`
  wrapping comfy's `CheckpointFunction`
* `attention_checkpointing.py` — sets a sentinel attribute on comfy's
  `BasicTransformerBlock` at runtime

Both are **extensions, not replacements**. Each needs comfy's class present
in order to wrap or patch it. Neither contains a copy of anything.

A genuine fork, measured by walking the relative-import closure from the
four modules this project imports:

| Component | Lines | Nature |
|---|---:|---|
| `comfy/ldm/modules/diffusionmodules/openaimodel.py` | 927 | model definition |
| `comfy/ldm/modules/attention.py` | 1,335 | model definition |
| `comfy/ldm/modules/diffusionmodules/util.py` | 306 | checkpointing — wrapped, never called |
| `comfy/ldm/modules/sub_quadratic_attention.py` | 276 | attention backend |
| `comfy/ldm/models/autoencoder.py` | 280 | model definition |
| `comfy/ldm/util.py` | 197 | helpers |
| `comfy/ldm/modules/sdpose.py` | 130 | pose head |
| `comfy/sdxl_clip.py` | 95 | model definition |
| `comfy/utils.py` | 1,535 | of which this project uses **20 lines** |

**9 modules, 5,081 lines**, split as:

* **582 lines** of checkpointing this project *wraps* and never calls — a
  fork would not need them, since the wrappers keep working against
  torch's own `checkpoint`.
* **2,964 lines** of model definitions a fork would have to own.
* **1,535 lines** of `utils.py`, of which two functions totalling 20 lines
  are used.

Third-party surface the fork inherits: torch, numpy, einops, PIL,
safetensors, tqdm, and `comfy_aimdo`. **Every one except `comfy_aimdo` is
already in this project's training four.**

### Why a fork is not worth it — but a clean implementation might be

1. **Licence, correctly stated.** ComfyUI is GPL-3.0, so *copying its
   source* into this project carries that licence with it. That is the
   narrow claim, and it is the only one about copying.

   **It is not a claim about the architecture.** SDXL is published — there
   is a paper, a public checkpoint, and an Apache-2.0 reference
   implementation — so its structure can be implemented from the
   specification without touching ComfyUI at all. And this project already
   demonstrates the point: `unet_wrapper.py:42` carries `SDXL_CONFIG`,
   the published architecture (320 model channels, `channel_mult [1,2,4]`,
   `adm_in_channels 2816`, …), written down in this repository's own words
   before ComfyUI is imported on the next line.

   So the real question is not "fork or vendor". It is whether writing
   SDXL's UNet and text encoder from the spec is worth it — a maintenance
   question, not a licensing one. What this project needs from a
   checkpoint is a **key-to-shape mapping**, and that is data: the
   published config plus the `.safetensors` file's own keys. Nothing about
   it requires Comfy's code to be the code that reads it.

   Recorded here because the first version of this document got it wrong in
   the direction of "GPL therefore impossible", which conflates copying an
   implementation with implementing a published architecture.

2. **The fork's cost, which stands regardless.** These files track ComfyUI's
   releases, weekly. A fork is 2,964 lines to keep in sync with an upstream
   that moves often, and the divergence would be silent — a ComfyUI release
   adding a LoRA-compatible layer leaves the fork behind, with no error
   anywhere.
3. **A directory dependency is the cheapest dependency today.** ComfyUI
   already has to exist for anyone using it for its actual purpose. Making
   it required adds no work for that user and removes a class of divergence
   bug for everyone.

**Conclusion: ComfyUI stays required *for now*, on maintenance grounds.**
The honest framing is that this is a deferral, not a proof. It is a
directory dependency rather than a pip dependency either way, because
`comfy` is a checkout that must be on `sys.path` — `comfyui` is absent from
the venv's distribution list even on a machine where ComfyUI is fully set
up — so it cannot be installed into any venv by any means. That is why the
wizard's first question is "where is it", not "do you have it".

Separating this project from ComfyUI is the next piece of work after the
installer, and §7 below says what it would take.

One genuine finding: `clip_encoder.py:141` and `unet_wrapper.py:207` import
`Timestep` from `comfy.model_base`, but ComfyUI itself imports it from
`comfy.ldm.modules.diffusionmodules.openaimodel`. The import path is wrong
on our side; `Timestep` is defined at `openaimodel.py:41` and is 41 lines.
That is a 41-line vendoring candidate and the only one worth taking — it
removes a `model_base` import (which drags in far more) for one small
class. **Not urgent**: it works today.

---

## 5. Torch is backend-specific, and the installer must treat it so

`torch` on PyPI is the CUDA build. Intel's XPU build is published from a
different index. So "install torch" is ambiguous, and an installer that
treats it as one package gets it wrong on any non-NVIDIA machine.

This is why the GPU question comes **before** the install and not after:

1. Enumerate what is present. `torch.xpu.device_count()` and
   `torch.cuda.device_count()` behind one port, **in a subprocess** — both
   initialise a driver context, and importing either costs seconds.
2. Show what was found, by name, with its VRAM. `Intel(R) Arc(TM) B580
   Graphics — 12,216 MB` is a fact; "XPU available" is not.
3. Choose the index URL from that, and record it.

The enumeration is **lazy**: not asked until the user reaches that step.
Asking at page load would cost 1.7 s on every render for an answer most
users never look at.

### Unsupported hardware is a future feature, not a peer option

The first build's instinct — offer `cuda`/`xpu` as a manual override beside
the detected device — was wrong, and the reason is worth recording.

This project is **Intel Arc B580 only**, with `xpu` hardcoded
([ADR 0004](../decisions/0004-b580-only.md)). A user who installs CUDA torch
on an unsupported card gets 3 GB downloaded, a multi-gigabyte wheel chosen
for hardware the project will refuse to train on, and a failure that
surfaces at run time rather than at install time. So the override:

* is **not** offered in the same list as the detected device — it sits
  below a divider, under its own heading;
* **states that CUDA is not supported yet**, and that the option exists so
  the choice is visible and forward-compatible rather than absent;
* carries the size, so the cost is known before it is chosen.

This is a small UX decision with a real principle behind it: an option that
is known to lead to a failure should say so at the point of choice, not
rely on the user having read a document.

---

## 6. What this document changes about what is built

| Built | Verdict |
|---|---|
| requirement manifest, four tiers | **keep** — it is what sizes screen 2 |
| readiness report + device probe | **keep** — screen 1 |
| first-run state, the `installer_not_allowed` gate | **keep** — screen 3, and the security decision stands |
| `/setup` wizard, two screens | **extend** to three, insert the venv/GPU screen between them |
| `never_install` on torch | **replace** — should be conflict-*detectable*, not forbidden; screen 2's conflict check is the escape hatch the flag lacks |
| ADR 0005's install-order argument | **withdraw** — the conclusion was right, the ordering argument was wrong |
| `run_server.sh` unchanged | **done** — the dispatch is `backend/first_run.py` (§2) |

`path_tiers` was fixed as part of the first build and is unrelated to any
of this; it stays.

---

## 7. Separating this project from ComfyUI (next, not now)

The conclusion in §4 is a deferral, so this says what ending it would take.
Three steps, in dependency order, each independently useful.

### 7.1 Stop importing `Timestep` from `model_base`

A genuine bug in the current code, not a design question:
`clip_encoder.py:141` and `unet_wrapper.py:207` import `Timestep` from
`comfy.model_base`, but ComfyUI itself imports it from
`comfy.ldm.modules.diffusionmodules.openaimodel` (defined at line 41, 41
lines). Our path is wrong *and* it drags in `model_base`, which is one of
the largest modules in the tree.

Copying 41 lines of a published embedding class is the one vendoring
candidate worth taking. It removes a wrong import and a heavy transitive
edge at the cost of a small, attributable file.

### 7.2 Own the two `utils` functions

`state_dict_prefix_replace` (11 lines) and `clip_text_transformers_convert`
(9 lines) are used by `clip_encoder.py` for checkpoint-key translation.
They are data manipulation with no architectural content — the kind of
thing that is written from the checkpoint format, not from an
implementation.

### 7.3 Own the SDXL model definitions

This is the real work: `UNetModel`, `SpatialTransformer`,
`BasicTransformerBlock`, `SDXLClipModel`, `AutoencoderKL` — the 2,964
lines measured in §4, implemented from the published SDXL specification.

**What is already done, which is most of the hard part.** The architecture
is written down in this repository today: `SDXL_CONFIG` in
`unet_wrapper.py` is the published UNet configuration, not a transcription
of ComfyUI's. The checkpoint contract is likewise already fixed — the
project loads `.safetensors` state dicts and needs a key-to-shape mapping
that the published config plus the checkpoint's own keys fully determine.

So this is *not* reverse-engineering. It is implementing a published
architecture against a known data contract, with an existing
Apache-2.0 reference implementation available for the shapes.

**What it would cost.** Six modules, plus the test surface for them: every
existing LoRA injection point is written against Comfy's module layout, so
`lora.py`, `adapter_injection.py` and the phase-splitting code all move at
once. The VRAM measurements in `docs/known-issues/` were taken with
Comfy's implementation resident and would need re-taking — a reimplementation
that allocates one extra tensor mid-forward would change the floors those
documents assert.

**Sequencing.** 7.1 and 7.2 are small and independent; 8.3 is not, and
starting it before the installer is finished would put the two hardest
pieces of work in flight at once. The installer's conflict check also gets
*easier* the moment ComfyUI is optional, because "use ComfyUI's venv"
becomes one option among several rather than the only cheap one.

### What would make it wrong

That the reimplementation diverges numerically from Comfy's, in a way the
existing tests do not catch. Every test here is a shape, a count, or a
memory number; none of them compares an output tensor against Comfy's. A
first step for 7.3 is a numerical equivalence test against Comfy's
implementation — run both, compare forward outputs on a fixed input, and
refuse the fork if they disagree. That test is also what proves the
separation is safe, and it does not exist yet.

## 8. Open questions, not decided here

* **Whether the conflict check reads ComfyUI's `requirements.txt` or asks
  its venv.** *Answered in §3: both, because they answer different
  questions.* The file is what ComfyUI needs, the venv is what is there,
  and a divergence is the interesting case rather than an edge case — three
  outcomes, not two, because installed-and-undeclared is neither safe nor a
  conflict but *unknown*, and unknown gets a strict pin.
* **Progress reporting for a 2.5 GB download.** A job id plus polling is
  the honest minimum; SSE is available and would be better. Either way this
  is the first stateful thing in the installer, and it is where the
  "no third path on failure" rule gets hardest to honour.
* **Whether the GPU list is persisted.** A machine with two cards has a
  genuine choice, and re-asking on every server start is worse than
  remembering — but remembering makes a hardware change silent.