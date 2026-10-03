# 12. Installer design, and the cost of not needing ComfyUI

*[← design index](README.md)*

**Status: design, not built.** This is the reasoning behind a proposed
rebuild of the first-run installer, plus a measured answer to a question
that came up while building the first version: *could this project drop its
dependency on ComfyUI?* The answer is below, with numbers.

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
  ├─ python -c "import fastapi, uvicorn, multipart, tomli_w"   # fast, no deps
  │    └─ fails →
  │        └─ python -m backend.preflight        # one file, stdlib only
  │             ├─ prints the URL, opens the browser
  │             ├─ serves ONE page on 8767 (never 8766)
  │             └─ installs the four into a temporary venv, then says
  │                "restart run_server.sh"
  └─ succeeds → the real server starts, and /setup takes over
```

**The preflight gets its own port and is never reachable once the server
runs.** It is a bootstrap, not a second interface, and having two servers
answering on loopback at once would be a worse problem than the one it
solves. It exits with a distinct code so `run_server.sh` can say something
better than a traceback.

**The temporary venv is disposable by construction.** It holds four small
packages and nothing else; it is created under the system temp directory,
named with the pid, and removed on preflight exit. It is not where the
project lives, it is not on any `sys.path` the server uses afterwards, and
"did it get left behind" is answerable by looking in one place. The user's
instruction that it be separable and removable is satisfied by construction
rather than by a cleanup path someone has to remember.

This also settles the console-message question honestly: there genuinely is
a terminal moment, and it is exactly the moment the server cannot start.

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

### Screen 3 — the paths, as now

ComfyUI directory and model locations, pre-filled from resolution. Smaller
than it looks: screen 2 already decided the venv, so this screen is
mostly confirming a value the server already knows.

---

## 4. ComfyUI: required, not optional

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

### Why a fork is not worth it

1. **Licence.** ComfyUI is GPL-3.0. Vendoring its model definitions into
   this project makes that code GPL-3.0 here. There is currently no
   licence file at this repository's root, so the interaction is undefined
   rather than merely permissive.
2. **The maintenance cost is the real cost.** These files track ComfyUI's
   releases. A fork is 2,964 lines to keep in sync with an upstream that
   moves weekly, and the divergence would be silent — a ComfyUI upgrade
   that adds a LoRA-compatible layer would leave the fork behind, with no
   error anywhere.
3. **A directory dependency is the cheaper dependency.** ComfyUI already
   has to exist for anyone using it for its actual purpose. Making it
   required adds no work for that user and removes a class of divergence
   bug for everyone.

**Conclusion: ComfyUI stays required.** It is a directory dependency, not a
pip dependency, because `comfy` is a checkout that must be on `sys.path` —
`comfyui` is absent from the venv's distribution list even on a machine
where ComfyUI is fully set up. So it cannot be installed into any venv by
any means, and the wizard's first question has to be "where is it", not "do
you have it".

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
| `run_server.sh` unchanged | **replace** with the preflight dispatch |

`path_tiers` was fixed as part of the first build and is unrelated to any
of this; it stays.

## 7. Open questions, not decided here

* **Where the preflight's page lives** if the server never starts on any
  port — a separate static bundle, or the same `frontend/` tree served by
  the preflight. The first is smaller and cannot drift from the app; the
  second is one codebase and needs the server's own assets to load.
* **Whether the conflict check reads ComfyUI's `requirements.txt` or asks
  its venv.** The file is simpler and is what a user reads; the venv is what
  is actually installed. They can disagree — a user edits one and not the
  other — and that disagreement is the interesting case.
* **Progress reporting for a 2.5 GB download.** A job id plus polling is
  the honest minimum; SSE is available and would be better. Either way this
  is the first stateful thing in the installer, and it is where the
  "no third path on failure" rule gets hardest to honour.
* **Whether the GPU list is persisted.** A machine with two cards has a
  genuine choice, and re-asking on every server start is worse than
  remembering — but remembering makes a hardware change silent.