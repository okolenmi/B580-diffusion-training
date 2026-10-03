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

**What it found, measured against the real checkout.** 185 packages in
ComfyUI's venv, 0.16 s to read both sources. 35 are declared and satisfy
their declaration. **150 are installed without being declared** — most of a
real venv is transitive dependencies ComfyUI's file never mentions. **None
conflict.** And all four of this project's server packages are *absent from
ComfyUI's file entirely*, so none of them can break a declaration: adding
them is clear by measurement, not by argument.

That 150 is the number that justifies the pin. "Constrain what is declared"
would leave four fifths of the venv free for pip to move as collateral.

### What is built

`backend/application/ports/comfy_environment.py` reads both sources;
`check_comfy_conflicts.py` classifies them; `GET /api/v1/installer/conflicts`
serves the report. Every row carries its outcome *and* a sentence, so the
wizard does not write prose that can drift from the rule.

Two decisions inside it that the outline above did not settle:

* **`prereleases=True` when comparing.** This machine has numpy `2.5.0rc1`
  against a declared `>=1.25.0`, and PEP 440 says a prerelease does not
  satisfy a `>=` range. Reporting that as a conflict would refuse a correct
  install and tell a user their ComfyUI is broken while it is running. The
  rule exists to stop an installer *choosing* a prerelease, which is not what
  this check does — it reads a version someone already chose.
* **`venv_python` is a per-call argument, resolved from settings.** It is a
  setting the wizard sets and the user can change. A port that captured it
  at wiring time would report on whichever venv happened to be configured
  when the server started, and would do so silently.

An unreadable source is `200` with `checked: false, safe: false`, never a
pass. A report that cannot tell "we could not check" from "we checked and
it is fine" is the failure mode worth designing against.

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
| the conflict check, three outcomes | **done** — `GET /api/v1/installer/conflicts` (§3) |

`path_tiers` was fixed as part of the first build and is unrelated to any
of this; it stays.

---

## 7. Separating this project from ComfyUI (next, not now)

The conclusion in §4 is a deferral, so this says what ending it would take.
Three steps, in dependency order, each independently useful.

### 7.1 Stop importing `Timestep` from `model_base`

**Two steps, and the first is done.** `clip_encoder.py` and
`unet_wrapper.py` imported `Timestep` from `comfy.model_base`. ComfyUI
defines it in `comfy.ldm.modules.diffusionmodules.openaimodel` and
*re-exports* it from `model_base`, so the definition is in one place and the
alias in the other — and `model_base.Timestep is openaimodel.Timestep`
measures `True`. The import was therefore never wrong in what it returned.

It was wrong in what it cost, and that is measurable:

| | time | modules loaded |
|---|---:|---:|
| `from comfy.model_base import Timestep` | 4.32 s | 3323 |
| `from ...openaimodel import Timestep` | 3.05 s | 2422 |

**1.27 s and 901 modules per process**, for a class that is seven lines
long, in an import that runs lazily on the first forward pass rather than
at module load — so it is also a latency spike at a point where a training
step is already waiting on the device. Both call sites now import from
`openaimodel`.

This section previously said the class was "41 lines, defined at line 41".
Both numbers were wrong, and the line 41 citation pointed into
`forward_timestep_embed` rather than at the class. It is seven lines at
line 360.

**The second step is done too**: `nodes/model/timestep_embedding.py`
reimplements the embedding and its seven-line wrapper, so neither call site reaches into
`openaimodel` for this. The output is **bitwise identical** to the previous
implementation for every dtype either call site uses — verified against
ComfyUI's class directly, not just against the reference formula.

It is written from the published definition (Ho et al.; the same closed
form in guided-diffusion and diffusers), not copied, and two things were
left out deliberately:

* **ComfyUI's `repeat_only` branch.** It returns
  `repeat(timesteps, 'b -> b d')` — not an embedding, the timestep numbers
  tiled. Neither call site wants it, and a caller reaching for it is asking
  for something its name does not describe. ComfyUI's own `UNetModel` does
  use it, so this module is deliberately not a drop-in for `openaimodel`
  and is not trying to be.
* **ComfyUI's unvalidated `dim`.** `dim=1` divides by `dim // 2` with no
  check and fails as a broadcast error between `(N, 1)` and `(1, 0)`. Here
  it is a `ValueError` that says the minimum is 2.

**Two claims this exposed, both now pinned by the test.** `Timestep` has no
parameters and no buffers, so `.to(device=..., dtype=...)` on it does
nothing at all — not even the device half. Both call sites wrote exactly
that, implying a control they did not have; the output is float32 for every
input dtype because that is the algorithm's precision, and the device comes
from the input tensor. And `_EMBEDDER_CACHE` in `unet_wrapper.py` was a dict
keyed by `(device, dtype)` "to save VRAM and time", holding a module with
zero bytes to save. One module now.

This is characterisation, not an equivalence gate, as the next section
argues. The test records the difference if there is one and moves on, and
skips rather than fails when ComfyUI is not installed — a test that failed
for their absence would put the coupling straight back in.

### 7.2 Reimplement the two `utils` functions

**Done, and it was three functions rather than two.** `clip_encoder.py`
used `state_dict_prefix_replace` and `clip_text_transformers_convert` from
`comfy.utils`, and the second calls a third, `transformers_convert`, so the
real dependency is about sixty lines rather than the twenty this section
originally claimed. `import comfy.utils` costs **2.71 s and 2136 modules**
on this machine, lazily, at model-load time.

`nodes/model/clip_state_dict.py` now owns all three. Verified equal to
ComfyUI's on a real SDXL checkpoint's key set — 587 conditioner keys in,
715 out, identical key set and identical shapes — and equal value-for-value
on synthetic cases including both `text_projection` spellings.

Two behaviours are reproduced deliberately rather than tidied, because both
are things a cleanup would break:

* **The QKV split stores views, not copies.** `attn.in_proj_{weight,bias}`
  is one fused 3× tensor; slicing it leaves q/k/v aliasing one buffer, at
  offsets 0, third, two-thirds. A `.contiguous()` here would be a behaviour
  change.
* **`filter_keys=True` is not a filter in the usual sense.**
  `state_dict_prefix_replace` returns a *different* dict holding only the
  renamed keys, while the caller's dict keeps the unmatched ones and loses
  the matched ones — a key ends up in exactly one of the two. We pass
  `filter_keys=True` where ComfyUI passes `False`, which looked like it
  would silently drop keys. Checked on a real checkpoint rather than
  assumed: all 587 `conditioner.` keys match one of the two prefixes
  (197 CLIP-L, 390 CLIP-G, zero unmatched), so nothing is dropped and the
  two settings coincide. Recorded at the call site.

### 7.3 Reimplement the SDXL model definitions

This is the real work: `UNetModel`, `SpatialTransformer`,
`BasicTransformerBlock`, `SDXLClipModel`, `AutoencoderKL` — implemented from
the published SDXL specification.

**The remaining surface is five imports, measured rather than estimated.**

| Call site | Imports | Module |
|---|---|---|
| `unet_wrapper.py:79` | `openaimodel.UNetModel` | 927 lines |
| `attention_checkpointing.py:139` | `attention.BasicTransformerBlock` | 1,335 lines |
| `attention_checkpointing.py:154` | `util.checkpoint` | 306 lines |
| `gradient_checkpointing.py:205` | `util.CheckpointFunction` | (same file) |
| `clip_encoder.py:79` | `sdxl_clip.SDXLClipModel`, `SDXLTokenizer` | 95 lines |
| `vae_decode.py:84` | `autoencoder.AutoencoderKL` | 280 lines |

Import *cost* does not discriminate — all five cost 2.8–3.8 s and ~2,200
modules because they share the same base. What splits them is dependency and
whether the removal is independently observable.

**Two costs this section overestimated, both now measured.**

* *"Every existing LoRA injection point is written against Comfy's module
  layout, so `lora.py`, `adapter_injection.py` and the phase-splitting code
  all move at once."* They do not. Nothing in the LoRA stack imports
  `comfy` at all: `LoRALinear` is our own `nn.Module`, and injection
  selects targets by **module name** (`to_q`, `to_k`, `to_v`, `to_out.0`).
  So the contract is the published SDXL layout plus the checkpoint's own
  keys — which we already have to honour — and not ComfyUI's class
  identities. The LoRA code does not move.
* *2,964 lines.* That counts whole ComfyUI files. What is actually
  reachable is smaller, and much of it is not SDXL at all.

**The split, in three sections, each removing an import on its own.**

* **A — the diffusion path. Done.** `checkpoint()` + `CheckpointFunction`
  (part 1), the attention stack (part 2), then `UNetModel`,
  `TimestepEmbedSequential`, `Upsample`, `Downsample`, `ResBlock`,
  `TimestepBlock` (part 3). Removes three of the five imports.
  `AttentionBlock` turned out not to be needed: SDXL's UNet only uses
  `SpatialTransformer`, and the `AttentionBlock` classes in ComfyUI belong to
  the genmo and wan VAEs.
* **B — the VAE. Done.** `AutoencoderKL`, `Encoder`, `Decoder`,
  `ResnetBlock`, `AttnBlock`, `Upsample`, `Downsample`,
  `DiagonalGaussianDistribution` — in `nodes/model/vae.py`. Removes one.
  Independent of A, and it turned out to be: nothing in it touches the UNet.

  ComfyUI splits this across two files, `ldm/models/autoencoder.py` and
  `ldm/modules/diffusionmodules/model.py`. Most of the second is video —
  `conv3d`, `CarriedConv3d`, `conv_carry_causal_3d`, `time_compress`, the
  5-D branches — and none of it applies to an image VAE. `AttnBlock` turned
  out to be needed after all, because `mid.attn_1` is built
  unconditionally in both halves while SDXL configures
  `attn_resolutions: []`: two attention blocks in a VAE whose config names
  no attention.

  Two things that are easy to get wrong and are reproduced deliberately.
  `Downsample` pads *asymmetrically*, `(0, 1, 0, 1)`, which a symmetric
  stride-2 convolution cannot express — same shape, half-pixel shift per
  level, four levels of it. And `AttnBlock` treats **channels as features
  and `H*W` as the sequence**, the reverse of the UNet's attention in
  `attention.py`. Both give the right shape either way, so neither is
  catchable by a shape test; `smoke_test_vae.py` checks both by value and
  proves each check can fail.
* **C — CLIP. Done, in two parts.** C1 is `nodes/model/clip.py`: both text
  towers, with the two text configs inlined as dicts rather than read from
  ComfyUI's `sd1_clip_config.json` and `clip_config_bigg.json` — 1.1 KB of
  published architecture description, and inlining it leaves no path into
  ComfyUI's *tree* for the model half. C2 is `nodes/model/tokenizer.py`:
  OpenAI's byte-level BPE and the sequence layout around it.

  C1: 716 tensors with identical names and shapes against ComfyUI, and the
  context and pooled outputs bitwise identical on four token layouts.

  C2 is the one piece whose contract is *data* rather than code, and it
  agrees with ComfyUI's tokenizer token-for-token over a 25-prompt corpus —
  empty string, single characters, mixed case, whitespace runs,
  contractions, digits, accented Latin, CJK, Cyrillic, emoji, a
  300-character word, and prompts long enough to span several sections.

  **Three deliberate divergences**, all cases where ComfyUI has prompt
  grammar this project does not have, or where HuggingFace has drifted from
  CLIP. All three are pinned by `smoke_test_tokenizer.py`:

  1. `(word)` is text, not a weight group. ComfyUI's `token_weights` runs
     `parse_parentheses` and consumes a balanced `(...)` as a weighting
     expression, so `"a (b) c"` reaches its tokenizer with the parentheses
     gone. All 32 ASCII punctuation marks *between letters* do agree, which
     is what says this is grammar rather than a bug in the alphabet.
  2. A literal `<|startoftext|>` in the prompt is text. CLIP's split pattern
     passes the special tokens through as ids, so a prompt containing them
     gets a second BOS in the middle.
  3. HTML entities are unescaped, twice — CLIP's `basic_clean`. The
     installed `transformers` 5.12.1 has dropped that step, so ComfyUI now
     tokenizes `&lt;` as four tokens where CLIP's own tokenizer gives one.
     It also no longer calls `ftfy.fix_text`, which this project could not
     do anyway: ftfy is not installed and not declared.

  **The vocabulary's location is the one thing left open.** The merges and
  ids are OpenAI's published CLIP BPE data, not ComfyUI's work, but the copy
  on this machine lives in ComfyUI's tree. `default_vocabulary_dir()` reads
  `$CLIP_TOKENIZER_DIR`, then `assets/clip_tokenizer/` in this repository,
  then ComfyUI's tree — the last being a fallback so the tokenizer works
  where it was developed, and a real dependency on someone else's *files*
  until the middle one exists. It is 1.6 MB of published data and the
  decision is a project's, not a technical one.

#### What we do not port, and why

ComfyUI's `BasicTransformerBlock.forward` is ~90 lines, of which about 15
are transformer arithmetic. The rest is a model-patching framework —
`transformer_patches`, `transformer_patches_replace`, `attn1_patch`,
`attn2_patch`, `middle_patch`, `attn1_output_patch`, `block`, `block_index`,
`extra_options`, `switch_temporal_ca_to_sa`, `disable_temporal_crossattention`
— which exists so one class can serve Flux, SD3, Hunyuan and SDXL from
ComfyUI's own checkpoint format. SDXL uses none of it. Our version is the
arithmetic, and `attention_checkpointing.py`'s patch becomes correspondingly
smaller because it patches a small function rather than a dispatch table.

Likewise `operations.Linear` / `operations.GroupNorm` / `operations.Conv2d`
are ComfyUI's dtype/device dispatch wrappers. This project passes device and
dtype explicitly at construction and never through them, so ours are
`nn.Linear` and friends.

#### Bugs and quirks found in ComfyUI while porting, none copied

Five, all measured rather than inferred:

1. **`CheckpointFunction` raises on frozen parameters**, and re-enters a
   CUDA-only autocast in backward. Both fixed in `nodes/model/checkpoint.py`,
   the second measured as a 4.581e-04 relative gradient error becoming 0.0.
   Both reported upstream.
2. **`SpatialTransformer`'s `is_linear=True` branch** builds `proj_out` as
   `Linear(in_channels, inner_dim)` — identical to `proj_in`, roles never
   swapped — so it cannot work where `inner_dim != in_channels`. Invisible
   for SDXL only by coincidence: `dim_head = num_head_channels = 64` and
   `num_heads = ch // 64` give `inner_dim == ch` at every level, so both
   projections come out square. Measured on the real UNet: 11
   SpatialTransformers, every one e.g. `Linear(640, 640)`. **And SDXL does
   use this branch** (`use_linear_in_transformer: True`), so the coincidence
   is load-bearing rather than incidental.
3. **`BasicTransformerBlock.__init__` takes `inner_dim` as a parameter
   defaulting to `None`**, and `ff_in` is `ff_in or inner_dim is not None`.
   Recomputing `inner_dim = n_heads * d_head` instead looks equivalent and
   is not — it makes `ff_in` permanently true and adds a `norm_in` and an
   `ff_in` FeedForward whose weights are in no checkpoint. The real UNet
   settles it: 70 blocks, none with `norm_in`.
4. **`comfy.ops.Linear` leaves its weight uninitialised** (~3e29 when
   constructed and read), because ComfyUI assumes a checkpoint always
   overwrites it. Harmless there; it makes any naive comparison produce NaN
   on both sides, where `nan == nan` passes everything.
5. **`self.ff_in` is a bool before the branch and a `FeedForward` after it.**
   One attribute, two meanings, which the forward's `if self.ff_in:`
   happens to tolerate. Split into `has_ff_in` and `ff_in`.

One quirk was inherited rather than fixed, because it is load-bearing:
`label_emb` is wrapped in a redundant `nn.Sequential`, so its keys are
`label_emb.0.0.*` and `label_emb.0.2.*`. Unwrapping renames all four
tensors, and `load_state_dict(strict=False)` would report them missing and
carry on.

### What was verified, and how

The contract is the checkpoint, so the claims are mechanical and were
checked mechanically:

| | result |
|---|---|
| `SpatialTransformer` `state_dict` vs ComfyUI, every SDXL level | identical keys and shapes, 46 / 206 / 206 params |
| attention forwards vs ComfyUI, both projection branches | **bitwise identical** |
| `UNetModel` `state_dict` vs ComfyUI, real SDXL config | **1680 tensors, identical names, identical shapes** |
| UNet forward vs ComfyUI, small config with every SDXL feature | **bitwise identical** |
| a real SDXL checkpoint into our `UNetModel` | **0 missing, 0 unexpected** |
| `AutoencoderKL` `state_dict` vs ComfyUI, SDXL's config | **248 tensors, identical names and shapes** |
| VAE `encode` and `decode` vs ComfyUI, three configs | **bitwise identical** |
| a real SDXL checkpoint into our `AutoencoderKL` | **0 missing, 0 unexpected** |
| our tokenizer vs ComfyUI's, 25 prompts x 2 towers | **token-for-token identical** |
| `SDXLClipModel` `state_dict` vs ComfyUI | **716 tensors, identical names and shapes** |
| CLIP context and pooled, four token layouts | **bitwise identical** |

And end to end on the B580, from a real 6.9 GB checkpoint, with no ComfyUI
code in the process at all: CLIP produces a `(1, 77, 2048)` context and a
`(1, 2816)` `y`, the UNet loads with 0 missing and returns the latent's
shape, and the VAE decodes a 64x64 latent to a 512x512 image — each
measured alone, because the UNet is 10.3 GB in float32 against an 11.93 GB
card. That is `nodes/smoke_tests/gpu/smoke_test_reimplemented_stack.py`,
and it runs in the gate.

**The remaining ComfyUI imports in the tree are all inside the
characterisation tests**, which compare against it and skip when it is
absent. There is none in the live code.

That last one is the claim that matters, and `smoke_test_unet.py` re-runs it
whenever a checkpoint is present, skipping rather than failing when it is
not.

#### A misdiagnosis, recorded because it nearly became a change

`test_a_killed_child_reports_no_outcome` failed once in a gate run: the child
had run its graph and written a clean outcome while the SIGKILL was still in
flight. I diagnosed it as the fork/execve window — between `fork` and
`execve` a child's `/proc/<pid>/cmdline` is a copy of the parent's, which
does not contain `CMDLINE_MARKER`, so `_signal` refuses to signal what is
actually our own child. I wrote a fix to `graph_task_gateway.py` on that
basis, and a measurement appeared to support it: 40 of 40 freshly spawned
children read as `False`.

Both the diagnosis and the measurement were wrong.
`cmdline_mentions` *already* handles this: it compares the child's cmdline
against the parent's own and returns `None` — "cannot judge identity yet" —
which `_signal` treats as signal-me and only `False` refuses. And the 40 of
40 were all **post**-exec, where `False` is the correct answer. The fix was
reverted; nothing in `backend/` changed for it.

The actual cause was the margin the test assumed rather than had: it killed a
one-node graph, betting on the ~2 s of torch import before the child could
do anything. Its sibling signal tests use 4,000 nodes and are not in that
position. The graph is now 4,000 nodes, and disabling the kill makes the
check fail, so it has teeth rather than merely passing.

### What would make it wrong

That the reimplementation diverges numerically from Comfy's, in a way the
existing tests do not catch. Every test here is a shape, a count, or a
memory number; none of them compares an output tensor against Comfy's.

**A characterisation test, not an equivalence gate.** The obvious thing to
propose is: run both implementations on a fixed input, compare outputs,
and *refuse the fork if they disagree*. That framing is wrong, and not
hypothetically.

ComfyUI's `CheckpointFunction` is wrong twice, independently, and this
project has fixed both while ComfyUI has fixed neither:

| | ComfyUI | This project |
|---|---|---|
| frozen parameters in a checkpointed block | raises `One of the differentiated Tensors does not require grad` | filtered; frozen params get `None` |
| the backward's autocast context | re-enters `torch.cuda.amp.autocast`, which on an Intel card warns "Disabling autocast" and enters disabled — so an fp16 forward is recomputed in fp32 | re-enters the forward's own autocast, on the forward's device type |

Both measured, not inferred. The autocast one shows as 4.6e-04 relative
gradient error against a non-checkpointed reference on this B580, and 0.0
after the fix. Neither has landed upstream: nothing tracks this project,
so there is nothing to land it.

So "disagrees with ComfyUI ⇒ refuse" would have **blocked the fix**. The
test to write is a **characterisation** test — record what ComfyUI does,
then decide independently whether that is correct. Divergence is a prompt
to work out which side is wrong, not a failure condition. The same applies
to the LoRA injection points: they are written against Comfy's module
layout because that layout is what the checkpoints use, not because
Comfy's behaviour is the target.

It does not exist yet, and it is a prerequisite for 7.3 — not because it
gates the fork, but because a reimplementation nobody has compared against
anything is not a reimplementation, it is a second guess.

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
* **Whether the GPU list is persisted.** *Answered: not the list, and the
  choice is narrower than it looks.* Measured, `importlib.metadata` on this
  machine: there is **one** distribution named `torch`, and the
  accelerator is a local version tag — `2.12.1+xpu`. The backend lives in
  the wheel, not in anything selectable at run time, and pip resolves by
  distribution name, so installing the CUDA build **replaces** the XPU one
  rather than sitting beside it.

  So "which GPU" was the wrong question. Two consequences:

  - **Architecture is not a choice.** It is decided by which wheel is
    installed, once, and it is a platform decision rather than a per-device
    one. CUDA remains a future feature for exactly the reason it always
    was — it is a different index and a different wheel, not a different
    selection on this machine.
  - **Mixed-architecture machines cannot be offered this at all.** Within
    one torch build every device is the same architecture, so a machine
    with an Intel card *and* an NVIDIA card is not a machine with two
    choices — it is two environments, and one of them would be a lie. The
    wizard reports what torch actually has, which is one backend.

  What remains is the **device index**: which card training runs on, offered
  only when there are two or more. That is persisted, because re-asking on
  every server start is worse than remembering, and a changed index is
  visible on the settings page where the resolved value is shown — so the
  "hardware change becomes silent" worry is answered by showing it, not by
  re-asking.