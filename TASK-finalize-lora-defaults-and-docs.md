# TASK: finalize the LoRA-quality work, harden the last commit, fix the docs

For the repo's AI agent. Read all of it before touching anything. Written against
repo HEAD `11adb0f` ("Latest work. Some may be unfinished.").
Order matters: **A (safety) -> B (gate) -> C (defaults, needs maintainer) -> D, E, F -> G**.

## 0. Ground rules (same as before, restated because they matter here)

1. **Every claim carries its evidence class:** *measured on real SDXL weights* /
   *measured on tiny random weights (CPU)* / *read in code only*. Only the first counts for
   "fixed" or "default is now safe".
2. **Negative results are results.** A test that finds nothing still gets a `MEASURED-*.md` entry.
   Never weaken, skip or delete a test to make a gate pass.
3. **One commit per item**, measurements in the commit body. Do not batch unrelated work into
   a commit named "Latest work".
4. **Do not change a default silently.** Items marked **[MAINTAINER DECISION]** end with you
   preparing both options and stopping to ask. Defaults change the behaviour of every existing run.
5. **You cannot run GPU work yourself in every environment.** When something needs the Arc B580,
   write the exact command, the expected output shape and the pass/fail rule into
   `docs/maintainer-queue.md` and move on. Do not guess results.
6. `docs/known-issues/README.md` rules apply: an entry lives in exactly one of
   open / resolved / pending-testing / deferred; entries are cited from source comments, so
   never delete or rename one without grepping for references first.
7. Out of scope here: the node-graph rework (see G, which is read-only).

## 1. State of play (verified by reading the repo; GPU numbers are the agent's own, from its commits)

* The training loop has been checked against diffusers / transformers / DDPMScheduler
  (UNet, LoRA export, gradients through checkpointing, VAE, CLIP, noising) and against kohya on
  real weights (step-0 loss 0.0-0.1% apart, latents x1.000). The loss path and data path are exonerated.
* The documented cause of the quality gap (`archive/bad-LoRA-investigation/MEASURED-targets-and-schedule.md`):
  1. cosine decay to 5% applied ~0.52x of the nominal LR (kohya default is constant);
  2. the trainer adapted 560 modules (attention only) vs kohya's 722 (adds `ff.net.0.proj`,
     `ff.net.2`, `proj_in`, `proj_out`); `target_modules` had been dead code.
  Presets now exist: `attention` (564 incl. cond path, still the default), `kohya_default`
  (722 + cond), `kohya_plus` (722 + the 4 `time_embed`/`label_emb` modules = 726; maintainer says
  it renders best).
* L4 (size-only batch grouping + per-sample prompts) and L5.3 (XPU graph capture, `use_xpu_graph`,
  default off) have landed. Their entries are **not** in `docs/known-issues/`.

### What that verdict does NOT yet show (this is why B exists)

* The A1 evidence is **one 512 px image, empty caption, rank 16/alpha 16, 500 steps, constant 1e-4.**
  The original complaint was a *different* symptom: a 200-image run, default settings, **rank 64 /
  alpha 48**, output looking like a collage of unrelated patches (broken denoising, not just
  blur / weakness). "Undertrained + attention-only" explains blur and weak style; it has **not**
  been shown to explain the collage. It might. Nobody has re-run the original failing setup.
* On one image with an empty caption the conditioning-path modules are the *only* global
  carrier (label_emb median |dW| 1.66). That is why they help there. On a many-image,
  many-caption, many-size dataset they may behave differently, and they were never tested
  outside this trainer's loader chain (ComfyUI works; diffusers / others unknown).

## A. Make the last commit safe  (do first, ~half a day)

`11adb0f` touched `clip_encoder.py`, `unet_wrapper.py`, `step_pipeline.py`, `supervised.py`.

A1. **Run every gate and record the numbers:** `python run_tests.py`, `python backend/tests/run_all.py`,
    `python scripts/check_quality.py`, plus `nodes/smoke_tests/smoke_test_xpu_graph.py`.
    If anything fails, fix or revert *that part* with an explanatory commit. Do not edit the test.

A2. **Meta-device construction is a silent-corruption hazard until proven otherwise.**
    The UNet and CLIP are now built on `meta` and materialised with `to_empty()`. Anything the
    checkpoint does not supply is now **uninitialised memory** (possibly NaN/Inf, different on every
    run), where before it was deterministic random init. Known case: `logit_scale` is on the
    "expected missing" allowlist in `clip_encoder.py` and will be garbage. The existing check
    ("bit-identical across 1680 state_dict tensors") only compares tensors that were *loaded*.
    Do this:
    * After every `load_state_dict(strict=False)` on a meta-built model, compute the set of
      parameters and buffers **not covered** by loaded keys. Allowlisted ones get a deterministic fill
      (zeros, or the original init value) immediately. Any other uncovered tensor => **raise**,
      naming the first 10 keys. `unexpected` keys should also raise unless allowlisted.
    * Add a CPU test with the tiny UNet/CLIP configs used in
      `archive/bad-LoRA-investigation/unet_vs_diffusers.py` and `clip_vs_hf.py`: build once eagerly,
      once via meta, load the same weights, compare **forward outputs** bit-for-bit (not state_dicts),
      including after LoRA injection. Add a negative test: delete one key from the state dict and
      assert construction fails loudly.
    * Tag: tiny weights, CPU. Real-weight confirmation goes to the maintainer queue.

A3. **`use_xpu_graph` guard rails (main route).** Confirm by test that: a fused optimizer raises a
    clear error (it does in code; add the test), Dropout active raises, and `skip_zero_grad` is only
    ever True together with a graph runner. Add a CPU/fallback test that the graph route and the eager
    route produce the same loss for 5 steps on the tiny model *if the runner has a CPU fallback*; if it
    does not, say so in the commit and queue a GPU parity run (200 steps, loss parity within 1e-3 relative).

## B. The gate: does the fix cure the ORIGINAL symptom?  (GPU; maintainer-assisted)

Do not touch defaults (C) before this has a written result.

B1. **2x2 factorial on the original failing setup.** Dataset: the original ~200-image set (the
    non-empty-caption variant if there is one; otherwise as it was). Rank 64 / alpha 48, same seed,
    same steps (>= 1000), same optimizer, same LR value.

    | arm | schedule | targets |
    |---|---|---|
    | 1 (= original defaults) | cosine | attention, cond path on (564) |
    | 2 | constant | attention, cond path on (564) |
    | 3 | cosine | kohya_plus (726) |
    | 4 | constant | kohya_plus (726) |

    Add arm 5 = constant + `kohya_default` (722, cond off) so the cond-path contribution is isolated.
    Rationale: this separates schedule from targets and tells us which one (or both) removed the collage.

B2. **Evaluate the same way for every arm** (use `archive/bad-LoRA-investigation/a5_per_t_holdout.py`
    for the numbers; keep every image):
    * neutral per-t eps-MSE on held-out images (not in the training set), buckets
      `[0,100) [100,300) [300,500) [500,700) [700,900) [900,1000)`, base vs base+LoRA, **absolute**
      differences, not relative percentages of a tiny base loss;
    * fixed-seed renders in ComfyUI at LoRA strength 0 / 0.5 / 1.0 for (a) a training caption,
      (b) a **novel prompt**, (c) a **different resolution** than training;
    * Laplacian variance on background crops for sharpness (one number per image);
    * `lora_forensics.py` table per arm (group medians and `attn_output / attn_input` ratio).
    Produce one contact sheet per arm. The maintainer judges the images; you provide the grid and a
    results table, and you do **not** declare "fixed" yourself.

B3. **Cond-path risk check (needed before kohya_plus can become a default).** With arm 4: render
    with a prompt and size very different from the training set, and with strength 0.5. Compare against
    arm 5. If arm 4 shows global colour/composition shifts that arm 5 does not, the cond-path modules
    are baking in a dataset constant (caption/size) and `kohya_plus` must stay opt-in.
    Also load the arm-4 file with diffusers `load_lora_weights` (CPU, tiny model is fine for the key
    test) and record whether the 4 cond-path modules load, are ignored, or error. Document the result
    in the preset docstring: other tools may not honour these keys.

B4. Write `archive/bad-LoRA-investigation/MEASURED-original-symptom.md`: per arm, numbers, verdict
    (*cured / partly / not cured*), and what remains. If **no arm** removes the collage, say so
    plainly and reopen the investigation in `docs/known-issues/open.md`. The remaining suspects would
    then be real-weight bf16 behaviour and the optimizer path; do not close the issue.

## C. Defaults and reachability  **[MAINTAINER DECISION after B]**

C1. **Reachability (do now, changes no default).** The recommended settings are currently only
    reachable through graph-node ports:
    * `nodes/config_model.py` has `lr_schedule: Literal["cosine", "poly"]` - no `constant`. Add it.
    * `target_modules` / `target_conditioning_path` appear in none of `config_model.py`,
      `config_io.py`, `backend/infrastructure/config_ui_data.py`, `config.example.toml`,
      `convert-cfg.toml`. Add them (enum of preset names, plus the bool), with UI labels and help text
      that state the module counts (564 / 722 / 726) and the cond-path caveat from B3.
    * Add a smoke test: config -> graph -> injector reports the expected module count for each preset.
    * Find out whether `lr_schedule` from config actually reaches the schedule node (grep showed no
      consumer besides the model definition). If it is dead, say so in `open.md` and wire it.

C2. **Effective-LR visibility (no default change).** At trainer build time print one line:
    schedule, peak LR, **mean LR over the run** (cosine/poly are not "the LR you typed"), warmup.
    Warn when mean/peak < 0.7. This alone would have shown the 0.52x months ago.

C3. **Prepare, then stop and ask.** Write a one-page decision note in
    `docs/decisions/` with the options and the B results:
    (a) keep defaults (cosine, attention) and document; (b) default new graphs/configs to
    `constant` + `kohya_default`; (c) default to `constant` + `kohya_plus`.
    Rules for whichever is chosen: existing saved graphs/configs keep their stored values (store the
    values explicitly so a default change cannot alter an old run); add a one-line changelog entry.
    My recommendation, subject to B1/B3: (b) or (c) for *new* configs, `kohya_plus` only if B3 is clean.

## D. Remaining open items from the verdict

D1. **Captions longer than 77 tokens (A4c, never tested).** Build a test with a 150-token caption and
    compare the project's `ctx` `[1,154,2048]` and pooled vector with a reference that implements
    kohya's chunk rule (75 tokens per chunk, BOS/EOS re-added per chunk, pooled from the first chunk)
    and with the chunk rule the project implements. State which of the three conventions the project
    follows and whether it matches what ComfyUI does at inference. Tiny CLIP weights, CPU, are enough
    to settle the *logic*; queue one real-weights check.

D2. **`block_weights` vs exported alpha.** `LoRALinear.scaling = alpha/rank * block_weight`, but the
    saved `.alpha` omits `block_weight`, so training and inference strength disagree whenever
    `block_weights` is used. Fix in the exporter (write effective alpha) and add a round-trip test:
    train-time forward == merged forward with `block_weights` set.

D3. **LoRA health summary at save time (the permanent version of `lora_forensics.py`).**
    On save, compute per-group `||dW||_F / ||W||_F` medians (attn input/middle/output, FF/proj, cond
    path), print them, and store them in the safetensors metadata. Warn when
    `attn_output / attn_input > 1.8` (the signature of the attention-only run was 2.0x) or when any group
    exceeds 5x the attention median. Port the Frobenius shortcut from
    `archive/bad-LoRA-investigation/lora_forensics.py` (already validated against brute force to 1e-16).
    Unit test on synthetic data. No training-loop changes.

D4. **A7, document only:** optimizer state is fp32, there is no `full_bf16` equivalent, and rank scales
    memory on this card (kohya rank 32 died with `DEVICE_LOST`). Record in `deferred.md` or `open.md`
    (one place) with the evidence from `MEASURED-kohya-on-xpu.md`.

## E. Document L4 and L5.3 properly

E1. Add entries to `docs/known-issues/pending-testing.md` for: L4 (`group_by_size_only`, per-sample
    prompts: what is proven on CPU/GPU, what is not: quality on a real many-caption dataset) and
    L5.3 (`use_xpu_graph` on both routes: +37% steps/s for +1.5 GB measured on the managed route;
    main route unmeasured; refusals; default off). Each entry: what is unproven, how to confirm.
E2. Update `shapes diversity problem 2/STATUS.md` and the backlog (`docs/design/09-prioritized-backlog.md`)
    to the real state. Do not rename `shapes diversity problem 2/`: source comments cite it
    (`nodes/dataset/managed.py:138`, `nodes/train/xpu_graph_step.py:4`).
E3. **Reference check script** (`scripts/check_doc_refs.py`): scan `*.md`, `*.py`, `*.toml`, `*.sh` for
    repo-relative paths in backticks that do not exist, print them, exit non-zero. Fix what it finds
    (the two archive moves on 2026-10-09 are the likely source). Wire it into `run_tests.py`.

## F. One user-facing page

`docs/lora-training-guide.md`, short, honest about evidence:
* recommended starting settings (constant LR, the preset chosen in C3, rank, alpha) with the
  one-image evidence **labelled as such** plus the B results;
* what "effective LR" means for cosine/poly;
* how to read the health summary (D3) and `lora_forensics.py` output;
* compatibility note for cond-path modules (B3);
* link to the MEASURED files. Do not paste benchmark tables; link them.

## G. Handoff for the node-graph rework (READ-ONLY. Do not fix anything here.)

The maintainer reports the graph cannot be used: (1) the memory budget does not take effect,
(2) graph initialisation wants more memory than is available so it will not start, (3) there is no
graph-settings section. Produce `docs/design/graph-rework-inputs.md` containing, for each of the three:
exact reproduction steps, measured numbers (peak host RAM and VRAM during init; compare before and
after `11adb0f`'s meta-device change, which may already reduce init memory), the files and functions
involved, which design doc (`docs/design/resources-controller/`, `archive/memory rework ...`) claims the
behaviour should exist and where reality differs. Add an inventory of UI/graph usability problems you can
observe, each marked *verified* or *reported by maintainer, unverified*. No code changes, no new nodes.

## Report format (each item)

Command run, config, seed, evidence class, numbers, verdict (*done / partly / blocked / ruled out*),
what is still untested, and what the maintainer must run (with exact commands). "Everything was checked,
no problems found" is not an acceptable report.

## Deliverables checklist

- [ ] A: gates recorded; meta-construction hardened + tests; graph-route tests
- [ ] B: 5-arm results + `MEASURED-original-symptom.md`; maintainer queue for GPU runs
- [ ] C1/C2 landed (no default changed); C3 decision note written; **stopped for approval**
- [ ] D1-D4, E1-E3, F
- [ ] G document
