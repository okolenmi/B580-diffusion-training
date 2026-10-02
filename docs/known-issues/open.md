*[← docs/known-issues index](README.md)*

# Open

- **[2026-07] "Device lost" errors and silent training hangs after
  VRAM-pressure events, reported from real ComfyUI use (the `core/`
  pipeline, not `nodes/`).** User-reported, not yet investigated here.
  Symptom: not a normal OOM -- either a device-lost error or a silent
  hang, most reliably reproduced by a VRAM-heavy sequence (merging three
  6GB models, generating with an intermediate merge state, then
  generating again with a different base model) and, separately, seen in
  this project itself after a VRAM spike during preview generation's VAE
  decode step -- loss of a few hundred MB stabilizes (frees back down)
  but training hangs a few steps later *despite* free VRAM being
  available afterward. User's own read, worth taking seriously: something
  gets offloaded under memory pressure but isn't correctly loaded back,
  even though there's room for it. A `kohya-ss/musubi-tuner` discussion
  training Wan2.2 on the same B580 hardware describes a matching
  hang-after-offload symptom, traced there to a `synchronize_device()`
  call missing its `device` argument on the non-CUDA path. **That
  root-cause shape (an async/non-blocking transfer without a matching
  explicit synchronize on the XPU path) has been checked here against
  `core/trainer.py` and does not apply** -- the offload and reload
  transitions all call `xpu_synchronize()` explicitly, and the code's own
  comment records that the VRAM spike was seen at exactly that
  reload-after-preview transition, which the sync there addresses. Don't
  re-investigate that hypothesis; the TOML route has otherwise still
  never been run under pressure since this report.
  **2026-09-28 update (hardware now available):** the *rewrite's own*
  offload path -- a different codebase from the `core/trainer.py`
  implicated here -- was exercised under real, sustained VRAM pressure
  (30 steps at `vram_budget_mb=2500` against ~8114 MB actual usage,
  offload taken every step) with no hang and no device-lost
  (`scripts/hw_validate.py`, label `C_pressure`; see the confirmed
  entry in [`resolved.md`](resolved.md)). That says the `nodes/`
  `synchronize()` hardening behaves under pressure; it says nothing
  about the TOML route this entry is about. What *has* been run: a
  plain health check of the TOML CLI route on this hardware
  (2026-09-28 -- 100 steps on `datasets/test` via `core.cli` with
  `runs/hw_validation/legacy_check.toml`: 100/100 steps, ~794 ms/step,
  clean LoRA save, no hang or device-lost) -- a healthy baseline, but
  not the pressure-plus-preview-decode trigger this entry describes.
  Note the CLI run couldn't exercise previews at all: `core/trainer.py`
  skips preview generation without a server `run_id`, so previews only
  fire on server-launched runs -- which means the CLI route can never
  have triggered this report, and the repro has to go through the
  server. The TOML-route repro (training under pressure through preview
  generation's VAE decode, the reported trigger) remains the next
  concrete step here.

  **2026-10-02: the repro path no longer exists.** The reported trigger
  needs training-in-process to preview-sample, and preview sampling only
  existed on the `core/` route -- `nodes/` has none, so a graph
  execution cannot reach it. `core/` is now `archive/core/` and the
  backend no longer spawns it. `manager/preview.py` still holds a VAE
  decoder, but nothing in `backend/` imports it: it is dataset-task
  territory, and that runs in its own child process.

  So this entry can no longer be reproduced, and not because anyone
  reproduced it and it went away. The remaining useful work is the
  opposite of a repro: decide whether the *shape* of the report is worth
  guarding against in the route that now exists, given that the one
  route it was reported against had explicit `xpu_synchronize()` at
  every offload/reload transition (checked, does not match) and the
  `nodes/` route has been run under sustained pressure with no hang
  (see the `C_pressure` entry in `resolved.md`). Left open because
  "the path is gone" is not the same as "the cause is understood".

- **[2026-10] Retiring `AdafactorOptimizerNode` traded an unmeasured
  batched-tiny-parameter optimization for canonical per-parameter math;
  nobody has measured what that cost.** `nodes/optimizer/adafactor.py` is
  deleted, so `ComposedAdafactorOptimizerNode` is the only Adafactor node
  (2026-10-02). It was retired rather than its behavior reimplemented, on
  purpose: `ChunkedXPUAdafactor`'s tiny-parameter path concatenates every
  parameter under 10,000 elements into one shared clip/EMA state, and
  Part C of `nodes/smoke_tests/smoke_test_adafactor_tiny_parameter_gap.py`
  measured that this *contaminates* results -- a parameter's update
  depends on unrelated parameters' gradients sharing its batch. Adopting
  it would mean putting that coupling back deliberately.

  What is genuinely lost is performance, not correctness: that
  concatenation turned hundreds of small parameters' clip/EMA/normalize
  into one kernel launch each. `strategy="foreach"` recovers much of the
  launch-overhead win through `torch._foreach_*` without contaminating
  anything, and is what to reach for on a graph with many small LoRA
  matrices. **Unmeasured: whether that is in fact fast enough on the
  B580.** Nothing here has run on hardware -- every equivalence result in
  that smoke test is CPU torch.

  To settle it: train the same config and seed twice, once with
  `strategy="foreach"` and once with `strategy="simple"`, on a graph with
  the realistic number of small LoRA matrices, and compare wall-clock
  per step. Then, if foreach is short of the old behavior, the remaining
  work is a `TinyBatchedStrategy` grouping "every parameter under a size
  threshold, any shape" -- but it should be written only with that
  measurement in hand, since writing it blind is exactly what this
  project's own rule against unevidenced techniques forbids.

## Measured and closed

Recorded here rather than in [`resolved.md`](resolved.md) because there
is no bug: this is a measured question that was asked, answered on
hardware, and closed. Nothing is outstanding.

- **[2026-09-28] At 1024x1024/batch 2 (the compute-ceiling operating
  point), attention-checkpointing density is binary and every measured
  floor lever costs more speed than it frees VRAM.** Motivation: the
  remaining ~25% recompute penalty of gradient checkpointing looked
  attackable two ways -- checkpoint fewer blocks, or shrink the floor
  so fewer blocks need checkpointing. Both were measured on real
  hardware (`scripts/hw_validate.py`, batch 2, dataset 1024, 40 steps,
  new `--attn-ckpt-fraction` knob plus new per-stage floor capture in
  `summary.json`'s `floor_stages` and the first step's
  `component_footprints_mb`). Composition first: both routes
  (main and managed) have the identical floor -- 7,529 MB allocated at
  loop entry, 7,888 MB at step 0, split as UNet+LoRA 4,897 MB + text
  encoder 1,561 MB + optimizer states 714 MB (lazy, appear on step 1)
  + ~716 MB grads/conds/misc; full density (1.0) peaks at 9,268 MB
  reserved (0.768 steps/sec main, 0.716 managed; bs1 reference:
  8,592 MB peak, ~0.946 steps/sec). Density sweep: **1.0 fits, 0.75
  and 0.5 both OOM on step 0** at ~10.7 GiB allocated mid-forward
  (14-22 MB free), and 0.75 still OOMs on the managed route with the
  floor released to 6,327 MB -- because `AdaptiveResidencyController`
  calibrates on 3 fully-resident steps, and step 0 dies before any
  release can happen. Skipping even a quarter of the blocks costs more
  than the ~2 GB of headroom that exists; there is no useful middle
  ground at this operating point. Floor levers, all measured
  against the managed 0.716 steps/sec / 9,268 MB baseline: `nf4`
  weight store gives back 1,361 MB of floor (566 MB of peak) but runs
  0.574 steps/sec (-20%, dequant on every forward); `int8_blockwise`
  optimizer states give back 529 MB (peak actually *rises* to 9,440 MB)
  at 0.491 steps/sec (-31%, per-step cast cost); a budget-forced
  release of optimizer+text encoder (`--budget 8000
  --cache-text-encoder`) gets allocated down to 5,613-6,327 MB but
  reserved only to 8,960 MB (freed memory lingers in the allocator
  pool) at 0.331 steps/sec (-54%, dominated by re-uploading the 714 MB
  of optimizer states that are needed every step). The one
  *theoretically* cheap lever -- releasing the text encoder alone
  (1,561 MB, nearly free once `cache_text_encoder` has the conds) --
  is unreachable as designed: the controller releases candidates
  smallest-footprint-first, so the always-needed optimizer always
  comes out first and drags its per-step transfer cost along with any
  encoder release. Making that lever real would mean ordering release
  candidates by *use cost* (encoder with cached conds ~0, optimizer
  ~every step) instead of footprint -- untested whether ~1.5-1.9 GB of
  free floor would then also make density 0.75 fit (0.75 died
  mid-forward at 10.9 GiB against a 7.9 GB floor, so the remaining
  forward plus backward is plausibly but not certainly under the wall
  at a 6.3 GB floor). Not a bug -- the allocator and the accounting
  both check out (this continues the 2026-09-28 VRAM-correction work
  above: no phantom memory, floor and peak both real). What it means
  for the pending "reduce checkpointing's recompute penalty" work:
  at 1024/bs2 the penalty can only be attacked by making recompute
  itself cheaper or by landing the encoder-only release; density
  tuning and the quantized/offload levers are closed by these
  measurements.
  **2026-09-28 update (same day, later same session): the encoder-only
  release landed as `ManagedLoRATrainerNode`'s `prewarm_text_encoder`
  Port -- the order-by-use-cost change to the controller described
  above was not needed, because prewarming sidesteps the ordering
  problem entirely (the encoder is unloaded once at build, before
  calibration, and never re-uploaded: every later step's encode is a
  cache hit, misses self-load through the cache's bound handle, and a
  0-footprint candidate is one AdaptiveResidencyController stops
  considering). Measured, managed route, batch 2, dataset 1024, 40
  steps: floor 7,888 -> 6,327 MB allocated at step 0, peak reserved
  9,268 -> 7,666 MB, throughput 0.716 -> 0.789 steps/sec (+10% --
  the per-step CLIP forward is gone too). This also confirms the
  causal story above: with the floor cut applied *before* calibration
  (which the budget path couldn't do -- it only releases after 3
  fully-resident calibration steps), density 0.75 now completes,
  peak 10,922 MB (~300 MB under the wall -- feasible, not
  comfortable). But it is not a speed win: 0.759 steps/sec vs 0.789
  at density 1.0, i.e. skipping 25% of attention recompute came out
  slightly *slower* while adding 3.3 GB of activation residency --
  recomputing these blocks is cheaper than carrying their
  activations. Practical verdict unchanged and now stronger: keep
  density 1.0 (full checkpointing), add prewarm; density tuning stays
  closed, and the quantized/offload levers stay closed (nothing here
  rescues them -- they were never blocked on floor). The remaining
  open question is only whether the ~25% recompute penalty figure
  measured elsewhere still applies to this operating point, where
  checkpointing's activation residency costs less than its saved
  compute.

## The backend test suite leaked its scratch directories (found 2026-10-02)

Every file under `backend/tests/` creates scratch with
`tempfile.mkdtemp`, which returns a name and hands back no handle, so
nothing removed them. One full suite run left ~40 directories behind, and
over many runs that reached **4,834 directories and 2.1 GB on the machine
that noticed**.

`/tmp` here is a 20 GB tmpfs, so it was not disk that filled but RAM --
which makes the failure mode worse than slow: at the wrong moment a test
fails because it could not create its scratch directory, and the failure
is attributed to a test that has nothing wrong with it. One such failure
was seen (`test_graph_execution.py`, once, not reproducible) and the real
cause is very unlikely to be a race in that file at all.

Fixed in `backend/tests/run_all.py`: each file gets its own `TMPDIR`,
which `tempfile` honours, and it is removed afterwards. A full suite run
now leaves nothing in `/tmp` at all -- verified. `run_all.py` is the gate's
entry point, so this covers every gate run; running a single test file
directly still leaves what that file makes, which is the honest boundary.

*Not fixed:* the ~85 bare `tmpXXXXXXXX` directories in `/tmp` are not
ours (they carry no project prefix and appeared alongside other tools'
output), and `/tmp/opencode` is deliberately left alone.
