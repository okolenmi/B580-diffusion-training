*[← docs/known-issues index](README.md)*

# Open

- **[2026-10] Adafactor `foreach` as the replacement for the retired
  batched-tiny-parameter path: measured, and the answer was "no, but a
  different strategy already had".** Measured 2026-10-02 on the B580, so
  this no longer belongs in "Open" -- the measurement is recorded here
  because the *conclusion* still has a consequence for a config choice,
  and because the hypothesis it tested turned out to be wrong in an
  instructive way.

  **What was being asked.** `nodes/optimizer/adafactor.py` is deleted, so
  `ComposedAdafactorOptimizerNode` is the only Adafactor node. It was
  retired rather than reimplemented because `ChunkedXPUAdafactor`'s
  tiny-parameter path concatenates every parameter under 10,000 elements
  into one shared clip/EMA state, and
  `smoke_test_adafactor_tiny_parameter_gap.py` Part C measured that this
  *contaminates* results -- a parameter's update depends on unrelated
  parameters' gradients sharing its batch. Correctness was never lost.
  What was lost was performance, and the entry's own hypothesis was that
  `strategy="foreach"` recovered it through `torch._foreach_*`.

  **The measurement.** `scripts/hw_validate.py main`, `1024 aes`,
  100 steps, batch 1, rank 64, seed 1234, same process-per-run harness
  throughout. 1.70 s/step ≈ 1.0 s/step of that is optimizer work, so
  this is not a rounding difference:

  | optimizer | strategy | n | steps/s | sec/step | peak reserved |
  |---|---|---|---|---|---|
  | adafactor | `shape_grouped_foreach` | 2 | **1.007** | 0.993 | 7984 MB |
  | adafactor | `shape_grouped` | 2 | 0.991 | 1.009 | 7984 MB |
  | adamw | (default `simple`) | 1 | 0.919 | 1.088 | 8592 MB |
  | adafactor | `simple` | 3 | 0.590 | 1.694 | 7884 MB |
  | adafactor | `foreach` | 1 | 0.574 | 1.742 | 8234 MB |
  | adafactor | `chunked` | 1 | 0.567 | 1.764 | 7886 MB |
  | came | (default `simple`) | 1 | 0.471 | 2.123 | 8248 MB |

  Losses agree across every Adafactor strategy to 1.9e-5 (first step) and
  3.5e-5 (last) -- the same numbers, as the equivalence smoke tests
  require. Not bit-identical, which fp32 on a GPU is not.

  **The hypothesis was wrong.** `foreach` is *slower* than `simple` here
  (0.97x) and reserves 350 MB more. `torch._foreach_*` is not the win on
  this hardware.

  **The win was the shape grouping, and it already existed.**
  `shape_grouped_foreach` is **1.71x** `simple` and 10% faster than AdamW
  -- while computing the same values. Isolating the two halves:
  `shape_grouped` alone is 0.991, so the grouping does essentially all
  of it and the `_foreach` suffix adds ~1.6% on top. The lesson for the
  next time something looks like it needs a new batching strategy: ask
  what shape the work is, not how many launches it takes.

  **Consequence.** No `TinyBatchedStrategy` is needed. The one it would
  have been written for -- grouping every parameter under a size
  threshold, any shape -- is what `shape_grouped` already does, and it
  is the fastest option measured. Nothing else changes; the default is
  still `simple`, because changing a node's default is a decision about
  what most callers want, not something this measurement made.

  **Caveats, because they bound the claim.** One dataset (`1024 aes`),
  batch 1, rank 64, one card, fp32 state. AdamW and CAME are n=1. The
  result is about *this* parameter population: LoRA rank 64 on SDXL,
  which is the case the retired code was written for. A different rank or
  target-module set moves the shape distribution and could move the
  ranking.

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
