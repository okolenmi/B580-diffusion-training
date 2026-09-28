*[← docs/known-issues index](README.md)*

# Open

- **[2026-07] "Device lost" errors and silent training hangs after
  VRAM-pressure events, reported from real ComfyUI use (legacy `core/`
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
  even though there's room for it. Likely related to the "Persistent
  ~500MB VRAM growth after preview generation" entry below (same VAE
  decode trigger point) but the *symptom* here (hang/device-lost, not
  just VRAM not dropping back down) is a distinct, arguably more serious
  report -- not confirmed to be the same root cause, not assumed to be
  either. A `kohya-ss/musubi-tuner` discussion training Wan2.2 on the
  same B580 hardware describes a matching hang-after-offload symptom,
  traced there to a `synchronize_device()` call missing its `device`
  argument on the non-CUDA path -- a plausible root-cause *shape* (async/
  non-blocking transfer without a matching explicit synchronize on the
  XPU path) worth checking `core/trainer.py`'s own offload code for, not
  a confirmed diagnosis here. Not investigated further this session --
  out of scope for `nodes/`-only work, and needs `core/trainer.py`,
  which `nodes/` doesn't touch.
  **2026-09-28 update (hardware now available):** the *rewrite's own*
  offload path -- a different codebase from the legacy `core/trainer.py`
  implicated here -- was exercised under real, sustained VRAM pressure
  (30 steps at `vram_budget_mb=2500` against ~8114 MB actual usage,
  offload taken every step) with no hang and no device-lost
  (`scripts/hw_validate.py`, label `C_pressure`; see the confirmed
  entry in [`resolved.md`](resolved.md)). That says the `nodes/`
  `synchronize()` hardening behaves under pressure; it says nothing
  about the legacy path this entry is about, which still has never been
  run under pressure since this report. What *has* been run now: a
  plain health check of the legacy CLI route on this hardware
  (2026-09-28 -- 100 steps on `datasets/test` via `convert.py` with
  `runs/hw_validation/legacy_check.toml`: 100/100 steps, ~794 ms/step,
  clean LoRA save, no hang or device-lost) -- a healthy baseline, but
  not the pressure-plus-preview-decode trigger this entry describes.
  **VRAM numbers from that run, corrected (the earlier version of this
  note said "reserved flat at ~6034 MB", which understated real
  consumption):** the card actually sat at **~11.4 of 12.2 GB used**
  (user's own monitoring and a cross-process `torch.xpu.mem_get_info`
  query agree), i.e. only ~450-800 MB headroom -- directly relevant to
  a hang-under-pressure report. The ~6034 MB figure was real but a
  different quantity: every training-loop `[vram]` snapshot was taken
  *after* `xpu_empty_cache()` in the maintenance block, which collapses
  reserved to roughly allocated (steady reserved during training is
  ~9942 MB; the allocator keeps a ~4.3 GB free-block pool above live
  5624 MB tensors, plus ~900 MB desktop baseline and context overhead,
  which reconciles exactly with the 11.4 GB driver total). The snapshot
  order was fixed the same day (see the confirmed entry in
  [`resolved.md`](resolved.md)). Note the
  CLI run couldn't exercise previews at all: `core/trainer.py` skips
  preview generation without a server `run_id`, so previews only fire
  on server-launched runs. The legacy-path repro (training
  under pressure through preview generation's VAE decode, the reported
  trigger) remains the next concrete step here.

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
