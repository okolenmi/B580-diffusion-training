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
