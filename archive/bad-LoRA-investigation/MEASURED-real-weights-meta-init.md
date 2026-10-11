# A2 on real weights: meta-built == eager-built, forward included

Measured 2026-10-11 on the maintainer's B580 box, CPU, fp32 (UNet) /
fp16 (CLIP), against `div_4.safetensors`. Script:
`archive/bad-LoRA-investigation/a6_real_meta_forward.py`, log:
`runs/a6_real_meta.log`.

**Evidence class: measured on REAL SDXL weights.** This closes the A2
follow-up in `docs/maintainer-queue.md`, which until now rested on tiny
CPU weights plus a state-dict comparison.

## Why a forward comparison was needed

The existing check ("bit-identical across 1680 state_dict tensors")
compares only tensors the checkpoint *covered*. A tensor the checkpoint
does **not** supply would be uninitialised memory under meta
construction — NaN, or different every run — and would never appear in
that comparison. The claims below are therefore stated as forward
equality, which is the thing a user actually gets.

## Results

| claim | result |
|---|---|
| UNet state dict, eager vs meta | **1680/1680 bit-identical**, max diff 0.0 |
| UNet coverage agrees | missing == 0, unexpected == 0 on both routes |
| UNet **forward** (1x4x64x64 latent, t=500, 77x2048 context, 2816-wide y) | **bit-identical**, max diff **0.000e+00** |
| UNet forward **after LoRA injection** (724 adapters, `kohya_label`) | **bit-identical**, 0.000e+00 |
| meta-built non-finite tensors | **0** (UNet and CLIP) |
| CLIP state dict | 714/716 bit-identical — see below |
| CLIP **encode** (context 1x77x2048, pooled 1x1280) | **bit-identical**, 0.000e+00 |

Peak RSS 28.1 GB (two SDXL UNets resident at once), so the "~20 GB"
estimate in the queue item was about right.

## The two differing CLIP tensors are the intended ones

`clip_l.logit_scale` and `clip_l.transformer.text_projection.weight` are
the two allowlisted-missing tensors: the checkpoint does not supply
them, so the meta route zero-fills them deterministically while the
eager route left random init. Verified directly — both are all-zero in
the meta build, and they are exactly the 2 tensors that differ.

That they are *provably unused* is no longer an argument in a comment:
the encode through both towers is bit-identical (0.000e+00) even though
these two tensors differ. `logit_scale` is contrastive-loss-only, and
CLIP-L's pooled projection is computed then discarded (`l_out, _ =
...`), so neither can reach the output — now demonstrated rather than
asserted.

## Verdict

Meta-device construction is **not** a silent-corruption hazard on real
SDXL weights, for the UNet or the CLIP towers, before or after LoRA
injection. `verify_loaded` did its job: zero uncovered tensors, raised
on nothing, and the surviving differences are the documented,
deliberate, provably-invisible ones.

**Still untested here:** non-SDXL checkpoints (SD1.5 has no
`label_emb`/`adm_in_channels`, so the coverage sets differ), and
bf16 device forwards — this ran on CPU in fp32/fp16 specifically so a
rounding difference could not hide a real one.