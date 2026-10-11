# MEASURED: the original symptom (B1/B2/B3) — 200-image rank-64 study

Task item B (`TASK-finalize-lora-defaults-and-docs.md`). Status: **training
done, numbers done, maintainer render ranking recorded**; contact sheets /
blur table still open.
Hardware: B580, 12,216 MB. One run at a time (violated twice by
diagnostic probes -- noted where it matters).

## B1 setup (all five arms)

Dataset `test2` (204 images, 512x512, 1 caption -- the empty one, as it
was), rank 64 / alpha 48, batch 4, 1000 steps, seed unset (loader
shuffles unseeded -- batch order differs per arm; noted, not hidden),
AdamW 1e-4 fp32. Batch 4, not 8: batch 8 wedges late at sustained
~11 GB (open.md entry); B1 fixes rank/alpha/seed/steps/optimizer/LR,
not batch. Arms 3-5 additionally unload the text encoder (towers to CPU
after warmup / before step 0 -- cache hits only, training math
identical) to sit under the wedge zone; arms 1-2 kept resident.
`a2_mine.py --progress` on all arms (unbuffered step lines).

| file | schedule | targets | modules | trainable |
|---|---|---|---|---|
| `B_arm1_cosine_attn` | cosine | attention + cond | 564 | 93.6M |
| `B_arm2_const_attn` | constant | attention + cond | 564 | 93.6M |
| `B_arm3_cosine_plus` | cosine | kohya_plus | 726 | 170.9M |
| `B_arm4_const_plus` | constant | kohya_plus | 726 | 170.9M |
| `B_arm5_const_722` | constant | kohya_default, cond off | 722 | ~170M |

## Forensics (median |dW|, scale 0.75)

| arm | attn_in | attn_mid | attn_out | out/in | other | cond (t/l) |
|---|---|---|---|---|---|---|
| 1 cosine/attn | 0.201 | 0.221 | 0.262 | **1.30** | — | 0.94/0.95 |
| 2 const/attn | 0.358 | 0.381 | 0.484 | **1.35** | — | 1.41/1.32 |
| 3 cosine/plus | 0.163 | 0.165 | 0.250 | 1.54 | 0.20 | 0.64/1.19 ⚠ label >5x |
| 4 const/plus | 0.359 | 0.348 | 0.424 | **1.18** | 0.39 | 0.97/1.04 |
| 5 const/722 | 0.396 | 0.389 | 0.429 | **1.08** | 0.45 | — |

Two readings: (a) constant roughly doubles trained magnitude vs cosine
at the same nominal LR (the 0.52x schedule effect, reproduced at scale);
(b) the output/input balance predicts the maintainer's render ranking
exactly: 1.08 < 1.18 < 1.35.

## Maintainer render ranking (ComfyUI, recorded 2026-10-11)

1. **arm 5 (kohya_default, 722)** -- best on this dataset.
2. **arm 4 (kohya_plus, 726)** -- second.
3. **arm 2 (constant attention)** -- third.
(Cosine arms unranked against these; constant-attn placing third implies
constant beats cosine as in A1.)

Maintainer hypothesis: `time_embed` is riskier to adapt than `label_emb`
("timing is serious, labels less so -- may be wrong"). Evidence on the
table: label_emb medians exceed time_embed in 3 of 4 cond-carrying arms
(arm2 is the exception, 1.41 > 1.32), and arm 3's 5x warning fires on
label_emb. The data leans label, not time -- flagged, not settled; the
B3 far-prompt/size comparison (checklist) would separate them.

## B2 per-t holdout (unseen images)

Scored on `non-square` latents (different images, never trained on),
base vs each arm, same noise/t/ctx per sample. Per-arm injection matched
to the file's key set (attention / plus+cond / 722-no-cond); base
reproducibility across the three runs is the control.

Base MSE (identical to 5 decimals across all three runs -- control
holds): 0.52540 / 0.30790 / 0.20590 / 0.09391 / 0.02803 / 0.00693.

% better than base per bucket ([0,100) [100,300) [300,500) [500,700)
[700,900) [900,1000)):

| arm | low | | | mid-high | | high-t |
|---|---|---|---|---|---|---|
| 1 cosine/attn | +5.13 | +4.67 | +4.68 | +5.74 | +6.48 | +18.07 |
| 2 const/attn | +5.47 | +4.79 | +4.82 | +5.80 | +7.00 | +19.25 |
| 3 cosine/plus | +5.57 | +4.89 | +4.87 | +6.12 | +6.70 | +18.34 |
| 4 const/plus | +5.79 | +5.40 | +5.40 | +6.61 | +6.81 | +19.53 |
| 5 const/722 | +4.77 | +5.25 | +5.36 | +6.17 | +6.80 | +19.31 |

Absolute terms (high-t base is 0.00693, so ~19% is ~0.0013; mid-t base
0.2059, ~5% is ~0.01).

Reading: **no arm raises loss in any bucket** -- there is no damaged t
region in any of the five, which answers B2's localization question
negatively. But the spread between best and worst arm is only ~1.5pp
while the renders rank them clearly (722 > 726 > attn): fit-metric
cannot see the render difference, same lesson as the old A5 (fit is not
quality). The per-t numbers rule damage *out*; they do not rank.

## B3 cond-path risk

- diffusers 0.41 + peft loads the 726 file **completely**: attn, ff/proj,
  `time_embedding` linears AND `label_emb` keys (as `add_embedding`
  linears) all attach -- nothing ignored, no error. Other tools honour
  these keys; the risk is semantic (baked constants), not compatibility.
- Far-prompt/size + strength-0.5 arm4-vs-arm5 comparison: open (checklist
  with maintainer).

## Memory appendix (the batch-8 detour, briefly)

- No leak anywhere: 5 s VRAM sampling across arm-3/arm-4 shows flat
  sustained lines (11.3 / 11.1 GB) -- startup rises to plateau in ~15 s
  and stays; no growth over 1000 steps.
- The wedge correlates with sustained level (≳11 GB), not batch, not
  steps: 560-module arms idle 10.6 and complete; 726-module arms idle
  11.1-11.3 and wedge or OOM on transients (first-update 150 MB temp
  with 236 KB free, twice). Hangs instead of raising OOM per maintainer:
  B580/driver combination.
- nvtop-vs-mem_get_info: both sample the same card; brief transients
  (compile scratch, foreach temps) are visible to one sampler and not
  the other depending on phase -- sampling aliasing, neither lying. The
  5 s curve misses sub-second excursions that instant reads catch.
- Batch 8 (10.8 sustained) leaves no room for those transients; batch 4
  (10.6) does, barely. Production guidance stands: stay meaningfully
  under 11 GB sustained on this card.

## Verdict

*Partly cured, pending the maintainer's final word.* Schedule + targets
move every number in the right direction and the forensics balance
(out/in ratio) tracks render quality monotonically -- but no arm has been
compared against the ORIGINAL Test_02 collage symptom on identical
prompts, and the contact sheets + blur table are still open. If the
sheets show the collage gone on arm 5, close it; if any arm still
collages, reopen in `open.md` -- remaining suspects per the task are
real-weight bf16 behaviour and the optimizer path.
