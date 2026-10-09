# MEASURED: A1 verdict — LR schedule + LoRA targets (the quality gap, closed)

Task: `TASK-lora-quality-and-multishape.md`, Part A (A1/A2/A3) + loss-parity
addendum steps 1–2.
Status: **Part A has a conclusion. The training loop is sound; the gap was
configuration, not a defect.** Part B gate lifted.
Hardware: Intel Arc B580, 12,216 MB. bf16 UNet, fp32 optimizer/LoRA master
weights, real SDXL weights (`div_4.safetensors`, = `divergence_3` VAE).
One run at a time; ComfyUI closed during GPU runs.

A1 config (identical for every arm below): one 512x512 image, empty
caption, rank 16 / alpha 16, 500 steps, batch 1, lr 1e-4, AdamW.
Kohya arm trained by the maintainer (rank 16/alpha 16; rank 64/32 died
with `DEVICE_LOST` — rank gates steps-before-death on this card, verified
by rank32 dying at ~40 steps and rank16 completing 500).

## 1. Schedule: cosine decay was undertraining ~2x vs kohya's constant

Nominal LR was 1e-4 in both trainers. Effective LR was not:

| | kohya | this trainer (then-default) |
|---|---|---|
| schedule | `constant` (kohya default) | cosine decay to 5% (`CosineLRSchedule`, `lr_min_frac=0.05`) |
| mean LR over run | 1e-4 | ~0.52e-4 |

Forensics (median |dW|, rank 16) predicted the render verdict exactly:

| group | kohya | ours, cosine | ours, constant |
|---|---|---|---|
| attn_input | 0.273 | 0.149 (1.8x under) | **0.272** |
| attn_middle | 0.262 | 0.177 (1.5x under) | 0.317 |
| attn_output | 0.324 | 0.350 | 0.644 |

Maintainer renders: cosine arm "doesn't work at 1.0, needs 1.5 power";
constant arm "close image style, maybe character style, ~0.9x kohya".
**The 1.5–1.8x undertraining factor measured in the weights is the same
number as the render verdict.** Cause confirmed by measurement, not argument.

Side effect this also explains: the addendum's "kohya loss ~50% higher"
is consistent with kohya applying ~2x the LR·steps, i.e. a harder,
further-along optimization — not different data, a different loss, or
different weighting. Steps 1–2 below confirm the data and loss path are
identical, so dynamics (this schedule first among them) is where the gap
lived. Do not re-investigate the 50% number.

Fix: `a2_mine.py --schedule constant`; `ConstantLRScheduleNode` already
existed. The trainer default is unchanged (cosine) — changing it is a
follow-up decision, not part of this finding.

## 2. Targets: kohya adapts 722 modules, we adapted 560 — and it matters

`target_modules` was dead code: `_inject_lora` hard-coded
q/k/v/to_out.0 while the config carried a `target_modules` field nobody
read. Kohya's default SDXL LoRA adapts every Linear in each
`Transformer2DModel` (forensics on the kohya A1 file: 240/100/360/22 =
722, incl. `ff.net.0.proj`, `ff.net.2`, `proj_in`, `proj_out`).
`lora_targets_and_cond_switch.patch` wires the field through and adds
`target_conditioning_path` (the old `conditioning_path_switch.patch` is
superseded by it — same switch, plus the targets).

Matched A1 arms, constant LR, cond path off (560 vs 722 isolates exactly
the FF/proj modules):

| group | kohya (722) | ours attn-only (560) | ours kohya_default (722) | ours kohya_plus (726) |
|---|---|---|---|---|
| attn_input | 0.273 | 0.310 | 0.240 | 0.263 |
| attn_middle | 0.262 | 0.298 | 0.225 | 0.230 |
| attn_output | 0.324 | **0.623** | **0.321** | **0.321** |
| other_unet | 0.289 | — | 0.321 | 0.317 |
| time/label_emb | — | — | — | 1.06 / 1.66 |

Two readings: (a) with attention-only targets the output projections
carry ~2x what kohya's do — capacity placement, not just magnitude;
(b) adding the FF/proj targets lands every attention median on top of
kohya's (0.321 vs 0.324 on `attn_output`).

Maintainer renders: 722 "close enough"; **726 (`kohya_plus`: 722 + the
4 conditioning-path embedding modules) "works much better"**. The cond
path at matched targets is a carrier, not damage — on one image with an
empty caption it is the only global carrier (label_emb median 1.66), and
excluding it starves the run.

Presets shipped (`LoRAConfig.target_modules`): `attention` (560 + cond
= 564, historical default, unchanged), `kohya_default` (722 + cond),
`kohya_plus` (same UNet set, cond kept by default = 726).
Recommendation for style LoRAs: `--schedule constant --target-modules
kohya_plus`. Whether `kohya_plus` becomes the trainer default is
**undecided** — it changes every existing run's behaviour and needs its
own item, not a drive-by.

## 3. Addendum steps 1–2: data and loss path exonerated on real weights

- Step 1 (`latent_stats.py`): project shards vs diffusers reference —
  std/roughness/hf_energy all x1.000. Ingestion exonerated.
- Item 2 (mode vs sample): VAE posterior std is 0 (max 0.001 on real
  weights) — mode == sample, dead.
- Step 2 (`step2_loss_parity.py`): 16 fixed (x0, eps, t) triples,
  empty-caption ctx, LoRA off — project XPU/bf16 vs diffusers CPU/fp32
  agree to **0.0–0.1% in every t bucket**. Loss path exonerated.
- Weighting: both trainers uniform. t distribution, resolution, empty
  captions, conditioning path (as damage theory): all previously ruled
  out, unchanged.

## 4. Per-item verdicts

| Item | Verdict |
|---|---|
| A1 | **Concluded**: loop sound with matched settings; gap was schedule + targets |
| A2 | **Concluded**: kohya XPU toolchain works (see `MEASURED-kohya-on-xpu.md`); matched 201-step + A1 comparisons done |
| A3 | **Concluded**: switch built and tested; plus-arm keeps embeddings and wins |
| A4 (a,b,d,e,f) | Superseded by steps 1–2 (real weights, bf16 vs fp32); (c) >77-token captions still untested |
| A5/A6 | Ruled out earlier (see `MEASURED-captions-and-resolution.md`); the "largest at high t" relative-% claim is withdrawn — absolute improvement is largest at mid-t |
| A7 (bf16 optimizer line) | Still open: no `full_bf16` equivalent, optimizer state is fp32. Not needed for A1 but binds at high rank on this card |
| A9 | Partially answered by the schedule finding; per-step logging never built |

## Files (in ComfyUI `models/loras/`)

`A1_kohya.safetensors` (kohya ref) · `A1_attn560.safetensors` (ours,
attention, no cond) · `A1_kohya722.safetensors` (ours, matched 722) ·
`A1_kohyaPlus726.safetensors` (ours, 726 — best) ·
`A1_mine_const.safetensors` (ours, 564 constant) ·
`A1_mine.safetensors` (ours, 564 cosine — the undertrained control).

## Reproduce

    python bad_LoRA_investigation/a2_mine.py --dataset a1_one --steps 500 \
      --batch 1 --rank 16 --alpha 16 --schedule constant \
      --target-modules kohya_plus --out <name>
    python bad_LoRA_investigation/step2_loss_parity.py
    python bad_LoRA_investigation/latent_stats.py --shards datasets/a1_one/shards \
      --images runs/a1_one_image/images \
      --ckpt /home/okolenmi/comfy/ComfyUI/models/checkpoints/div_4.safetensors --px 512
