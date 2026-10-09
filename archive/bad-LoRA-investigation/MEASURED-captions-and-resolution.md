# MEASURED: captionless training + resolution conditioning (A2/A5/A6 evidence)

Task: `archive/bad-LoRA-investigation/TASK-lora-quality-and-multishape.md`
Status: **negative results for two suspects; one risk identified, not a cause.**
Hardware: Intel Arc B580, 12,216 MB. bf16, real SDXL weights (`div_4.safetensors`).
One run at a time.

## Summary

| Suspect | Verdict |
|---|---|
| Noising / sigma / UNet input scaling | **Ruled out** (this project's own test, 2.9e-3) |
| Resolution conditioning (A6) | **Ruled out** — LoRA advantage does not decay off its trained size |
| Empty captions | **A real train/inference mismatch, but NOT the bug** — see below |

None of these is the cause. What survives is a narrower observation about
*where* in the timestep range the LoRA acts, below.

## 1. Empty captions: a real mismatch, and a legitimate design choice

Every caption in every ingested dataset is the empty string — counted in
the sqlite `trajectories` table, so this is what was **ingested**:

| dataset | samples | empty caption |
|---|---|---|
| `non-square` | 273 | **273 (100%)** |
| `test2` | 204 | **204 (100%)** |
| `1024 aes` | 201 | **201 (100%)** |

Measured on the real towers (`caption_gap.py`), the empty-caption ctx is
a distinct point: cosine 0.68–0.76 against five real prompts, where real
prompts sit 0.80–0.86 from *each other*.

**This is not presented as the bug.** Captionless training is a valid
choice for a style LoRA whose whole point is that changes are not bound
to tags — which is a stated goal here. The mismatch that *does* follow is
narrower: **training captionless while sampling with a prompt.** The
LoRA's cross-attention output modules (measured `attn_output` median
0.8848 vs `attn_input` 0.4366 — 2.03x) were fit against the
empty-caption ctx; used at inference against a different ctx, they edit
in a direction that was never trained.

That is avoidable *without* captions: sample with the same fixed neutral
text used in training, rather than a tag prompt. Whether that fixes the
visual artefact is **untested** — no image has been rendered.

## 2. A5: the bad LoRA lowers eps-MSE at every t — and that is fit, not quality

`a5_per_t_holdout.py`, 4 latents, 2 t per bucket, paired noise:

| t bucket | base | +Test_02 | better |
|---|---|---|---|
| [0,100) | 0.52540 | 0.52278 | +0.49% |
| [100,300) | 0.30790 | 0.30196 | +1.93% |
| [300,500) | 0.20590 | 0.19996 | +2.88% |
| [500,700) | 0.09391 | 0.08975 | +4.43% |
| [700,900) | 0.02803 | 0.02661 | +5.08% |
| [900,1000) | 0.00693 | 0.00583 | +15.78% |

**This does not exonerate the LoRA.** The latents come from
`non-square` — the same images it trained on, so this scores fit, not
generalization (the task says the same of its own holdout). A LoRA can
fit its training set better and still generate worse; those are not in
conflict.

## 3. A6: resolution conditioning ruled out

`a6_resolution.py`, same 4 latents resized, % better than base:

| latent (px) | mid-t [300,500) | high-t [900,1000) |
|---|---|---|
| 88x64 (704x512, trained shape) | +2.31% | +13.38% |
| 64x64 (512x512, modal trained width) | **+7.83%** | +15.99% |
| 128x128 (1024x1024) | +0.30% | +13.48% |

A resolution-conditioned LoRA would decay away from its trained size.
This one does the opposite: its advantage is **largest** at 512×512, the
modal training width, and it holds at 1024×1024. So the global-size axis
of `y` is not where it breaks.

## 4. What actually survives

The one pattern consistent across every measurement: **the LoRA's
largest effect is at high t** (+13% to +16% at [900,1000), versus +0.3%
to +8% at low t). Whatever it learned, it learned it in the
early-sampling regime — which is exactly where the maintainer reports
the damage ("usually they exist only during first steps of generation").

That is a lead, not a diagnosis. It is consistent with several
mechanisms (the captionless ctx, the loss weighting at high t, or the
`t` sampling range) and separating them needs images, not scalars.

## Reproduce

    python archive/bad-LoRA-investigation/caption_gap.py
    python archive/bad-LoRA-investigation/a5_per_t_holdout.py \
      --lora bad=/home/okolenmi/comfy/ComfyUI/models/loras/Test_02.safetensors
    python archive/bad-LoRA-investigation/a6_resolution.py \
      --lora bad=/home/okolenmi/comfy/ComfyUI/models/loras/Test_02.safetensors \
      --sizes 88x64,64x64,128x128

## Untested, and what would actually settle this

* **No image has been rendered.** Every number above is a scalar. The task's
  A1/A2 (overfit-one-image, and the same data through kohya sd-scripts) are
  the items that produce images, and kohya is not installed.
* `good` LoRA arm unscorable: it has no `time_embed`/`label_emb` keys and
  `load_lora_into_registry` correctly refuses a partial load. Needs A3's
  `target_conditioning_path` switch, which `build_lora_injected_unet` does
  not expose yet.
* 4 latents, 1 dataset, 2 t per bucket. Bucket *ordering* is robust; the
  third decimal is not.
* The maintainer's report that adding captions once did not help is
  **unexplained by anything measured here** and is not contradicted by it
  either — the caption path has only been checked on random tiny weights
  in fp32 (task §1.1), never on real weights.