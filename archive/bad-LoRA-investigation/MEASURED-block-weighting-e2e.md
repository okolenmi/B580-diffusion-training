# MEASURED: block weighting end-to-end (D2 real-weights check)

Measured 2026-10-11 on the B580. Two 200-step LoRAs, identical in every
respect except `block_weighting`.

**Evidence class: measured on real SDXL weights (GPU training) + read in
code (the render leg is the maintainer's).**

## Setup

| | arm A | arm B |
|---|---|---|
| dataset | `a1_one` (1 image, 64x64, caption "low quality") | same |
| steps / batch | 200 / 1 | same |
| rank / alpha | 16 / 16 | same |
| schedule | constant 1e-4 | same |
| targets | `kohya_label`, cond off | same |
| **block_weighting** | **off** | **`input_blocks.7:0.25`** |

Both run through the study harness (`a2_mine.py --block-weighting`), so
the value goes through the real `ComfyUNetLoRANode` Port.

## Result: the two files DIFFER, and they differ in exactly the right way

- 2172 tensors each, identical key sets.
- **1550 of 2172 tensors differ.** Only 622 are bit-identical -- so the
  arms are clearly distinguishable, which is the whole point: a run where
  the two files had come out identical would mean the weight never
  reached the model at all.
- The 102 modules of the weighted block carry **alpha 4.0**; the other
  622 carry **alpha 16.0**. Verified against the metadata: every one of
  the 102 satisfies `alpha == 16.0 * weight`, zero mismatches.
- Arm A's metadata records an empty `block_weights` map and a single
  alpha value (16.0 everywhere) -- a clean control.

The differing `lora_down`/`lora_up` tensors are the expected second-order
effect: a 0.25-scaled module produces smaller gradients, so its learned
factors differ from the control's even at the same step count.

## What this proves

1. `block_weighting` reaches the model through the node Port (it was a
   dead key until this session -- see the reachability commit).
2. The weight is folded into the exported alpha, so a reader that only
   knows `alpha / rank` (ComfyUI, diffusers, peft) applies the trained
   strength without knowing the metadata exists.
3. Before the fix this was measurably wrong in the opposite direction: a
   0.5 weight came back at 2x the trained strength, a 2.0 weight
   reloaded at 6.0 instead of 3.0.

## What is NOT proven

The render leg. The numerical claim -- that an outside reader's
`(B@A) * (alpha/rank)` reproduces the trained delta exactly -- is proven
on CPU for weights 0.5/1.0/2.0 across LoRA and DoRA
(`smoke_test_lora_file_format.py`), and the files here are consistent
with it. What has not been done is opening both files in ComfyUI and
confirming the renders differ. Only the maintainer can judge that, so it
stays as queue item 4.

Also unproven: diffusers loading a real weighted file (the converter claim
rests on executing its own code against a toy state dict), and non-SDXL
checkpoints.