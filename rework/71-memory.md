# 71. Device memory: sources, levers, and what is still open

The monitor's data is `ResidencyManager.breakdown()` (reported as the `memory` stream, and printed by `python -m next.run`):
each resident (model, optimizer state, encoders), other live tensors, the step's activation peak, free blocks the
allocator keeps, and what the driver holds outside the allocator. `device_used` (from the driver) is the number that
predicts out-of-memory; the allocator's own peak understates it.

## Levers in place

| Lever | Cost to speed | Where |
|---|---|---|
| Text encoders on the host except while encoding (`OnDemandKit`); the device cache is emptied when they leave | one round trip per epoch | `engine/runtime/encoders.py`, Conditioner node |
| VAE released and the device cache emptied after the latent cache is built | none | Latent Cache node |
| Frozen weights in bf16, loaded straight to the device (meta construction, no fp32 copy) | none | `families/sdxl/denoiser.py` |
| Activation checkpointing: residual blocks always, transformer blocks by `attention_fraction` | recompute of the chosen blocks | `engine/runtime/checkpointing.py` |
| Fused optimizer (no gradient pile-up), grouped updates, int8 optimizer state | small | `engine/optim` |
| Allocator cache emptied every step (setting), and after any offload | small | `ResidencyManager` |
| Pressure levers between steps: cache every step, then park residents (offload after the step, reload before the next) | one transfer per parked resident per step | `ResidencyManager` |

## Open (port or build when measurement says it pays)

NF4 frozen weights (old `nf4_weight_store`, `nf4_lora_layer`); per-block checkpoint placement from measured block costs
(old `block_profiler`, `checkpoint_placement`); block swap; activation offload to the host; graph capture; oneDNN
primitive-cache sizing and thrash detection (old `xpu_env`, `cache_thrash`).
