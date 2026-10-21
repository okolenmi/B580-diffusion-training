# 72. Quantized frozen weights and the owner's kernels

Status: **notes, nothing built.** The kernels (`github.com/okolenmi/oko-xpu-kernels`) are early and forward-only; the trainer
waits for their backward support. This page records what was read there and where the trainer's seam is, so the port starts from it.

## What the kernels are
- `xpu_kernels.int8`: ConvRot INT8 linear. The activation is rotated (Kronecker-Hadamard, blocks of 256 along K) and quantized per row;
  the weight is stored pre-rotated as int8 plus a per-output-row fp32 scale; the GEMM is `torch._int_mm`; the epilogue dequantizes.
  `prepare_weight` is the only valid way to make a weight. Needs `K % 256 == 0`; the library is built with oneAPI 2025.3 only.
- `xpu_kernels.triton`: rms_norm+rope and rms_norm+adaLN, single pass, bit-exact against the chain they replace, each behind a
  once-per-process verification gate and an env switch. Tuned for a Cosmos-style DiT (Anima, Phase 8), forward only.
- Their measurements on this card: INT8 equals a fair bf16 linear in speed (0.94-0.99x at M~4096, 15% faster at 1024). The win is weight memory.

## What it would mean for the SDXL trainer
- Weight memory: of the UNet's 2.57 B parameters, 2.17 B sit in Linear layers with `K % 256 == 0` (the 3x3 convs, 0.33 B, and the
  320/640-input linears stay bf16). bf16 weights are 4.78 GB; those layers in int8 are 2.0 GB less (int8 is half of bf16, not a quarter).
  That memory buys bigger batches or fewer checkpointed blocks (measured placement turns it into speed: 8% for ~2 GB more at batch 3).
- Speed: parity at best on the matmul; the rotation and quantization are extra work per layer in the forward, and the backward needs the
  activation gradient `dL/dx = dL/dy @ W`, which today has no kernel. Quantizing `dL/dy` adds noise to the gradient that LoRA training sees;
  a weight-only dequant-in-GEMM backward avoids that and is the safer first target.
- Probe and sample renders can use the int8 base now-ish (forward only), but they are then a different function from the bf16 base.

## The seam
`AdaptedLayer` keeps the frozen weight behind `frozen_weight` (`families/general/adapters/layers.py`); a quantized layer subclasses it,
and the denoiser loader chooses the layer class per module (fallback to bf16 when a shape or the library does not fit). Nothing else
in the trainer (optimizer, saver, checkpointing, graph replay) sees the weight. Needed from the kernels before it can train:
a backward that takes `dL/dy` and the stored weight, capturable in a device graph (no host sync, no fresh allocations),
and a CPU reference for the checks here.

## Takeaways to apply regardless
- Gate and escape hatch per kernel (verify once per process, env switch) is the pattern for any compiled op the trainer adopts.
- `use_deterministic_algorithms(True)` turns on `fill_uninitialized_memory` (about 92 ms per forward on their model); a deterministic
  mode for the trainer should clear it. Not built; the owner's runs differ by ~1e-4 between identical seeds today.
- Anima: port their adaLN and rope kernels with autograd wrappers (forward under no_grad, recompute or a Triton backward) when the DiT lands.
