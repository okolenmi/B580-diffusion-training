# 72. Quantized frozen weights and the owner's kernels

Status: **int8 frozen weights built, unverified on hardware** (published as `xpu-kernels` on PyPI, 0.2.0: ConvRot linear plus the
`dL/dx` backward). Fused rope/adaLN kernels wait for the Anima DiT (phase 8). This page records what the kernels give, how the trainer
uses them and what is still open.

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

## What was built (xpu-kernels 0.2.0)
- `families/general/quantize/`: `backends.py` (one `Int8Backend` interface, the `xpu-kernels` implementation and a plain-torch reference that
  is exact but slow and runs the checks here), `ops.py` (`Int8LinearFn`: forward through the backend, backward `dL/dy @ W` with the dense weight
  rebuilt for that layer only), `int8.py` (`Int8Linear` for plain linears, `Int8Quantizer`, `quantize_frozen`).
- `AdaptedLinear.use_int8`: the base weight buffer becomes int8 plus `base_scale`; `frozen_weight` rebuilds the dense weight (only merge and
  DoRA need it, both refuse). The family exposes `quantize_frozen`; the Inject Adapter node has `base_precision`, `int8_backend`, `int8_min_params`.
- Choices that follow the kernels' notes: the backward uses the weight-only path (their `linear_dx` after `dequant_weight`), not the quantized
  gradient path (`linear_dx_int8`: a memory option, slower, extra gradient noise). The dense weight is rebuilt per layer per backward instead of
  kept, since keeping it would undo the memory saving; the cost is one dequant of the quantized layers per backward (their figure: 2.6 ms per
  25 M parameters, so roughly 0.2 s for the 2.17 B here if it ran at that rate: to be measured, it is the number that decides the feature).
- Open: a captured step with the per-layer dequant inside the graph (allocations inside a capture come from the graph pool and replay at the same
  addresses, so it should work; device test 23 covers it), the first-call verification gate for the INT8 side (the package has `selftest`;
  the Triton kernels have their own gate), `dW` is not needed for LoRA.
