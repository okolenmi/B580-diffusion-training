# 72. Quantized frozen weights and the owner's kernels

Status: **int8 frozen base weights built, hardware numbers in; NF4 not built.** Kernels: `xpu-kernels` on PyPI (0.3 or newer; source
`github.com/okolenmi/oko-xpu-kernels`): ConvRot INT8 linear, `dL/dx` backward, fused weight dequantizer, fused NF4 dequantizer, fused
rms+rope and rms+adaLN (Anima, phase 8). Measured on the B580 with the trainer's own UNet by the kernels' `debug/probe_sdxl_trainer_step.py`.

## What the kernels are
- `xpu_kernels.int8`: the activation is rotated (Hadamard blocks of 256 along K) and quantized per row; the weight is stored pre-rotated as
  int8 plus a per-output-row fp32 scale; the GEMM is `torch._int_mm`. `K % 256 == 0` and `N % 16 == 0`; the library is built with oneAPI 2025.3 only.
- Backward `dL/dx`: `dequant_weight` (fused SYCL kernel, 3 bytes per parameter, 63 ms for the 2.17 B quantized parameters of the SDXL UNet,
  3.4x the torch chain) then a bf16 matmul (exact gradient of the int8 network), or `linear_dx_int8` (weight stays int8, `dy` quantized per
  row by a fused Triton kernel; about 1.33x the dense path at M=4096, gradient 1.05-1.35% from the bf16 one against 0.93%).
- `xpu_kernels.triton.nf4.dequant_nf4`: fused, bit-exact against the torch chain, 31x faster (0.66 ms per 100 M parameters), no scratch.
- Gates: each fused kernel verifies itself once per process on first use (it synchronises, so first use must be outside a graph capture);
  one environment switch per kernel. The trainer's three eager warm-up steps per shape cover that before any capture.

## What the trainer does with them (`families/general/quantize/`)
- `backends.py`: one `Int8Backend` interface (`prepare`, `linear`, `dequant`, `dx_int8`); `XpuKernelsBackend` (checks the version, falls back per
  call to the dense weight for shapes it cannot take, with a warning) and `TorchBackend` (exact, slow, the reference the checks run and the
  fallback off an XPU). `make_backend("auto"|"xpu_kernels"|"torch", device)`.
- `ops.py`: `Int8LinearFn` (autograd; frozen weight, so only `dL/dx` comes back) and `Int8Weight` (`grad="dense"|"int8"`).
- `int8.py`: `Int8Linear` for plain frozen linears, `Int8Quantizer` (one layer at a time, so the peak is one dense weight), `quantize_frozen`.
- `AdaptedLinear.use_int8`: the base weight buffer becomes int8 plus `base_scale`; `frozen_weight` rebuilds the dense weight (merge and DoRA need
  it and refuse). Inject Adapter ports: `base_precision`, `int8_backend`, `int8_grad`, `int8_min_params`; the family exposes `quantize_frozen`.
- The trainer warns when an int8 base runs without graph replay.

## Measured on the card
| | bf16 base | int8 base |
|---|---|---|
| weights (UNet, 662 eligible linears of 2.57 B parameters) | 4.89 GB | 2.84 GB (the project's own run: 2059 MB saved, 652 layers) |
| peak at 512, batch 1 (kernels' probe) | 5.04 GB | 3.00 GB |
| eager step | 367 ms | 930 ms (about 2.5x: the path launches about twice the kernels, 85.8 k aten calls against 44.3 k) |
| **graph-replayed step** | **190 ms** | **210 ms (+10-12%)** |
| the project's own run, batch 3 at 512 | about 1010 ms | about 1500 ms eager, 1450 ms with graph (before the fused dequantizer and the new input-gradient path; retest) |

- The INT8 base is worth having **only with graph replay** (`trainer.graph=true`), and then it costs about a tenth of the step for 2 GB.
- It is a different function from the bf16 base (0.9-1.2% from the unquantized weight): the loss curve has to be taken again with it, and the
  trainer's probe compares the base against itself within the same numerics.
- Device time of one bf16 step at 512 (0.263 s over 12 171 kernels): GEMMs 45%, convolutions 22% (not reachable by ConvRot: a conv's contraction is
  spatial; 335 M of the 2.57 B parameters), attention 5%, elementwise and norms 25%.
- Checkpointing's recompute is a third of the captured step (190 ms with it, 142 ms without; peak 5.04 against 6.19 GB). Under capture the host
  cost is gone, so the placement lever (`checkpoint_by=measured`, measured eagerly: 930 against 1010 ms) is worth a bigger share than it showed.
  **Open: the tuner measures eagerly; it should be measured with graph replay in the loop.**
- Host share of an eager step (profiler, shares only): GEMM dispatch 25%, kernel launches 22%, recompute 11%, norms 7.5%, attention 4%, convs 3.4%;
  25 k of 44 k aten calls are view/empty plumbing and cost 4%.
- Profiling on this card: the XPU profiler works (its registration warning is noise). An out-of-memory inside a profiled step loses the device and
  the process dies in a destructor, so it looks like a profiler crash; the protection is the memory budget, not a profiler setting.

## NF4 (not built; plan and arithmetic)
- Storage for 2.17 B parameters: bf16 4.34 GB, int8 2.17 GB, NF4 about 1.26 GB (0.5 + 0.06 absmax + 0.02 double-quantized). NF4 is the next step
  down in memory (about 0.9 GB under int8), not a speed plan: forward is dequant plus bf16 GEMM (no int8 GEMM), and the gradient carries the NF4
  weight error (1-2% against 0.9%). Worth it only where memory is binding and speed is not; the loss baseline must be taken again.
- The old tree's `nf4_weight_store` dequantizes through a torch chain (about 40 bytes and 16 bytes of scratch per parameter, about 200 ms per step,
  and a peak that can exceed the bf16 weight's). The kernel above replaces it: output is the only allocation, safe inside a capture.
- Missing library side: double-quantization of the absmax stays on the trainer's side for now; a packer for the one-time conversion is optional.
- Trainer side when it is built: the same seam (`AdaptedLinear.use_int8` generalised to a weight store, `Int8Weight` next to an `Nf4Weight`);
  layers chosen by the same eligibility rule; the narrow linears and 3x3 convs stay bf16. A hybrid (store NF4, re-quantize to int8 per step) is not
  worth it.

## Takeaways that apply regardless
- Gate and escape hatch per kernel (verify once per process, env switch) for any compiled op the trainer adopts.
- `use_deterministic_algorithms(True)` switches on `fill_uninitialized_memory` (2174 fill kernels, 92 ms per forward on their model); `runtime.deterministic`
  clears it.
- Anima: port rope and adaLN with their autograd wrappers (both are differentiable in the library now), with the same gate; re-derive the
  constants (they are tuned for a 3072-wide, 24-head, 128-dim DiT).
- `sycl::tanh` has a data-dependent slow path on this card (3.5x on saturated inputs); any tanh-based activation on saturating inputs is a candidate.
