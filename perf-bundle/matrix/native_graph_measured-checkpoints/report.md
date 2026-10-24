# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- step 1730 ms, median of 3 unprofiled steps (best 1726); eager 1730 ms, replayed - ms
- the profiled window's steps took 10396 ms each (hooks and profiler); device and module times come from them
- device busy 654 ms, utilisation 38%
- 42062 kernel launches, 321896 operators, 229 us host time per launch
- step peak memory 7641 MB
- phases: loss 0 ms

## Findings, largest first

1. **Much of the module time has no cost model** (2938 ms)
   - 2938 of 10892 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
2. **The step is launch bound** (1075 ms)
   - the device is busy 38% of the step (654 of 1730 ms) over 42062 launches and 321896 operators per step, 229 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
3. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (363 ms)
   - 369.3 ms per step against an ideal 5.8 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 7763 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (336 ms)
   - 341.6 ms per step against an ideal 5.8 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 7763 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: linear attn2.to_q (up)** (291 ms)
   - 293.5 ms per step against an ideal 2.6 ms (1% efficient, memory-bound), 36 layers; all rated pieces together leave 7763 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (down)** (281 ms)
   - 284.6 ms per step against an ideal 3.8 ms (1% efficient, compute-bound), 24 layers; all rated pieces together leave 7763 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: linear attn2.to_out (up)** (275 ms)
   - 277.3 ms per step against an ideal 2.6 ms (1% efficient, memory-bound), 36 layers; all rated pieces together leave 7763 ms
   - lever: see the module table for the individual layers
8. **Dtype and copy kernels take a large share** (153 ms)
   - 23% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
9. **Many small memory-bound kernels** (139 ms)
   - 43% of device time (25728 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
10. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 6005 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up
11. **Module times do not add up to the device time** (0 ms)
   - modules account for 10892 ms of 654 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 14114 | 263.3 | 40% |
| copy | 7852 | 152.5 | 23% |
| elementwise | 14705 | 101.9 | 16% |
| convolution | 877 | 59.4 | 9% |
| other | 1179 | 46.1 | 7% |
| norm | 1350 | 20.6 | 3% |
| attention | 164 | 7.1 | 1% |
| memory | 1806 | 2.2 | 0% |
| reduce | 15 | 1.4 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 104.6 TFLOPS, int8 199.4 TOPS, copy 386.2 GB/s, dispatch 10.5 us, queued launch 10.5 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention | transformer_blocks.attn2 | up | 36 | 369.3 | 5.8 | 2% | compute | 363.5 |
| attention | transformer_blocks.attn1 | up | 36 | 341.6 | 5.8 | 2% | compute | 335.8 |
| linear | attn2.to_q | up | 36 | 293.5 | 2.6 | 1% | memory | 291.0 |
| attention | transformer_blocks.attn2 | down | 24 | 284.6 | 3.8 | 1% | compute | 280.8 |
| linear | attn2.to_out | up | 36 | 277.3 | 2.6 | 1% | memory | 274.7 |
| linear | attn1.to_out | up | 36 | 276.3 | 2.6 | 1% | memory | 273.7 |
| linear | ff.net | up | 36 | 282.6 | 9.4 | 3% | compute | 273.2 |
| linear | attn1.to_q | up | 36 | 273.8 | 2.6 | 1% | memory | 271.2 |
| linear | attn1.to_k | up | 36 | 272.7 | 2.6 | 1% | memory | 270.1 |
| linear | net.proj | up | 36 | 287.0 | 18.8 | 7% | compute | 268.1 |
| linear | attn1.to_v | up | 36 | 266.7 | 2.6 | 1% | memory | 264.1 |
| attention | transformer_blocks.attn1 | down | 24 | 239.5 | 3.8 | 2% | compute | 235.8 |
| linear | attn2.to_k | up | 36 | 228.0 | 2.6 | 1% | memory | 225.3 |
| linear | attn2.to_v | up | 36 | 225.4 | 2.6 | 1% | memory | 222.8 |
| feedforward | transformer_blocks.ff | up | 36 | 203.8 | 19.4 | 10% | memory | 184.4 |
| linear | attn1.to_v | down | 24 | 184.9 | 1.6 | 1% | memory | 183.3 |
| linear | ff.net | down | 24 | 189.1 | 5.8 | 3% | compute | 183.3 |
| linear | attn2.to_out | down | 24 | 183.0 | 1.6 | 1% | memory | 181.4 |
| linear | attn1.to_out | down | 24 | 182.8 | 1.6 | 1% | memory | 181.2 |
| linear | attn2.to_q | down | 24 | 180.8 | 1.6 | 1% | memory | 179.2 |
| linear | net.proj | down | 24 | 189.8 | 11.6 | 6% | compute | 178.2 |
| linear | attn1.to_q | down | 24 | 179.2 | 1.6 | 1% | memory | 177.6 |
| linear | attn1.to_k | down | 24 | 176.5 | 1.6 | 1% | memory | 174.9 |
| linear | attn2.to_k | down | 24 | 134.1 | 1.6 | 1% | memory | 132.5 |
| feedforward | transformer_blocks.ff | down | 24 | 144.1 | 12.3 | 9% | memory | 131.8 |
| linear | attn2.to_v | down | 24 | 133.3 | 1.6 | 1% | memory | 131.7 |
| attention | transformer_blocks.attn2 | mid | 10 | 110.4 | 0.9 | 1% | memory | 109.6 |
| attention | transformer_blocks.attn1 | mid | 10 | 96.7 | 0.9 | 1% | memory | 95.8 |
| norm | transformer_blocks.norm1 | up | 36 | 86.6 | 1.6 | 2% | memory | 85.0 |
| norm | transformer_blocks.norm3 | up | 36 | 83.6 | 1.6 | 2% | memory | 82.0 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.5.1.transformer_blocks.1 | block | 2 | 32.6 | 0.0 | 119.9 | 69.0 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 2 | 32.4 | 0.0 | 123.1 | 67.4 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 2 | 33.2 | 0.0 | 121.6 | 67.2 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 2 | 33.4 | 0.0 | 117.6 | 66.5 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 2 | 32.4 | 0.0 | 118.7 | 66.1 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 2 | 32.5 | 0.0 | 116.5 | 65.7 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 2 | 32.1 | 0.0 | 116.1 | 64.8 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 2 | 31.6 | 0.0 | 112.2 | 62.8 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 2 | 32.7 | 0.0 | 110.6 | 62.2 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 2 | 32.4 | 0.0 | 109.0 | 61.5 | - | - |
| middle_block.1.transformer_blocks.1 | block | 2 | 32.4 | 0.0 | 108.6 | 60.3 | - | - |
| middle_block.1.transformer_blocks.9 | block | 2 | 33.9 | 0.0 | 104.6 | 60.1 | - | - |
| output_blocks.0.1.transformer_blocks.1 | block | 2 | 34.0 | 0.0 | 107.1 | 60.1 | - | - |
| middle_block.1.transformer_blocks.2 | block | 2 | 32.2 | 0.0 | 107.2 | 60.0 | - | - |
| output_blocks.0.1.transformer_blocks.0 | block | 2 | 40.9 | 0.0 | 105.7 | 59.7 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 2 | 32.2 | 0.0 | 107.9 | 59.6 | - | - |
| middle_block.1.transformer_blocks.6 | block | 2 | 32.2 | 0.0 | 107.3 | 59.3 | - | - |
| middle_block.1.transformer_blocks.5 | block | 2 | 32.6 | 0.0 | 106.4 | 58.9 | - | - |
| output_blocks.0.1.transformer_blocks.4 | block | 2 | 33.4 | 0.0 | 103.3 | 58.4 | - | - |
| middle_block.1.transformer_blocks.7 | block | 2 | 32.6 | 0.0 | 104.3 | 57.8 | - | - |
| output_blocks.0.1.transformer_blocks.7 | block | 2 | 34.3 | 0.0 | 102.5 | 57.3 | - | - |
| output_blocks.0.1.transformer_blocks.5 | block | 2 | 33.9 | 0.0 | 101.7 | 56.9 | - | - |
| output_blocks.1.1.transformer_blocks.0 | block | 2 | 36.3 | 0.0 | 100.2 | 56.8 | - | - |
| output_blocks.1.1.transformer_blocks.8 | block | 2 | 35.2 | 0.0 | 100.5 | 56.0 | - | - |
| output_blocks.1.1.transformer_blocks.9 | block | 2 | 35.0 | 0.0 | 98.8 | 55.8 | - | - |
| output_blocks.1.1.transformer_blocks.2 | block | 2 | 34.8 | 0.0 | 99.7 | 55.5 | - | - |
| output_blocks.1.1.transformer_blocks.5 | block | 2 | 34.5 | 0.0 | 99.1 | 55.4 | - | - |
| output_blocks.1.1.transformer_blocks.7 | block | 2 | 34.7 | 0.0 | 99.9 | 55.4 | - | - |
| output_blocks.1.1.transformer_blocks.3 | block | 2 | 33.7 | 0.0 | 100.1 | 55.2 | - | - |
| output_blocks.2.1.transformer_blocks.0 | block | 2 | 34.8 | 0.0 | 98.2 | 54.9 | - | - |

## Reconciliation

device 654 ms; modules' own time 10892 ms, of which 2938 ms is in modules with no cost model; wall 1730 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 13834 | 253.55 | 18 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 3598 | 67.49 | 19 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 3710 | 65.12 | 18 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 5162 | 46.69 | 9 |
| `gen_conv` | convolution | 210 | 42.30 | 201 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 280 | 40.26 | 144 |
| `conv_reorder` | convolution | 387 | 16.05 | 42 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 140 | 13.58 | 97 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 362 | 10.60 | 29 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 666 | 9.42 | 14 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 5227 | 8.75 | 2 |
| `micro_sdpa` | attention | 164 | 7.08 | 43 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 140 | 6.50 | 46 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 200 | 6.40 | 32 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 420 | 6.03 | 14 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 222 | 5.69 | 26 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 2912 | 5.10 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 140 | 4.61 | 33 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 108 | 4.35 | 40 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 80 | 3.36 | 42 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 3.21 | 97 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 420 | 2.89 | 7 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 108 | 2.74 | 25 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 280 | 2.38 | 9 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 78 | 2.25 | 29 |
