# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- step 1036 ms, median of 3 unprofiled steps (best 1004); eager 1036 ms, replayed - ms
- the profiled window's steps took 6344 ms each (hooks and profiler); device and module times come from them
- device busy 669 ms, utilisation 65%
- 25413 kernel launches, 188742 operators, 245 us host time per launch
- step peak memory 6303 MB
- phases: loss 0 ms

## Findings, largest first

1. **Much of the module time has no cost model** (2418 ms)
   - 2418 of 7138 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
2. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (198 ms)
   - 204.5 ms per step against an ideal 6.0 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 4510 ms
   - lever: see the module table for the individual layers
3. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (186 ms)
   - 192.1 ms per step against an ideal 6.0 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 4510 ms
   - lever: see the module table for the individual layers
4. **Dtype and copy kernels take a large share** (169 ms)
   - 25% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
5. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (down)** (165 ms)
   - 169.4 ms per step against an ideal 4.0 ms (2% efficient, compute-bound), 24 layers; all rated pieces together leave 4510 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: linear attn1.to_q (up)** (158 ms)
   - 160.7 ms per step against an ideal 2.6 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 4510 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (down)** (157 ms)
   - 161.1 ms per step against an ideal 4.0 ms (2% efficient, compute-bound), 24 layers; all rated pieces together leave 4510 ms
   - lever: see the module table for the individual layers
8. **Many small memory-bound kernels** (155 ms)
   - 46% of device time (16088 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
9. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5280 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up
10. **Module times do not add up to the device time** (0 ms)
   - modules account for 7138 ms of 669 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 241.8 | 36% |
| copy | 4610 | 168.5 | 25% |
| elementwise | 9051 | 117.2 | 18% |
| convolution | 512 | 61.7 | 9% |
| other | 610 | 46.0 | 7% |
| norm | 782 | 20.5 | 3% |
| attention | 140 | 9.7 | 1% |
| memory | 1627 | 2.2 | 0% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 104.6 TFLOPS, int8 199.4 TOPS, copy 386.2 GB/s, dispatch 10.5 us, queued launch 10.5 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention | transformer_blocks.attn1 | up | 36 | 204.5 | 6.0 | 3% | compute | 198.4 |
| attention | transformer_blocks.attn2 | up | 36 | 192.1 | 6.0 | 3% | compute | 186.0 |
| attention | transformer_blocks.attn1 | down | 24 | 169.4 | 4.0 | 2% | compute | 165.4 |
| linear | attn1.to_q | up | 36 | 160.7 | 2.6 | 2% | compute | 158.1 |
| attention | transformer_blocks.attn2 | down | 24 | 161.1 | 4.0 | 2% | compute | 157.0 |
| linear | attn2.to_out | up | 36 | 154.0 | 2.6 | 2% | compute | 151.3 |
| linear | attn1.to_out | up | 36 | 153.8 | 2.6 | 2% | compute | 151.1 |
| linear | attn2.to_v | up | 36 | 149.4 | 1.6 | 1% | memory | 147.8 |
| linear | attn2.to_q | up | 36 | 149.9 | 2.6 | 2% | compute | 147.2 |
| linear | attn2.to_k | up | 36 | 148.7 | 1.6 | 1% | memory | 147.1 |
| linear | attn1.to_k | up | 36 | 148.6 | 2.6 | 2% | compute | 145.9 |
| linear | ff.net | up | 36 | 151.9 | 10.6 | 7% | compute | 141.4 |
| linear | net.proj | up | 36 | 162.3 | 21.2 | 13% | compute | 141.2 |
| linear | attn1.to_v | up | 36 | 141.3 | 2.6 | 2% | compute | 138.7 |
| linear | attn1.to_out | down | 24 | 111.8 | 1.8 | 2% | compute | 110.1 |
| linear | attn1.to_q | down | 24 | 109.9 | 1.8 | 2% | compute | 108.1 |
| linear | attn2.to_out | down | 24 | 109.5 | 1.8 | 2% | compute | 107.8 |
| linear | attn1.to_k | down | 24 | 107.8 | 1.8 | 2% | compute | 106.1 |
| linear | attn2.to_v | down | 24 | 106.7 | 1.1 | 1% | memory | 105.6 |
| linear | attn2.to_k | down | 24 | 106.5 | 1.1 | 1% | memory | 105.4 |
| linear | attn2.to_q | down | 24 | 107.0 | 1.8 | 2% | compute | 105.2 |
| linear | ff.net | down | 24 | 111.5 | 7.1 | 6% | compute | 104.4 |
| linear | net.proj | down | 24 | 116.5 | 14.1 | 12% | compute | 102.4 |
| linear | attn1.to_v | down | 24 | 102.0 | 1.8 | 2% | compute | 100.3 |
| feedforward | transformer_blocks.ff | up | 36 | 115.7 | 20.9 | 18% | memory | 94.8 |
| feedforward | transformer_blocks.ff | down | 24 | 97.9 | 13.9 | 14% | memory | 83.9 |
| attention | transformer_blocks.attn1 | mid | 10 | 60.0 | 0.9 | 2% | memory | 59.1 |
| attention | transformer_blocks.attn2 | mid | 10 | 58.5 | 0.9 | 2% | memory | 57.5 |
| norm | transformer_blocks.norm3 | up | 36 | 49.0 | 1.7 | 4% | memory | 47.3 |
| norm | transformer_blocks.norm1 | up | 36 | 49.0 | 1.7 | 4% | memory | 47.2 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 16.1 | 0.0 | 70.5 | 40.3 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 15.9 | 0.0 | 69.9 | 39.5 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 15.8 | 0.0 | 69.5 | 39.3 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 15.8 | 0.0 | 67.6 | 37.9 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 15.2 | 0.0 | 65.1 | 37.3 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 16.3 | 0.0 | 65.3 | 36.2 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 16.2 | 0.0 | 62.9 | 35.8 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 16.0 | 0.0 | 65.8 | 35.7 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 15.7 | 0.0 | 63.2 | 35.6 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 15.5 | 0.0 | 65.9 | 35.4 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 15.6 | 0.0 | 63.2 | 35.4 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 16.3 | 0.0 | 61.6 | 35.3 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 16.3 | 0.0 | 63.6 | 35.2 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 16.1 | 0.0 | 58.8 | 34.8 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 15.7 | 0.0 | 61.5 | 34.8 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 15.6 | 0.0 | 60.3 | 34.6 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 15.3 | 0.0 | 60.1 | 33.7 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 15.6 | 0.0 | 60.3 | 33.5 | - | - |
| middle_block.1.transformer_blocks.1 | block | 1 | 15.7 | 0.0 | 59.1 | 32.9 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 15.5 | 0.0 | 57.9 | 32.8 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 15.6 | 0.0 | 58.1 | 32.6 | - | - |
| middle_block.1.transformer_blocks.3 | block | 1 | 16.6 | 0.0 | 58.2 | 32.5 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 15.8 | 0.0 | 57.9 | 32.4 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 15.6 | 0.0 | 57.7 | 32.2 | - | - |
| middle_block.1.transformer_blocks.0 | block | 1 | 15.4 | 0.0 | 55.6 | 32.0 | - | - |
| middle_block.1.transformer_blocks.7 | block | 1 | 15.9 | 0.0 | 55.5 | 31.8 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 15.6 | 0.0 | 55.5 | 31.6 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 15.9 | 0.0 | 56.1 | 31.5 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 15.8 | 0.0 | 56.1 | 31.2 | - | - |
| middle_block.1.transformer_blocks.6 | block | 1 | 15.8 | 0.0 | 55.4 | 31.2 | - | - |

## Reconciliation

device 669 ms; modules' own time 7138 ms, of which 2418 ms is in modules with no cost model; wall 1036 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 234.35 | 30 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 76.41 | 36 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2141 | 70.20 | 33 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 56.73 | 19 |
| `gen_conv` | convolution | 131 | 47.40 | 362 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 40.59 | 290 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 17.63 | 84 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 14.79 | 211 |
| `conv_reorder` | convolution | 241 | 12.83 | 53 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 10.09 | 24 |
| `micro_sdpa` | attention | 140 | 9.70 | 69 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 7.48 | 53 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 7.06 | 50 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.73 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.32 | 90 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.29 | 90 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.80 | 23 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 3.27 | 44 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.23 | 2 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 2.94 | 89 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.91 | 39 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 2.61 | 19 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 210 | 2.46 | 12 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 39 | 2.27 | 58 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::SiluFunctor<c10::BFloat16` | elementwise | 96 | 2.00 | 21 |
