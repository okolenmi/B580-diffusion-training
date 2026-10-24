# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- step 1024 ms, median of 3 unprofiled steps (best 1020); eager 1024 ms, replayed - ms
- the profiled window's steps took 6112 ms each (hooks and profiler); device and module times come from them
- device busy 664 ms, utilisation 65%
- 25413 kernel launches, 188742 operators, 236 us host time per launch
- step peak memory 6303 MB
- phases: loss 0 ms

## Findings, largest first

1. **Much of the module time has no cost model** (2302 ms)
   - 2302 of 6843 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
2. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (185 ms)
   - 190.8 ms per step against an ideal 6.2 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 4324 ms
   - lever: see the module table for the individual layers
3. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (173 ms)
   - 178.9 ms per step against an ideal 6.2 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 4324 ms
   - lever: see the module table for the individual layers
4. **Dtype and copy kernels take a large share** (166 ms)
   - 25% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
5. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (down)** (154 ms)
   - 158.3 ms per step against an ideal 4.1 ms (3% efficient, compute-bound), 24 layers; all rated pieces together leave 4324 ms
   - lever: see the module table for the individual layers
6. **Many small memory-bound kernels** (153 ms)
   - 46% of device time (16088 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
7. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (down)** (146 ms)
   - 149.7 ms per step against an ideal 4.1 ms (3% efficient, compute-bound), 24 layers; all rated pieces together leave 4324 ms
   - lever: see the module table for the individual layers
8. **Slowest pieces by recoverable time: linear attn1.to_q (up)** (145 ms)
   - 147.6 ms per step against an ideal 2.8 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 4324 ms
   - lever: see the module table for the individual layers
9. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5285 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up
10. **Module times do not add up to the device time** (0 ms)
   - modules account for 6843 ms of 664 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 241.9 | 36% |
| copy | 4610 | 166.3 | 25% |
| elementwise | 9051 | 116.3 | 18% |
| convolution | 512 | 61.7 | 9% |
| other | 610 | 45.0 | 7% |
| norm | 782 | 19.7 | 3% |
| attention | 140 | 9.1 | 1% |
| memory | 1627 | 2.1 | 0% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 97.7 TFLOPS, int8 200.6 TOPS, copy 396.5 GB/s, dispatch 10.3 us, queued launch 10.3 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention | transformer_blocks.attn1 | up | 36 | 190.8 | 6.2 | 3% | compute | 184.7 |
| attention | transformer_blocks.attn2 | up | 36 | 178.9 | 6.2 | 3% | compute | 172.7 |
| attention | transformer_blocks.attn1 | down | 24 | 158.3 | 4.1 | 3% | compute | 154.1 |
| attention | transformer_blocks.attn2 | down | 24 | 149.7 | 4.1 | 3% | compute | 145.5 |
| linear | attn1.to_q | up | 36 | 147.6 | 2.8 | 2% | compute | 144.8 |
| linear | attn2.to_k | up | 36 | 145.2 | 1.6 | 1% | memory | 143.7 |
| linear | attn2.to_out | up | 36 | 146.3 | 2.8 | 2% | compute | 143.5 |
| linear | attn1.to_out | up | 36 | 146.3 | 2.8 | 2% | compute | 143.5 |
| linear | attn1.to_v | up | 36 | 145.8 | 2.8 | 2% | compute | 143.0 |
| linear | attn2.to_q | up | 36 | 145.8 | 2.8 | 2% | compute | 142.9 |
| linear | attn2.to_v | up | 36 | 143.9 | 1.6 | 1% | memory | 142.3 |
| linear | attn1.to_k | up | 36 | 143.1 | 2.8 | 2% | compute | 140.3 |
| linear | net.proj | up | 36 | 159.6 | 22.7 | 14% | compute | 136.9 |
| linear | ff.net | up | 36 | 146.8 | 11.3 | 8% | compute | 135.4 |
| linear | attn2.to_k | down | 24 | 107.4 | 1.1 | 1% | memory | 106.4 |
| linear | attn2.to_out | down | 24 | 106.6 | 1.9 | 2% | compute | 104.7 |
| linear | attn1.to_q | down | 24 | 106.5 | 1.9 | 2% | compute | 104.6 |
| linear | attn2.to_v | down | 24 | 105.3 | 1.1 | 1% | memory | 104.2 |
| linear | attn2.to_q | down | 24 | 105.9 | 1.9 | 2% | compute | 104.0 |
| linear | attn1.to_out | down | 24 | 105.3 | 1.9 | 2% | compute | 103.5 |
| linear | attn1.to_k | down | 24 | 105.3 | 1.9 | 2% | compute | 103.4 |
| linear | attn1.to_v | down | 24 | 100.1 | 1.9 | 2% | compute | 98.2 |
| linear | net.proj | down | 24 | 113.3 | 15.1 | 13% | compute | 98.1 |
| linear | ff.net | down | 24 | 105.7 | 7.6 | 7% | compute | 98.1 |
| feedforward | transformer_blocks.ff | up | 36 | 108.6 | 20.4 | 19% | memory | 88.2 |
| feedforward | transformer_blocks.ff | down | 24 | 89.9 | 13.6 | 15% | memory | 76.3 |
| attention | transformer_blocks.attn1 | mid | 10 | 56.9 | 0.9 | 2% | memory | 56.0 |
| attention | transformer_blocks.attn2 | mid | 10 | 55.7 | 0.9 | 2% | memory | 54.8 |
| norm | transformer_blocks.norm1 | up | 36 | 51.3 | 1.7 | 3% | memory | 49.6 |
| norm | transformer_blocks.norm3 | up | 36 | 46.4 | 1.7 | 4% | memory | 44.7 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 15.2 | 0.0 | 63.1 | 36.4 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 15.3 | 0.0 | 62.1 | 35.6 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 16.8 | 0.0 | 62.4 | 35.5 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 15.9 | 0.0 | 62.2 | 35.0 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 15.9 | 0.0 | 62.0 | 35.0 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 15.7 | 0.0 | 61.5 | 34.6 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 16.1 | 0.0 | 61.5 | 34.5 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 15.6 | 0.0 | 60.9 | 34.1 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 15.9 | 0.0 | 61.2 | 34.1 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 15.2 | 0.0 | 60.0 | 33.9 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 15.3 | 0.0 | 59.8 | 33.7 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 15.7 | 0.0 | 58.7 | 33.3 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 15.4 | 0.0 | 59.7 | 33.1 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 15.6 | 0.0 | 61.9 | 33.0 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 15.8 | 0.0 | 56.9 | 32.6 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 15.2 | 0.0 | 59.0 | 32.6 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 15.5 | 0.0 | 56.4 | 31.9 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 15.8 | 0.0 | 56.1 | 31.6 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 16.1 | 0.0 | 55.8 | 31.5 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 15.7 | 0.0 | 56.4 | 31.5 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 15.7 | 0.0 | 55.8 | 31.4 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 16.1 | 0.0 | 56.7 | 31.3 | - | - |
| middle_block.1.transformer_blocks.0 | block | 1 | 15.8 | 0.0 | 56.3 | 31.2 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 15.9 | 0.0 | 55.4 | 31.2 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 15.5 | 0.0 | 56.4 | 31.1 | - | - |
| middle_block.1.transformer_blocks.1 | block | 1 | 15.7 | 0.0 | 55.4 | 31.0 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 15.4 | 0.0 | 56.2 | 31.0 | - | - |
| middle_block.1.transformer_blocks.4 | block | 1 | 15.4 | 0.0 | 54.4 | 30.6 | - | - |
| output_blocks.0.1.transformer_blocks.2 | block | 1 | 16.5 | 0.0 | 54.9 | 30.3 | - | - |
| middle_block.1.transformer_blocks.7 | block | 1 | 16.1 | 0.0 | 53.3 | 30.2 | - | - |

## Reconciliation

device 664 ms; modules' own time 6843 ms, of which 2302 ms is in modules with no cost model; wall 1024 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 234.43 | 30 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 75.64 | 35 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2141 | 68.67 | 32 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 56.09 | 18 |
| `gen_conv` | convolution | 131 | 47.14 | 360 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 39.75 | 284 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 17.79 | 85 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 14.88 | 213 |
| `conv_reorder` | convolution | 241 | 12.95 | 54 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.45 | 23 |
| `micro_sdpa` | attention | 140 | 9.11 | 65 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 7.47 | 53 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 7.20 | 51 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.56 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.29 | 90 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.93 | 85 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.69 | 22 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 3.29 | 44 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.23 | 2 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 2.98 | 90 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.74 | 37 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 2.61 | 19 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 39 | 2.25 | 58 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 210 | 2.24 | 11 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::SiluFunctor<c10::BFloat16` | elementwise | 96 | 2.02 | 21 |
