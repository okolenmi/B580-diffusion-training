# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- step 965 ms, median of 3 unprofiled steps (best 963); eager 965 ms, replayed - ms
- the profiled window's steps took 5621 ms each (hooks and profiler); device and module times come from them
- device busy 658 ms, utilisation 68%
- 25413 kernel launches, 188742 operators, 217 us host time per launch
- step peak memory 6411 MB
- phases: loss 0 ms

## Findings, largest first

1. **Much of the module time has no cost model** (2082 ms)
   - 2082 of 6230 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
2. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (167 ms)
   - 173.1 ms per step against an ideal 6.0 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 3939 ms
   - lever: see the module table for the individual layers
3. **Dtype and copy kernels take a large share** (162 ms)
   - 25% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
4. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (155 ms)
   - 161.3 ms per step against an ideal 6.0 ms (4% efficient, compute-bound), 36 layers; all rated pieces together leave 3939 ms
   - lever: see the module table for the individual layers
5. **Many small memory-bound kernels** (150 ms)
   - 46% of device time (16088 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
6. **Slowest pieces by recoverable time: linear attn2.to_v (up)** (142 ms)
   - 143.3 ms per step against an ideal 1.6 ms (1% efficient, memory-bound), 36 layers; all rated pieces together leave 3939 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: linear attn1.to_q (up)** (134 ms)
   - 136.8 ms per step against an ideal 2.6 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 3939 ms
   - lever: see the module table for the individual layers
8. **Slowest pieces by recoverable time: linear attn1.to_out (up)** (133 ms)
   - 135.7 ms per step against an ideal 2.6 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 3939 ms
   - lever: see the module table for the individual layers
9. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 4997 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up
10. **Module times do not add up to the device time** (0 ms)
   - modules account for 6230 ms of 658 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 246.1 | 37% |
| copy | 4610 | 162.1 | 25% |
| elementwise | 9051 | 115.1 | 17% |
| convolution | 512 | 62.2 | 9% |
| other | 610 | 41.1 | 6% |
| norm | 782 | 19.0 | 3% |
| attention | 140 | 8.9 | 1% |
| memory | 1627 | 2.1 | 0% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 104.6 TFLOPS, int8 199.4 TOPS, copy 386.2 GB/s, dispatch 10.5 us, queued launch 10.5 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention | transformer_blocks.attn1 | up | 36 | 173.1 | 6.0 | 3% | compute | 167.0 |
| attention | transformer_blocks.attn2 | up | 36 | 161.3 | 6.0 | 4% | compute | 155.3 |
| linear | attn2.to_v | up | 36 | 143.3 | 1.6 | 1% | memory | 141.7 |
| linear | attn1.to_q | up | 36 | 136.8 | 2.6 | 2% | compute | 134.2 |
| linear | attn1.to_out | up | 36 | 135.7 | 2.6 | 2% | compute | 133.1 |
| linear | attn2.to_out | up | 36 | 135.6 | 2.6 | 2% | compute | 133.0 |
| linear | attn2.to_q | up | 36 | 135.3 | 2.6 | 2% | compute | 132.7 |
| linear | attn2.to_k | up | 36 | 133.4 | 1.6 | 1% | memory | 131.7 |
| linear | attn1.to_v | up | 36 | 134.1 | 2.6 | 2% | compute | 131.5 |
| linear | attn1.to_k | up | 36 | 133.5 | 2.6 | 2% | compute | 130.9 |
| attention | transformer_blocks.attn1 | down | 24 | 134.7 | 4.0 | 3% | compute | 130.7 |
| linear | net.proj | up | 36 | 149.2 | 21.2 | 14% | compute | 128.0 |
| linear | ff.net | up | 36 | 135.2 | 10.6 | 8% | compute | 124.6 |
| attention | transformer_blocks.attn2 | down | 24 | 126.9 | 4.0 | 3% | compute | 122.8 |
| linear | attn1.to_q | down | 24 | 97.4 | 1.8 | 2% | compute | 95.7 |
| linear | attn2.to_out | down | 24 | 97.0 | 1.8 | 2% | compute | 95.2 |
| linear | attn1.to_k | down | 24 | 96.8 | 1.8 | 2% | compute | 95.0 |
| linear | attn2.to_q | down | 24 | 96.3 | 1.8 | 2% | compute | 94.6 |
| linear | attn2.to_k | down | 24 | 95.4 | 1.1 | 1% | memory | 94.3 |
| linear | attn2.to_v | down | 24 | 94.8 | 1.1 | 1% | memory | 93.7 |
| linear | attn1.to_out | down | 24 | 95.4 | 1.8 | 2% | compute | 93.6 |
| linear | ff.net | down | 24 | 99.5 | 7.1 | 7% | compute | 92.5 |
| linear | net.proj | down | 24 | 103.7 | 14.1 | 14% | compute | 89.6 |
| linear | attn1.to_v | down | 24 | 89.9 | 1.8 | 2% | compute | 88.1 |
| feedforward | transformer_blocks.ff | up | 36 | 96.1 | 20.9 | 22% | memory | 75.2 |
| feedforward | transformer_blocks.ff | down | 24 | 76.0 | 13.9 | 18% | memory | 62.0 |
| attention | transformer_blocks.attn1 | mid | 10 | 51.0 | 0.9 | 2% | memory | 50.1 |
| attention | transformer_blocks.attn2 | mid | 10 | 50.0 | 0.9 | 2% | memory | 49.1 |
| norm | transformer_blocks.norm1 | up | 36 | 43.9 | 1.7 | 4% | memory | 42.1 |
| norm | transformer_blocks.norm2 | up | 36 | 42.5 | 1.7 | 4% | memory | 40.8 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 15.1 | 0.0 | 54.3 | 32.3 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 15.1 | 0.0 | 54.8 | 31.4 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 15.4 | 0.0 | 54.5 | 30.9 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 15.1 | 0.0 | 54.6 | 30.8 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 15.0 | 0.0 | 53.5 | 30.5 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 15.5 | 0.0 | 53.9 | 30.5 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 15.2 | 0.0 | 52.8 | 29.9 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 15.0 | 0.0 | 52.9 | 29.7 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 14.9 | 0.0 | 52.1 | 29.4 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 14.9 | 0.0 | 54.7 | 29.4 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 14.8 | 0.0 | 51.9 | 29.4 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 15.1 | 0.0 | 52.4 | 29.2 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 15.1 | 0.0 | 52.9 | 29.1 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 15.0 | 0.0 | 50.7 | 29.1 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 15.8 | 0.0 | 50.0 | 29.0 | - | - |
| middle_block.1.transformer_blocks.1 | block | 1 | 15.1 | 0.0 | 50.2 | 28.7 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 15.1 | 0.0 | 50.8 | 28.6 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 15.0 | 0.0 | 50.1 | 28.3 | - | - |
| middle_block.1.transformer_blocks.6 | block | 1 | 15.5 | 0.0 | 49.9 | 28.3 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 14.9 | 0.0 | 50.4 | 28.1 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 15.0 | 0.0 | 50.2 | 28.1 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 15.0 | 0.0 | 50.5 | 28.1 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 15.0 | 0.0 | 49.8 | 28.0 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 14.8 | 0.0 | 49.5 | 27.9 | - | - |
| middle_block.1.transformer_blocks.7 | block | 1 | 15.5 | 0.0 | 49.2 | 27.8 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 14.9 | 0.0 | 49.4 | 27.8 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 15.1 | 0.0 | 49.1 | 27.8 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 15.4 | 0.0 | 49.4 | 27.8 | - | - |
| middle_block.1.transformer_blocks.8 | block | 1 | 15.3 | 0.0 | 48.8 | 27.8 | - | - |
| middle_block.1.transformer_blocks.5 | block | 1 | 15.4 | 0.0 | 49.4 | 27.7 | - | - |

## Reconciliation

device 658 ms; modules' own time 6230 ms, of which 2082 ms is in modules with no cost model; wall 965 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 238.69 | 30 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 75.25 | 35 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2141 | 66.66 | 31 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 55.80 | 18 |
| `gen_conv` | convolution | 131 | 46.53 | 355 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 36.34 | 260 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 17.54 | 84 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 14.87 | 212 |
| `conv_reorder` | convolution | 241 | 14.21 | 59 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.19 | 22 |
| `micro_sdpa` | attention | 140 | 8.88 | 63 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 7.45 | 53 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 6.87 | 49 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.74 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.26 | 89 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.04 | 86 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.47 | 21 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 3.04 | 41 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 2.99 | 2 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.70 | 37 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 39 | 2.24 | 57 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 210 | 2.20 | 10 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 2.19 | 16 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::SiluFunctor<c10::BFloat16` | elementwise | 96 | 2.02 | 21 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 1.74 | 53 |
