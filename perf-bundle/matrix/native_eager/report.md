# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- step 949 ms, median of 3 unprofiled steps (best 945); eager 949 ms, replayed - ms
- the profiled window's steps took 5614 ms each (hooks and profiler); device and module times come from them
- device busy 656 ms, utilisation 69%
- 25413 kernel launches, 188742 operators, 217 us host time per launch
- step peak memory 6303 MB
- phases: loss 0 ms

## Findings, largest first

1. **Much of the module time has no cost model** (2097 ms)
   - 2097 of 6255 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
2. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (168 ms)
   - 174.3 ms per step against an ideal 6.2 ms (4% efficient, compute-bound), 36 layers; all rated pieces together leave 3941 ms
   - lever: see the module table for the individual layers
3. **Dtype and copy kernels take a large share** (160 ms)
   - 24% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
4. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (160 ms)
   - 165.9 ms per step against an ideal 6.2 ms (4% efficient, compute-bound), 36 layers; all rated pieces together leave 3941 ms
   - lever: see the module table for the individual layers
5. **Many small memory-bound kernels** (149 ms)
   - 45% of device time (16088 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
6. **Slowest pieces by recoverable time: linear attn2.to_v (up)** (143 ms)
   - 144.9 ms per step against an ideal 1.6 ms (1% efficient, memory-bound), 36 layers; all rated pieces together leave 3941 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: linear attn1.to_q (up)** (134 ms)
   - 137.0 ms per step against an ideal 2.8 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 3941 ms
   - lever: see the module table for the individual layers
8. **Slowest pieces by recoverable time: linear attn2.to_out (up)** (134 ms)
   - 136.4 ms per step against an ideal 2.8 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 3941 ms
   - lever: see the module table for the individual layers
9. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5006 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up
10. **Module times do not add up to the device time** (0 ms)
   - modules account for 6255 ms of 656 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 243.8 | 37% |
| copy | 4610 | 160.3 | 24% |
| elementwise | 9051 | 114.5 | 17% |
| convolution | 512 | 61.4 | 9% |
| other | 610 | 44.3 | 7% |
| norm | 782 | 18.9 | 3% |
| attention | 140 | 8.8 | 1% |
| memory | 1627 | 2.1 | 0% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 97.7 TFLOPS, int8 200.6 TOPS, copy 396.5 GB/s, dispatch 10.3 us, queued launch 10.3 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention | transformer_blocks.attn1 | up | 36 | 174.3 | 6.2 | 4% | compute | 168.1 |
| attention | transformer_blocks.attn2 | up | 36 | 165.9 | 6.2 | 4% | compute | 159.7 |
| linear | attn2.to_v | up | 36 | 144.9 | 1.6 | 1% | memory | 143.4 |
| linear | attn1.to_q | up | 36 | 137.0 | 2.8 | 2% | compute | 134.2 |
| linear | attn2.to_out | up | 36 | 136.4 | 2.8 | 2% | compute | 133.6 |
| linear | attn1.to_out | up | 36 | 136.0 | 2.8 | 2% | compute | 133.1 |
| linear | attn2.to_q | up | 36 | 135.9 | 2.8 | 2% | compute | 133.1 |
| linear | attn2.to_k | up | 36 | 133.7 | 1.6 | 1% | memory | 132.1 |
| attention | transformer_blocks.attn1 | down | 24 | 135.7 | 4.1 | 3% | compute | 131.6 |
| linear | attn1.to_k | up | 36 | 133.6 | 2.8 | 2% | compute | 130.7 |
| linear | net.proj | up | 36 | 150.6 | 22.7 | 15% | compute | 127.9 |
| linear | attn1.to_v | up | 36 | 127.8 | 2.8 | 2% | compute | 125.0 |
| linear | ff.net | up | 36 | 135.6 | 11.3 | 8% | compute | 124.3 |
| attention | transformer_blocks.attn2 | down | 24 | 127.5 | 4.1 | 3% | compute | 123.4 |
| linear | attn1.to_q | down | 24 | 97.9 | 1.9 | 2% | compute | 96.0 |
| linear | attn2.to_out | down | 24 | 97.6 | 1.9 | 2% | compute | 95.8 |
| linear | attn2.to_q | down | 24 | 96.8 | 1.9 | 2% | compute | 94.9 |
| linear | attn2.to_k | down | 24 | 95.5 | 1.1 | 1% | memory | 94.4 |
| linear | attn1.to_out | down | 24 | 96.2 | 1.9 | 2% | compute | 94.3 |
| linear | attn2.to_v | down | 24 | 95.0 | 1.1 | 1% | memory | 94.0 |
| linear | attn1.to_k | down | 24 | 95.3 | 1.9 | 2% | compute | 93.5 |
| linear | net.proj | down | 24 | 106.1 | 15.1 | 14% | compute | 91.0 |
| linear | ff.net | down | 24 | 97.2 | 7.6 | 8% | compute | 89.7 |
| linear | attn1.to_v | down | 24 | 90.2 | 1.9 | 2% | compute | 88.3 |
| feedforward | transformer_blocks.ff | up | 36 | 97.1 | 20.4 | 21% | memory | 76.7 |
| feedforward | transformer_blocks.ff | down | 24 | 76.5 | 13.6 | 18% | memory | 62.9 |
| attention | transformer_blocks.attn1 | mid | 10 | 50.2 | 0.9 | 2% | memory | 49.3 |
| attention | transformer_blocks.attn2 | mid | 10 | 49.2 | 0.9 | 2% | memory | 48.3 |
| norm | transformer_blocks.norm1 | up | 36 | 44.0 | 1.7 | 4% | memory | 42.3 |
| norm | transformer_blocks.norm2 | up | 36 | 42.8 | 1.7 | 4% | memory | 41.1 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| output_blocks.1.1.transformer_blocks.7 | block | 1 | 15.9 | 0.0 | 53.7 | 34.8 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 14.7 | 0.0 | 54.9 | 31.3 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 15.0 | 0.0 | 53.3 | 31.2 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 14.8 | 0.0 | 55.0 | 31.0 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 14.9 | 0.0 | 54.8 | 30.9 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 14.6 | 0.0 | 53.7 | 30.6 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 15.1 | 0.0 | 54.0 | 30.4 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 14.5 | 0.0 | 53.1 | 30.1 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 14.7 | 0.0 | 53.3 | 30.1 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 14.7 | 0.0 | 52.7 | 29.7 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 14.7 | 0.0 | 55.1 | 29.7 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 14.7 | 0.0 | 52.9 | 29.5 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 15.3 | 0.0 | 50.6 | 29.4 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 14.8 | 0.0 | 52.7 | 29.4 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 14.8 | 0.0 | 52.3 | 29.1 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 14.6 | 0.0 | 51.6 | 29.0 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 14.8 | 0.0 | 51.2 | 28.9 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 15.2 | 0.0 | 50.9 | 28.7 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 15.6 | 0.0 | 50.8 | 28.7 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 15.5 | 0.0 | 50.8 | 28.5 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 15.0 | 0.0 | 51.1 | 28.4 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 14.8 | 0.0 | 50.4 | 28.3 | - | - |
| output_blocks.0.1.transformer_blocks.3 | block | 1 | 15.2 | 0.0 | 50.3 | 28.2 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 15.2 | 0.0 | 50.1 | 28.1 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 15.0 | 0.0 | 49.9 | 28.1 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 14.7 | 0.0 | 50.1 | 28.1 | - | - |
| middle_block.1.transformer_blocks.4 | block | 1 | 14.9 | 0.0 | 49.7 | 28.0 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 14.9 | 0.0 | 50.0 | 28.0 | - | - |
| middle_block.1.transformer_blocks.3 | block | 1 | 15.6 | 0.0 | 49.1 | 27.6 | - | - |
| middle_block.1.transformer_blocks.0 | block | 1 | 14.9 | 0.0 | 49.2 | 27.6 | - | - |

## Reconciliation

device 656 ms; modules' own time 6255 ms, of which 2097 ms is in modules with no cost model; wall 949 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 236.36 | 30 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 73.58 | 34 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2141 | 66.25 | 31 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 55.50 | 18 |
| `gen_conv` | convolution | 131 | 46.16 | 352 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 39.10 | 279 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 17.51 | 83 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 14.40 | 206 |
| `conv_reorder` | convolution | 241 | 13.65 | 57 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.23 | 22 |
| `micro_sdpa` | attention | 140 | 8.81 | 63 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 7.40 | 53 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 6.89 | 49 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.34 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.25 | 89 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.03 | 86 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.33 | 21 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.26 | 2 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 3.06 | 41 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.70 | 36 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 2.58 | 18 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 39 | 2.24 | 57 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 210 | 2.22 | 11 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 2.12 | 64 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::SiluFunctor<c10::BFloat16` | elementwise | 96 | 2.02 | 21 |
