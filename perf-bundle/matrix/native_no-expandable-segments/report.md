# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- step 1021 ms, median of 3 unprofiled steps (best 1015); eager 1021 ms, replayed - ms
- the profiled window's steps took 6140 ms each (hooks and profiler); device and module times come from them
- device busy 666 ms, utilisation 65%
- 25413 kernel launches, 188742 operators, 237 us host time per launch
- step peak memory 6411 MB
- phases: loss 0 ms

## Findings, largest first

1. **Much of the module time has no cost model** (2313 ms)
   - 2313 of 6882 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
2. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (185 ms)
   - 191.4 ms per step against an ideal 6.2 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 4352 ms
   - lever: see the module table for the individual layers
3. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (173 ms)
   - 179.5 ms per step against an ideal 6.2 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 4352 ms
   - lever: see the module table for the individual layers
4. **Dtype and copy kernels take a large share** (167 ms)
   - 25% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
5. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (down)** (155 ms)
   - 159.5 ms per step against an ideal 4.1 ms (3% efficient, compute-bound), 24 layers; all rated pieces together leave 4352 ms
   - lever: see the module table for the individual layers
6. **Many small memory-bound kernels** (154 ms)
   - 46% of device time (16088 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
7. **Slowest pieces by recoverable time: linear attn1.to_q (up)** (150 ms)
   - 152.9 ms per step against an ideal 2.8 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 4352 ms
   - lever: see the module table for the individual layers
8. **Slowest pieces by recoverable time: linear net.proj (up)** (147 ms)
   - 169.8 ms per step against an ideal 22.7 ms (13% efficient, compute-bound), 36 layers; all rated pieces together leave 4352 ms
   - lever: see the module table for the individual layers
9. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5250 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up
10. **Module times do not add up to the device time** (0 ms)
   - modules account for 6882 ms of 666 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 244.8 | 37% |
| copy | 4610 | 167.5 | 25% |
| elementwise | 9051 | 116.8 | 18% |
| convolution | 512 | 62.1 | 9% |
| other | 610 | 42.4 | 6% |
| norm | 782 | 19.9 | 3% |
| attention | 140 | 9.1 | 1% |
| memory | 1627 | 2.1 | 0% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 97.7 TFLOPS, int8 200.6 TOPS, copy 396.5 GB/s, dispatch 10.3 us, queued launch 10.3 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention | transformer_blocks.attn1 | up | 36 | 191.4 | 6.2 | 3% | compute | 185.2 |
| attention | transformer_blocks.attn2 | up | 36 | 179.5 | 6.2 | 3% | compute | 173.3 |
| attention | transformer_blocks.attn1 | down | 24 | 159.5 | 4.1 | 3% | compute | 155.4 |
| linear | attn1.to_q | up | 36 | 152.9 | 2.8 | 2% | compute | 150.1 |
| linear | net.proj | up | 36 | 169.8 | 22.7 | 13% | compute | 147.1 |
| attention | transformer_blocks.attn2 | down | 24 | 149.0 | 4.1 | 3% | compute | 144.9 |
| linear | attn2.to_out | up | 36 | 147.2 | 2.8 | 2% | compute | 144.4 |
| linear | attn1.to_k | up | 36 | 146.8 | 2.8 | 2% | compute | 143.9 |
| linear | attn1.to_out | up | 36 | 146.7 | 2.8 | 2% | compute | 143.8 |
| linear | attn2.to_q | up | 36 | 145.9 | 2.8 | 2% | compute | 143.1 |
| linear | attn2.to_k | up | 36 | 144.4 | 1.6 | 1% | memory | 142.8 |
| linear | attn2.to_v | up | 36 | 143.8 | 1.6 | 1% | memory | 142.3 |
| linear | attn1.to_v | up | 36 | 138.6 | 2.8 | 2% | compute | 135.7 |
| linear | ff.net | up | 36 | 145.9 | 11.3 | 8% | compute | 134.6 |
| linear | attn2.to_out | down | 24 | 111.0 | 1.9 | 2% | compute | 109.1 |
| linear | attn1.to_q | down | 24 | 108.6 | 1.9 | 2% | compute | 106.7 |
| linear | attn1.to_k | down | 24 | 107.5 | 1.9 | 2% | compute | 105.6 |
| linear | attn2.to_k | down | 24 | 105.8 | 1.1 | 1% | memory | 104.8 |
| linear | attn2.to_q | down | 24 | 106.0 | 1.9 | 2% | compute | 104.1 |
| linear | attn1.to_out | down | 24 | 105.7 | 1.9 | 2% | compute | 103.8 |
| linear | attn2.to_v | down | 24 | 104.6 | 1.1 | 1% | memory | 103.6 |
| linear | net.proj | down | 24 | 114.5 | 15.1 | 13% | compute | 99.3 |
| linear | ff.net | down | 24 | 106.9 | 7.6 | 7% | compute | 99.3 |
| linear | attn1.to_v | down | 24 | 100.9 | 1.9 | 2% | compute | 99.0 |
| feedforward | transformer_blocks.ff | up | 36 | 107.9 | 20.4 | 19% | memory | 87.6 |
| feedforward | transformer_blocks.ff | down | 24 | 88.9 | 13.6 | 15% | memory | 75.3 |
| attention | transformer_blocks.attn1 | mid | 10 | 57.8 | 0.9 | 2% | memory | 56.9 |
| attention | transformer_blocks.attn2 | mid | 10 | 56.2 | 0.9 | 2% | memory | 55.3 |
| norm | transformer_blocks.norm1 | up | 36 | 47.6 | 1.7 | 4% | memory | 45.9 |
| norm | transformer_blocks.norm3 | up | 36 | 45.9 | 1.7 | 4% | memory | 44.2 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 16.2 | 0.0 | 61.9 | 36.5 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 16.5 | 0.0 | 63.1 | 35.5 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 15.9 | 0.0 | 61.7 | 35.0 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 16.2 | 0.0 | 61.6 | 34.6 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 15.8 | 0.0 | 60.9 | 34.5 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 16.0 | 0.0 | 60.6 | 34.3 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 16.4 | 0.0 | 61.5 | 34.3 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 15.7 | 0.0 | 61.2 | 34.2 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 15.8 | 0.0 | 59.6 | 33.9 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 15.5 | 0.0 | 63.1 | 33.9 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 16.3 | 0.0 | 59.2 | 33.3 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 15.6 | 0.0 | 59.5 | 33.3 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 15.4 | 0.0 | 60.0 | 33.2 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 16.1 | 0.0 | 59.3 | 33.1 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 15.3 | 0.0 | 59.2 | 32.9 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 16.6 | 0.0 | 56.5 | 32.7 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 16.2 | 0.0 | 58.2 | 32.5 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 15.6 | 0.0 | 57.5 | 32.0 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 16.7 | 0.0 | 56.9 | 31.9 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 15.6 | 0.0 | 56.7 | 31.7 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 15.8 | 0.0 | 56.9 | 31.7 | - | - |
| middle_block.1.transformer_blocks.1 | block | 1 | 16.3 | 0.0 | 56.0 | 31.7 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 15.8 | 0.0 | 57.4 | 31.6 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 16.0 | 0.0 | 56.6 | 31.4 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 15.7 | 0.0 | 55.7 | 31.2 | - | - |
| middle_block.1.transformer_blocks.3 | block | 1 | 16.2 | 0.0 | 55.3 | 31.1 | - | - |
| middle_block.1.transformer_blocks.7 | block | 1 | 16.8 | 0.0 | 54.5 | 31.0 | - | - |
| middle_block.1.transformer_blocks.0 | block | 1 | 16.1 | 0.0 | 55.5 | 30.9 | - | - |
| middle_block.1.transformer_blocks.8 | block | 1 | 16.2 | 0.0 | 54.1 | 30.7 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 15.9 | 0.0 | 55.1 | 30.7 | - | - |

## Reconciliation

device 666 ms; modules' own time 6882 ms, of which 2313 ms is in modules with no cost model; wall 1021 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 237.30 | 30 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 76.99 | 36 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2141 | 69.28 | 32 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 56.49 | 19 |
| `gen_conv` | convolution | 131 | 47.16 | 360 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 37.63 | 269 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 17.59 | 84 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 15.29 | 218 |
| `conv_reorder` | convolution | 241 | 13.45 | 56 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.63 | 23 |
| `micro_sdpa` | attention | 140 | 9.10 | 65 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 7.50 | 54 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 7.29 | 52 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 7.03 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.44 | 92 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.99 | 86 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.80 | 23 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 3.11 | 42 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.00 | 2 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.78 | 38 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 210 | 2.26 | 11 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 39 | 2.26 | 58 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 2.11 | 64 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 2.10 | 15 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::SiluFunctor<c10::BFloat16` | elementwise | 96 | 2.02 | 21 |
