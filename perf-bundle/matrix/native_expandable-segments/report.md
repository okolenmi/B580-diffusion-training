# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- step 1030 ms, median of 3 unprofiled steps (best 1021); eager 1030 ms, replayed - ms
- the profiled window's steps took 6333 ms each (hooks and profiler); device and module times come from them
- device busy 667 ms, utilisation 65%
- 25413 kernel launches, 188742 operators, 245 us host time per launch
- step peak memory 6303 MB
- phases: loss 0 ms

## Findings, largest first

1. **Much of the module time has no cost model** (2421 ms)
   - 2421 of 7146 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
2. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (193 ms)
   - 199.2 ms per step against an ideal 6.0 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 4515 ms
   - lever: see the module table for the individual layers
3. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (181 ms)
   - 187.2 ms per step against an ideal 6.0 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 4515 ms
   - lever: see the module table for the individual layers
4. **Dtype and copy kernels take a large share** (168 ms)
   - 25% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
5. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (down)** (160 ms)
   - 164.0 ms per step against an ideal 4.0 ms (2% efficient, compute-bound), 24 layers; all rated pieces together leave 4515 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: linear attn1.to_v (up)** (155 ms)
   - 157.1 ms per step against an ideal 2.6 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 4515 ms
   - lever: see the module table for the individual layers
7. **Many small memory-bound kernels** (154 ms)
   - 46% of device time (16088 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
8. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (down)** (151 ms)
   - 155.2 ms per step against an ideal 4.0 ms (3% efficient, compute-bound), 24 layers; all rated pieces together leave 4515 ms
   - lever: see the module table for the individual layers
9. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5301 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up
10. **Module times do not add up to the device time** (0 ms)
   - modules account for 7146 ms of 667 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 242.5 | 36% |
| copy | 4610 | 168.3 | 25% |
| elementwise | 9051 | 116.8 | 18% |
| convolution | 512 | 61.9 | 9% |
| other | 610 | 45.3 | 7% |
| norm | 782 | 19.8 | 3% |
| attention | 140 | 9.0 | 1% |
| memory | 1627 | 2.1 | 0% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 104.6 TFLOPS, int8 199.4 TOPS, copy 386.2 GB/s, dispatch 10.5 us, queued launch 10.5 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention | transformer_blocks.attn1 | up | 36 | 199.2 | 6.0 | 3% | compute | 193.1 |
| attention | transformer_blocks.attn2 | up | 36 | 187.2 | 6.0 | 3% | compute | 181.2 |
| attention | transformer_blocks.attn1 | down | 24 | 164.0 | 4.0 | 2% | compute | 159.9 |
| linear | attn1.to_v | up | 36 | 157.1 | 2.6 | 2% | compute | 154.5 |
| attention | transformer_blocks.attn2 | down | 24 | 155.2 | 4.0 | 3% | compute | 151.2 |
| linear | attn1.to_q | up | 36 | 153.2 | 2.6 | 2% | compute | 150.5 |
| linear | attn2.to_out | up | 36 | 153.0 | 2.6 | 2% | compute | 150.4 |
| linear | attn2.to_v | up | 36 | 150.9 | 1.6 | 1% | memory | 149.3 |
| linear | attn1.to_out | up | 36 | 151.8 | 2.6 | 2% | compute | 149.2 |
| linear | attn2.to_q | up | 36 | 150.7 | 2.6 | 2% | compute | 148.0 |
| linear | attn2.to_k | up | 36 | 148.7 | 1.6 | 1% | memory | 147.1 |
| linear | attn1.to_k | up | 36 | 147.7 | 2.6 | 2% | compute | 145.0 |
| linear | net.proj | up | 36 | 165.7 | 21.2 | 13% | compute | 144.5 |
| linear | ff.net | up | 36 | 152.3 | 10.6 | 7% | compute | 141.7 |
| linear | attn1.to_q | down | 24 | 110.1 | 1.8 | 2% | compute | 108.3 |
| linear | attn2.to_out | down | 24 | 109.4 | 1.8 | 2% | compute | 107.7 |
| linear | attn1.to_out | down | 24 | 109.0 | 1.8 | 2% | compute | 107.3 |
| linear | attn2.to_q | down | 24 | 108.5 | 1.8 | 2% | compute | 106.7 |
| linear | attn2.to_k | down | 24 | 107.3 | 1.1 | 1% | memory | 106.2 |
| linear | attn1.to_k | down | 24 | 107.5 | 1.8 | 2% | compute | 105.8 |
| linear | attn2.to_v | down | 24 | 106.6 | 1.1 | 1% | memory | 105.6 |
| linear | attn1.to_v | down | 24 | 106.0 | 1.8 | 2% | compute | 104.3 |
| linear | net.proj | down | 24 | 117.5 | 14.1 | 12% | compute | 103.4 |
| linear | ff.net | down | 24 | 108.2 | 7.1 | 7% | compute | 101.1 |
| feedforward | transformer_blocks.ff | up | 36 | 112.5 | 20.9 | 19% | memory | 91.6 |
| feedforward | transformer_blocks.ff | down | 24 | 91.2 | 13.9 | 15% | memory | 77.3 |
| attention | transformer_blocks.attn1 | mid | 10 | 63.6 | 0.9 | 1% | memory | 62.7 |
| attention | transformer_blocks.attn2 | mid | 10 | 61.6 | 0.9 | 2% | memory | 60.7 |
| norm | transformer_blocks.norm1 | up | 36 | 48.5 | 1.7 | 4% | memory | 46.7 |
| norm | transformer_blocks.norm3 | up | 36 | 47.5 | 1.7 | 4% | memory | 45.7 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 16.5 | 0.0 | 70.3 | 40.1 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 15.8 | 0.0 | 70.1 | 39.7 | - | - |
| output_blocks.1.1.transformer_blocks.7 | block | 1 | 16.7 | 0.0 | 61.2 | 39.0 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 15.6 | 0.0 | 63.5 | 36.9 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 16.6 | 0.0 | 62.7 | 36.6 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 15.7 | 0.0 | 64.8 | 36.2 | - | - |
| middle_block.1.transformer_blocks.0 | block | 1 | 15.7 | 0.0 | 63.1 | 35.7 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 15.8 | 0.0 | 63.2 | 35.5 | - | - |
| middle_block.1.transformer_blocks.9 | block | 1 | 21.5 | 0.0 | 59.7 | 35.4 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 16.9 | 0.0 | 61.4 | 35.4 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 16.0 | 0.0 | 63.1 | 35.3 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 16.2 | 0.0 | 62.7 | 35.2 | - | - |
| middle_block.1.transformer_blocks.1 | block | 1 | 15.9 | 0.0 | 61.9 | 34.9 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 15.8 | 0.0 | 61.6 | 34.9 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 16.7 | 0.0 | 60.8 | 34.9 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 16.3 | 0.0 | 60.9 | 34.9 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 18.1 | 0.0 | 61.9 | 34.4 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 15.9 | 0.0 | 60.5 | 34.4 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 16.2 | 0.0 | 60.0 | 34.2 | - | - |
| middle_block.1.transformer_blocks.8 | block | 1 | 15.5 | 0.0 | 58.7 | 33.8 | - | - |
| middle_block.1.transformer_blocks.3 | block | 1 | 17.5 | 0.0 | 59.4 | 33.7 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 16.7 | 0.0 | 62.5 | 33.2 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 16.4 | 0.0 | 60.0 | 33.1 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 15.6 | 0.0 | 61.0 | 33.1 | - | - |
| output_blocks.0.1.transformer_blocks.1 | block | 1 | 16.9 | 0.0 | 57.6 | 33.0 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 17.3 | 0.0 | 58.1 | 32.5 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 15.6 | 0.0 | 55.9 | 32.3 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 16.1 | 0.0 | 57.2 | 32.0 | - | - |
| output_blocks.0.1.transformer_blocks.0 | block | 1 | 16.1 | 0.0 | 57.7 | 31.9 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 16.8 | 0.0 | 57.0 | 31.6 | - | - |

## Reconciliation

device 667 ms; modules' own time 7146 ms, of which 2421 ms is in modules with no cost model; wall 1030 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 234.84 | 30 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 77.01 | 36 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2141 | 68.66 | 32 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 56.96 | 19 |
| `gen_conv` | convolution | 131 | 47.02 | 359 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 40.07 | 286 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 17.58 | 84 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 15.05 | 215 |
| `conv_reorder` | convolution | 241 | 13.30 | 55 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.79 | 23 |
| `micro_sdpa` | attention | 140 | 9.00 | 64 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 7.63 | 54 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 7.18 | 51 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.61 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.28 | 90 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.94 | 85 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.57 | 22 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.26 | 2 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 3.13 | 42 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 2.88 | 87 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.74 | 37 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 2.52 | 18 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 210 | 2.38 | 11 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 39 | 2.25 | 58 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::SiluFunctor<c10::BFloat16` | elementwise | 96 | 2.14 | 22 |
