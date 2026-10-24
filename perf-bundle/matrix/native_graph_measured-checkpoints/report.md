# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- step 1672 ms, median of 3 unprofiled steps (best 1668); eager 1672 ms, replayed - ms
- the profiled window's steps took 9601 ms each (hooks and profiler); device and module times come from them
- device busy 641 ms, utilisation 38%
- 42155 kernel launches, 322340 operators, 211 us host time per launch
- step peak memory 7604 MB
- phases: loss 0 ms

## Findings, largest first

1. **Much of the module time has no cost model** (2745 ms)
   - 2745 of 9996 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
2. **The step is launch bound** (1031 ms)
   - the device is busy 38% of the step (641 of 1672 ms) over 42155 launches and 322340 operators per step, 211 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
3. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (330 ms)
   - 335.9 ms per step against an ideal 5.9 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 7054 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (293 ms)
   - 298.5 ms per step against an ideal 5.9 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 7054 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: linear attn1.to_k (up)** (265 ms)
   - 267.5 ms per step against an ideal 2.5 ms (1% efficient, memory-bound), 36 layers; all rated pieces together leave 7054 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: linear attn2.to_out (up)** (249 ms)
   - 251.6 ms per step against an ideal 2.5 ms (1% efficient, memory-bound), 36 layers; all rated pieces together leave 7054 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: linear attn1.to_out (up)** (249 ms)
   - 251.6 ms per step against an ideal 2.5 ms (1% efficient, memory-bound), 36 layers; all rated pieces together leave 7054 ms
   - lever: see the module table for the individual layers
8. **Dtype and copy kernels take a large share** (143 ms)
   - 22% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
9. **Many small memory-bound kernels** (132 ms)
   - 41% of device time (25772 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
10. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5762 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up
11. **Module times do not add up to the device time** (0 ms)
   - modules account for 9996 ms of 641 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 14122 | 264.4 | 41% |
| copy | 7852 | 143.1 | 22% |
| elementwise | 14737 | 98.3 | 15% |
| convolution | 910 | 61.3 | 10% |
| other | 1187 | 44.8 | 7% |
| norm | 1358 | 19.2 | 3% |
| attention | 164 | 6.6 | 1% |
| memory | 1806 | 2.1 | 0% |
| reduce | 19 | 1.4 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 97.7 TFLOPS, int8 200.6 TOPS, copy 396.5 GB/s, dispatch 10.3 us, queued launch 10.3 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention | transformer_blocks.attn2 | up | 36 | 335.9 | 5.9 | 2% | compute | 330.0 |
| attention | transformer_blocks.attn1 | up | 36 | 298.5 | 5.9 | 2% | compute | 292.6 |
| linear | attn1.to_k | up | 36 | 267.5 | 2.5 | 1% | memory | 265.0 |
| linear | attn2.to_out | up | 36 | 251.6 | 2.5 | 1% | memory | 249.1 |
| linear | attn1.to_out | up | 36 | 251.6 | 2.5 | 1% | memory | 249.1 |
| linear | attn1.to_q | up | 36 | 250.7 | 2.5 | 1% | memory | 248.2 |
| linear | attn2.to_q | up | 36 | 248.7 | 2.5 | 1% | memory | 246.2 |
| linear | ff.net | up | 36 | 253.6 | 9.9 | 4% | compute | 243.7 |
| attention | transformer_blocks.attn2 | down | 24 | 246.6 | 3.9 | 2% | compute | 242.6 |
| linear | net.proj | up | 36 | 261.6 | 19.7 | 8% | compute | 241.9 |
| linear | attn1.to_v | up | 36 | 240.8 | 2.5 | 1% | memory | 238.3 |
| attention | transformer_blocks.attn1 | down | 24 | 222.3 | 3.9 | 2% | compute | 218.3 |
| linear | attn2.to_v | up | 36 | 209.5 | 2.5 | 1% | memory | 207.0 |
| linear | attn2.to_k | up | 36 | 201.8 | 2.5 | 1% | memory | 199.3 |
| linear | attn1.to_out | down | 24 | 183.3 | 1.7 | 1% | memory | 181.6 |
| linear | attn2.to_out | down | 24 | 176.4 | 1.7 | 1% | memory | 174.8 |
| linear | attn1.to_q | down | 24 | 174.3 | 1.7 | 1% | memory | 172.6 |
| linear | attn2.to_q | down | 24 | 172.6 | 1.7 | 1% | memory | 170.9 |
| linear | attn1.to_k | down | 24 | 171.8 | 1.7 | 1% | memory | 170.2 |
| linear | ff.net | down | 24 | 176.3 | 6.6 | 4% | compute | 169.7 |
| linear | attn1.to_v | down | 24 | 167.1 | 1.7 | 1% | memory | 165.5 |
| linear | net.proj | down | 24 | 177.9 | 13.2 | 7% | compute | 164.7 |
| feedforward | transformer_blocks.ff | up | 36 | 177.8 | 18.7 | 10% | memory | 159.2 |
| linear | attn2.to_k | down | 24 | 144.8 | 1.7 | 1% | memory | 143.1 |
| linear | attn2.to_v | down | 24 | 143.7 | 1.7 | 1% | memory | 142.0 |
| feedforward | transformer_blocks.ff | down | 24 | 131.8 | 12.5 | 9% | memory | 119.3 |
| attention | transformer_blocks.attn2 | mid | 10 | 98.3 | 0.8 | 1% | memory | 97.5 |
| attention | transformer_blocks.attn1 | mid | 10 | 78.1 | 0.8 | 1% | memory | 77.3 |
| norm | transformer_blocks.norm1 | up | 36 | 77.2 | 1.6 | 2% | memory | 75.7 |
| norm | transformer_blocks.norm3 | up | 36 | 75.5 | 1.6 | 2% | memory | 73.9 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.4.1.transformer_blocks.0 | block | 2 | 30.0 | 0.0 | 107.6 | 60.7 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 2 | 30.4 | 0.0 | 105.8 | 59.4 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 2 | 30.4 | 0.0 | 108.6 | 59.3 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 2 | 30.9 | 0.0 | 105.4 | 59.2 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 2 | 30.1 | 0.0 | 102.7 | 57.4 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 2 | 30.3 | 0.0 | 102.3 | 57.3 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 2 | 29.9 | 0.0 | 101.5 | 57.0 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 2 | 31.9 | 0.0 | 101.5 | 56.8 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 2 | 30.0 | 0.0 | 100.8 | 56.5 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 2 | 30.0 | 0.0 | 100.3 | 56.3 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 2 | 30.4 | 0.0 | 99.5 | 56.1 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 2 | 30.6 | 0.0 | 99.4 | 55.8 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 2 | 33.8 | 0.0 | 98.5 | 55.5 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 2 | 30.1 | 0.0 | 99.3 | 55.5 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 2 | 30.1 | 0.0 | 98.3 | 55.3 | - | - |
| middle_block.1.transformer_blocks.0 | block | 2 | 30.6 | 0.0 | 97.1 | 54.5 | - | - |
| middle_block.1.transformer_blocks.4 | block | 2 | 30.8 | 0.0 | 96.2 | 54.1 | - | - |
| middle_block.1.transformer_blocks.6 | block | 2 | 30.7 | 0.0 | 95.5 | 53.6 | - | - |
| middle_block.1.transformer_blocks.8 | block | 2 | 30.6 | 0.0 | 95.2 | 53.5 | - | - |
| output_blocks.0.1.transformer_blocks.1 | block | 2 | 31.4 | 0.0 | 93.9 | 53.1 | - | - |
| output_blocks.0.1.transformer_blocks.0 | block | 2 | 37.1 | 0.0 | 96.1 | 53.0 | - | - |
| output_blocks.0.1.transformer_blocks.5 | block | 2 | 31.1 | 0.0 | 93.6 | 53.0 | - | - |
| output_blocks.0.1.transformer_blocks.4 | block | 2 | 31.2 | 0.0 | 93.9 | 52.8 | - | - |
| output_blocks.0.1.transformer_blocks.6 | block | 2 | 31.0 | 0.0 | 93.1 | 52.6 | - | - |
| output_blocks.0.1.transformer_blocks.3 | block | 2 | 31.0 | 0.0 | 93.7 | 52.5 | - | - |
| output_blocks.1.1.transformer_blocks.7 | block | 2 | 32.4 | 0.0 | 92.7 | 52.4 | - | - |
| output_blocks.0.1.transformer_blocks.7 | block | 2 | 31.3 | 0.0 | 92.5 | 52.0 | - | - |
| output_blocks.1.1.transformer_blocks.5 | block | 2 | 32.2 | 0.0 | 91.4 | 51.5 | - | - |
| output_blocks.1.1.transformer_blocks.6 | block | 2 | 32.2 | 0.0 | 91.0 | 51.2 | - | - |
| output_blocks.2.1.transformer_blocks.0 | block | 2 | 32.5 | 0.0 | 90.1 | 50.9 | - | - |

## Reconciliation

device 641 ms; modules' own time 9996 ms, of which 2745 ms is in modules with no cost model; wall 1672 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 13842 | 254.96 | 18 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 3598 | 64.02 | 18 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 3710 | 60.73 | 16 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 5166 | 44.67 | 9 |
| `gen_conv` | convolution | 222 | 42.70 | 192 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 280 | 39.01 | 139 |
| `conv_reorder` | convolution | 408 | 17.66 | 43 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 140 | 13.22 | 94 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 362 | 10.29 | 28 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 666 | 8.71 | 13 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 5227 | 8.47 | 2 |
| `micro_sdpa` | attention | 164 | 6.62 | 40 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 140 | 6.26 | 45 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 200 | 6.15 | 31 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 420 | 5.48 | 13 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 222 | 5.36 | 24 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 2912 | 5.03 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 140 | 4.64 | 33 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 116 | 4.10 | 35 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 80 | 3.30 | 41 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 420 | 2.81 | 7 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 116 | 2.70 | 23 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 280 | 2.43 | 9 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 2.28 | 69 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 78 | 2.24 | 29 |
