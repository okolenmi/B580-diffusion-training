# Performance report (xpu)

Window: 3 steps (2 eager, 1 replayed). All times are per step.

## Step budget

- step 1035 ms, median of 3 unprofiled steps (best 1023); eager 1029 ms, replayed 5272 ms
- the profiled window's steps took 6284 ms each (hooks and profiler); device and module times come from them
- device busy 743 ms, utilisation 72%
- 27628 kernel launches, 200652 operators, 163 us host time per launch
- step peak memory 6528 MB
- phases: loss 0 ms

## Findings, largest first

1. **Dtype and copy kernels take a large share** (191 ms)
   - 26% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
2. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (188 ms)
   - 197.5 ms per step against an ideal 9.2 ms (5% efficient, compute-bound), 36 layers; all rated pieces together leave 4397 ms
   - lever: see the module table for the individual layers
3. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (174 ms)
   - 183.5 ms per step against an ideal 9.2 ms (5% efficient, compute-bound), 36 layers; all rated pieces together leave 4397 ms
   - lever: see the module table for the individual layers
4. **Many small memory-bound kernels** (167 ms)
   - 45% of device time (18023 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
5. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (down)** (165 ms)
   - 170.9 ms per step against an ideal 6.1 ms (4% efficient, compute-bound), 24 layers; all rated pieces together leave 4397 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (down)** (161 ms)
   - 166.7 ms per step against an ideal 6.1 ms (4% efficient, compute-bound), 24 layers; all rated pieces together leave 4397 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: linear attn2.to_q (up)** (159 ms)
   - 162.1 ms per step against an ideal 3.5 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 4397 ms
   - lever: see the module table for the individual layers
8. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5272 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8390 | 281.7 | 38% |
| copy | 5829 | 191.0 | 26% |
| elementwise | 9331 | 119.3 | 16% |
| convolution | 466 | 61.0 | 8% |
| other | 516 | 33.9 | 5% |
| attention | 233 | 32.7 | 4% |
| norm | 782 | 19.6 | 3% |
| memory | 2063 | 2.2 | 0% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 104.6 TFLOPS, int8 199.4 TOPS, copy 386.2 GB/s, dispatch 10.5 us, queued launch 10.5 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention | transformer_blocks.attn1 | up | 36 | 197.5 | 9.2 | 5% | compute | 188.3 |
| attention | transformer_blocks.attn2 | up | 36 | 183.5 | 9.2 | 5% | compute | 174.3 |
| attention | transformer_blocks.attn1 | down | 24 | 170.9 | 6.1 | 4% | compute | 164.7 |
| attention | transformer_blocks.attn2 | down | 24 | 166.7 | 6.1 | 4% | compute | 160.6 |
| linear | attn2.to_q | up | 36 | 162.1 | 3.5 | 2% | compute | 158.6 |
| linear | attn1.to_q | up | 36 | 151.5 | 3.5 | 2% | compute | 148.1 |
| linear | attn2.to_out | up | 36 | 148.4 | 3.5 | 2% | compute | 144.9 |
| linear | attn2.to_v | up | 36 | 144.9 | 1.6 | 1% | memory | 143.3 |
| linear | attn2.to_k | up | 36 | 144.5 | 1.6 | 1% | memory | 142.9 |
| linear | attn1.to_out | up | 36 | 146.3 | 3.5 | 2% | compute | 142.8 |
| linear | attn1.to_k | up | 36 | 144.0 | 3.5 | 2% | compute | 140.5 |
| linear | net.proj | up | 36 | 164.5 | 27.7 | 17% | compute | 136.8 |
| linear | ff.net | up | 36 | 149.2 | 13.9 | 9% | compute | 135.3 |
| linear | attn1.to_v | up | 36 | 137.8 | 3.5 | 3% | compute | 134.4 |
| linear | attn2.to_out | down | 24 | 110.3 | 2.3 | 2% | compute | 108.0 |
| linear | attn2.to_k | down | 24 | 109.0 | 1.1 | 1% | memory | 107.9 |
| linear | attn1.to_out | down | 24 | 109.9 | 2.3 | 2% | compute | 107.6 |
| linear | attn2.to_q | down | 24 | 109.4 | 2.3 | 2% | compute | 107.1 |
| linear | attn1.to_q | down | 24 | 109.1 | 2.3 | 2% | compute | 106.8 |
| linear | attn2.to_v | down | 24 | 107.1 | 1.1 | 1% | memory | 106.0 |
| linear | attn1.to_k | down | 24 | 107.1 | 2.3 | 2% | compute | 104.8 |
| linear | ff.net | down | 24 | 110.2 | 9.2 | 8% | compute | 101.0 |
| linear | net.proj | down | 24 | 118.5 | 18.5 | 16% | compute | 100.0 |
| linear | attn1.to_v | down | 24 | 101.7 | 2.3 | 2% | compute | 99.4 |
| feedforward | transformer_blocks.ff | up | 36 | 110.7 | 27.4 | 25% | memory | 83.4 |
| feedforward | transformer_blocks.ff | down | 24 | 95.3 | 18.2 | 19% | memory | 77.1 |
| attention | transformer_blocks.attn1 | mid | 10 | 58.4 | 1.2 | 2% | memory | 57.2 |
| attention | transformer_blocks.attn2 | mid | 10 | 57.5 | 1.2 | 2% | memory | 56.2 |
| norm | transformer_blocks.norm1 | down | 24 | 49.0 | 1.5 | 3% | memory | 47.5 |
| norm | transformer_blocks.norm1 | up | 36 | 47.9 | 2.3 | 5% | memory | 45.7 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 16.7 | 0.0 | 72.7 | 40.5 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 16.6 | 0.0 | 69.9 | 38.9 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 16.3 | 0.0 | 65.5 | 37.7 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 16.5 | 0.0 | 67.5 | 37.4 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 15.8 | 0.0 | 65.2 | 37.1 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 15.9 | 0.0 | 71.5 | 36.6 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 21.6 | 0.0 | 65.3 | 36.4 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 16.4 | 0.0 | 63.1 | 35.5 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 16.5 | 0.0 | 62.9 | 35.4 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 16.5 | 0.0 | 64.2 | 35.0 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 16.0 | 0.0 | 71.0 | 34.7 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 15.7 | 0.0 | 60.6 | 34.7 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 16.3 | 0.0 | 60.4 | 34.2 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 16.6 | 0.0 | 61.1 | 34.0 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 16.0 | 0.0 | 60.6 | 33.8 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 16.6 | 0.0 | 58.1 | 32.9 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 16.2 | 0.0 | 57.8 | 32.8 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 16.9 | 0.0 | 58.0 | 32.7 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 16.9 | 0.0 | 57.2 | 32.4 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 16.6 | 0.0 | 56.9 | 32.0 | - | - |
| middle_block.1.transformer_blocks.0 | block | 1 | 16.4 | 0.0 | 56.1 | 31.9 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 16.8 | 0.0 | 56.3 | 31.9 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 16.1 | 0.0 | 57.2 | 31.7 | - | - |
| middle_block.1.transformer_blocks.1 | block | 1 | 16.0 | 0.0 | 56.8 | 31.7 | - | - |
| middle_block.1.transformer_blocks.7 | block | 1 | 16.8 | 0.0 | 55.8 | 31.6 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 16.7 | 0.0 | 56.4 | 31.6 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 16.1 | 0.0 | 56.9 | 31.4 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 15.8 | 0.0 | 56.7 | 31.3 | - | - |
| middle_block.1.transformer_blocks.3 | block | 1 | 16.2 | 0.0 | 55.5 | 31.3 | - | - |
| middle_block.1.transformer_blocks.4 | block | 1 | 15.9 | 0.0 | 54.8 | 31.2 | - | - |

## Reconciliation

device 743 ms; modules - ms (unrated - ms). Module times need eager steps: a window with replayed steps has no module table.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 8296 | 276.42 | 33 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2474 | 89.91 | 36 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2468 | 73.27 | 30 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 55.88 | 18 |
| `gen_conv` | convolution | 131 | 46.67 | 356 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 93 | 29.55 | 317 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 17.80 | 85 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 14.65 | 209 |
| `conv_reorder` | convolution | 241 | 13.17 | 55 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3937 | 9.71 | 2 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.44 | 22 |
| `at::native::xpu::impl::DispatchSoftmaxForwardKernelFunctor<8, 4, 32, float, float, float, ` | attention | 7 | 7.40 | 1110 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 7.28 | 52 |
| `at::native::xpu::impl::DispatchSoftmaxForwardKernelFunctor<8, 1, 32, float, float, float, ` | attention | 40 | 6.78 | 170 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.39 | 91 |
| `micro_sdpa` | attention | 93 | 6.39 | 68 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.99 | 86 |
| `at::native::xpu::impl::DispatchSoftmaxBackwardKernelFunctor<4, 1, 32, float, float, float,` | attention | 43 | 5.40 | 125 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 93 | 5.28 | 57 |
| `at::native::xpu::impl::DispatchSoftmaxBackwardKernelFunctor<4, 4, 32, float, float, float,` | attention | 3 | 5.20 | 1559 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.66 | 22 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.25 | 2 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 3.20 | 43 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::CopyScalarFunc<float>, at` | copy | 234 | 2.81 | 12 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.77 | 37 |
