# Performance report (xpu)

Window: 3 steps (2 eager, 1 replayed). All times are per step.

## Step budget

- step 957 ms, median of 3 unprofiled steps (best 956); eager 957 ms, replayed 5074 ms
- the profiled window's steps took 6119 ms each (hooks and profiler); device and module times come from them
- device busy 784 ms, utilisation 82%
- 28214 kernel launches, 202183 operators, 150 us host time per launch
- step peak memory 6475 MB
- phases: loss 0 ms

## Findings, largest first

1. **Dtype and copy kernels take a large share** (219 ms)
   - 28% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
2. **Many small memory-bound kernels** (189 ms)
   - 48% of device time (18523 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
3. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (182 ms)
   - 190.8 ms per step against an ideal 9.2 ms (5% efficient, compute-bound), 36 layers; all rated pieces together leave 4009 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (168 ms)
   - 177.0 ms per step against an ideal 9.2 ms (5% efficient, compute-bound), 36 layers; all rated pieces together leave 4009 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: linear attn2.to_out (up)** (149 ms)
   - 152.7 ms per step against an ideal 3.5 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 4009 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (down)** (138 ms)
   - 143.7 ms per step against an ideal 6.1 ms (4% efficient, compute-bound), 24 layers; all rated pieces together leave 4009 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: linear attn1.to_q (up)** (137 ms)
   - 140.6 ms per step against an ideal 3.5 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 4009 ms
   - lever: see the module table for the individual layers
8. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5074 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8466 | 283.1 | 36% |
| copy | 6155 | 218.7 | 28% |
| elementwise | 9455 | 134.2 | 17% |
| convolution | 466 | 60.7 | 8% |
| other | 516 | 33.7 | 4% |
| attention | 243 | 29.4 | 4% |
| norm | 782 | 18.9 | 2% |
| reduce | 68 | 3.4 | 0% |
| memory | 2063 | 2.2 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 104.6 TFLOPS, int8 199.4 TOPS, copy 386.2 GB/s, dispatch 10.5 us, queued launch 10.5 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention | transformer_blocks.attn1 | up | 36 | 190.8 | 9.2 | 5% | compute | 181.6 |
| attention | transformer_blocks.attn2 | up | 36 | 177.0 | 9.2 | 5% | compute | 167.7 |
| linear | attn2.to_out | up | 36 | 152.7 | 3.5 | 2% | compute | 149.2 |
| attention | transformer_blocks.attn1 | down | 24 | 143.7 | 6.1 | 4% | compute | 137.6 |
| linear | attn1.to_q | up | 36 | 140.6 | 3.5 | 2% | compute | 137.1 |
| linear | attn2.to_v | up | 36 | 137.2 | 1.6 | 1% | memory | 135.5 |
| linear | attn2.to_q | up | 36 | 138.7 | 3.5 | 2% | compute | 135.3 |
| linear | attn1.to_out | up | 36 | 138.4 | 3.5 | 3% | compute | 134.9 |
| linear | attn2.to_k | up | 36 | 136.4 | 1.6 | 1% | memory | 134.8 |
| linear | attn1.to_k | up | 36 | 135.6 | 3.5 | 3% | compute | 132.1 |
| attention | transformer_blocks.attn2 | down | 24 | 135.7 | 6.1 | 5% | compute | 129.5 |
| linear | net.proj | up | 36 | 156.4 | 27.7 | 18% | compute | 128.7 |
| linear | ff.net | up | 36 | 140.9 | 13.9 | 10% | compute | 127.0 |
| linear | attn1.to_v | up | 36 | 129.7 | 3.5 | 3% | compute | 126.2 |
| linear | attn1.to_k | down | 24 | 106.0 | 2.3 | 2% | compute | 103.7 |
| linear | attn2.to_out | down | 24 | 98.0 | 2.3 | 2% | compute | 95.7 |
| linear | attn1.to_q | down | 24 | 98.0 | 2.3 | 2% | compute | 95.6 |
| linear | attn2.to_q | down | 24 | 97.3 | 2.3 | 2% | compute | 95.0 |
| linear | attn2.to_v | down | 24 | 95.7 | 1.1 | 1% | memory | 94.6 |
| linear | attn1.to_out | down | 24 | 96.9 | 2.3 | 2% | compute | 94.6 |
| linear | attn2.to_k | down | 24 | 95.3 | 1.1 | 1% | memory | 94.3 |
| linear | ff.net | down | 24 | 98.3 | 9.2 | 9% | compute | 89.0 |
| linear | attn1.to_v | down | 24 | 90.5 | 2.3 | 3% | compute | 88.2 |
| linear | net.proj | down | 24 | 106.4 | 18.5 | 17% | compute | 87.9 |
| feedforward | transformer_blocks.ff | up | 36 | 105.4 | 27.4 | 26% | memory | 78.0 |
| feedforward | transformer_blocks.ff | down | 24 | 79.8 | 18.2 | 23% | memory | 61.5 |
| attention | transformer_blocks.attn1 | mid | 10 | 53.1 | 1.2 | 2% | memory | 51.8 |
| attention | transformer_blocks.attn2 | mid | 10 | 50.4 | 1.2 | 2% | memory | 49.2 |
| norm | transformer_blocks.norm1 | up | 36 | 47.1 | 2.3 | 5% | memory | 44.9 |
| norm | transformer_blocks.norm3 | up | 36 | 46.2 | 2.3 | 5% | memory | 43.9 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| output_blocks.0.1.transformer_blocks.9 | block | 1 | 16.4 | 0.0 | 60.8 | 39.6 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 15.5 | 0.0 | 55.7 | 31.6 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 16.4 | 0.0 | 55.3 | 31.3 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 16.6 | 0.0 | 55.6 | 31.3 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 15.6 | 0.0 | 55.7 | 31.3 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 15.3 | 0.0 | 54.5 | 30.5 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 15.1 | 0.0 | 54.2 | 30.4 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 15.1 | 0.0 | 54.2 | 30.3 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 15.2 | 0.0 | 53.9 | 30.2 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 15.4 | 0.0 | 53.9 | 30.1 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 15.0 | 0.0 | 53.2 | 29.9 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 15.1 | 0.0 | 53.6 | 29.9 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 15.2 | 0.0 | 53.2 | 29.8 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 15.2 | 0.0 | 53.0 | 29.7 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 15.2 | 0.0 | 54.3 | 29.7 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 15.2 | 0.0 | 52.8 | 29.7 | - | - |
| output_blocks.1.1.transformer_blocks.4 | block | 1 | 16.2 | 0.0 | 50.4 | 29.4 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 15.1 | 0.0 | 53.1 | 29.4 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 15.3 | 0.0 | 51.4 | 28.8 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 15.4 | 0.0 | 51.4 | 28.7 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 15.3 | 0.0 | 51.5 | 28.6 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 15.3 | 0.0 | 50.8 | 28.5 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 15.3 | 0.0 | 51.3 | 28.5 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 15.4 | 0.0 | 50.6 | 28.3 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 15.4 | 0.0 | 60.4 | 28.3 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 15.4 | 0.0 | 50.1 | 28.0 | - | - |
| middle_block.1.transformer_blocks.0 | block | 1 | 15.9 | 0.0 | 50.0 | 28.0 | - | - |
| middle_block.1.transformer_blocks.1 | block | 1 | 15.5 | 0.0 | 49.9 | 28.0 | - | - |
| middle_block.1.transformer_blocks.3 | block | 1 | 15.8 | 0.0 | 49.5 | 27.9 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 15.4 | 0.0 | 49.6 | 27.8 | - | - |

## Reconciliation

device 784 ms; modules - ms (unrated - ms). Module times need eager steps: a window with replayed steps has no module table.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 8373 | 277.87 | 33 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2440 | 97.24 | 40 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2528 | 92.17 | 36 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 55.38 | 18 |
| `gen_conv` | convolution | 131 | 45.94 | 351 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 93 | 29.29 | 314 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 17.50 | 83 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 14.39 | 206 |
| `conv_reorder` | convolution | 241 | 13.65 | 57 |
| `at::native::xpu::impl::DispatchSoftmaxForwardKernelFunctor<8, 4, 32, float, float, float, ` | attention | 20 | 10.92 | 546 |
| `at::native::xpu::impl::DispatchSoftmaxForwardKernelFunctor<8, 1, 32, float, float, float, ` | attention | 60 | 9.95 | 166 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 53 | 9.52 | 180 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.25 | 22 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 6.88 | 49 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3704 | 6.83 | 2 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<f` | elementwise | 50 | 6.76 | 135 |
| `micro_sdpa` | attention | 93 | 6.30 | 67 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::CopyScalarFunc<c10::BFloa` | copy | 784 | 6.26 | 8 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.24 | 89 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.01 | 86 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 93 | 5.27 | 57 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.31 | 21 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | copy | 243 | 3.65 | 15 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1473 | 3.53 | 2 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 3.07 | 41 |
