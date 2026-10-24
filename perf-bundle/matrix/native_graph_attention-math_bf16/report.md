# Performance report (xpu)

Window: 3 steps (2 eager, 1 replayed). All times are per step.

## Step budget

- step 965 ms, median of 3 unprofiled steps (best 963); eager 964 ms, replayed 5071 ms
- the profiled window's steps took 6014 ms each (hooks and profiler); device and module times come from them
- device busy 708 ms, utilisation 73%
- 26974 kernel launches, 197852 operators, 161 us host time per launch
- step peak memory 6305 MB
- phases: loss 0 ms

## Findings, largest first

1. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (175 ms)
   - 184.3 ms per step against an ideal 9.2 ms (5% efficient, compute-bound), 36 layers; all rated pieces together leave 4052 ms
   - lever: see the module table for the individual layers
2. **Dtype and copy kernels take a large share** (166 ms)
   - 23% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
3. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (160 ms)
   - 169.7 ms per step against an ideal 9.2 ms (5% efficient, compute-bound), 36 layers; all rated pieces together leave 4052 ms
   - lever: see the module table for the individual layers
4. **Many small memory-bound kernels** (152 ms)
   - 43% of device time (17369 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
5. **Slowest pieces by recoverable time: linear attn1.to_q (up)** (139 ms)
   - 142.9 ms per step against an ideal 3.5 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 4052 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: linear attn2.to_out (up)** (139 ms)
   - 142.5 ms per step against an ideal 3.5 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 4052 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (down)** (138 ms)
   - 144.6 ms per step against an ideal 6.1 ms (4% efficient, compute-bound), 24 layers; all rated pieces together leave 4052 ms
   - lever: see the module table for the individual layers
8. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5071 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8390 | 284.5 | 40% |
| copy | 5175 | 165.6 | 23% |
| elementwise | 9331 | 116.5 | 16% |
| convolution | 466 | 61.0 | 9% |
| other | 516 | 33.9 | 5% |
| attention | 233 | 24.1 | 3% |
| norm | 782 | 18.9 | 3% |
| memory | 2063 | 2.3 | 0% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 104.6 TFLOPS, int8 199.4 TOPS, copy 386.2 GB/s, dispatch 10.5 us, queued launch 10.5 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention | transformer_blocks.attn1 | up | 36 | 184.3 | 9.2 | 5% | compute | 175.1 |
| attention | transformer_blocks.attn2 | up | 36 | 169.7 | 9.2 | 5% | compute | 160.5 |
| linear | attn1.to_q | up | 36 | 142.9 | 3.5 | 2% | compute | 139.4 |
| linear | attn2.to_out | up | 36 | 142.5 | 3.5 | 2% | compute | 139.0 |
| attention | transformer_blocks.attn1 | down | 24 | 144.6 | 6.1 | 4% | compute | 138.5 |
| linear | attn2.to_q | up | 36 | 141.4 | 3.5 | 2% | compute | 138.0 |
| linear | attn2.to_k | up | 36 | 139.5 | 1.6 | 1% | memory | 137.9 |
| linear | attn2.to_v | up | 36 | 138.9 | 1.6 | 1% | memory | 137.3 |
| linear | attn1.to_out | up | 36 | 140.4 | 3.5 | 2% | compute | 136.9 |
| linear | attn1.to_k | up | 36 | 138.6 | 3.5 | 3% | compute | 135.1 |
| linear | net.proj | up | 36 | 158.6 | 27.7 | 17% | compute | 130.9 |
| linear | attn1.to_v | up | 36 | 133.4 | 3.5 | 3% | compute | 130.0 |
| linear | ff.net | up | 36 | 141.7 | 13.9 | 10% | compute | 127.9 |
| attention | transformer_blocks.attn2 | down | 24 | 133.9 | 6.1 | 5% | compute | 127.7 |
| linear | attn1.to_k | down | 24 | 109.2 | 2.3 | 2% | compute | 106.9 |
| linear | attn2.to_q | down | 24 | 101.0 | 2.3 | 2% | compute | 98.7 |
| linear | attn2.to_out | down | 24 | 100.2 | 2.3 | 2% | compute | 97.9 |
| linear | attn1.to_q | down | 24 | 100.0 | 2.3 | 2% | compute | 97.7 |
| linear | attn1.to_out | down | 24 | 99.3 | 2.3 | 2% | compute | 97.0 |
| linear | attn2.to_k | down | 24 | 98.0 | 1.1 | 1% | memory | 96.9 |
| linear | attn2.to_v | down | 24 | 97.5 | 1.1 | 1% | memory | 96.4 |
| linear | ff.net | down | 24 | 100.4 | 9.2 | 9% | compute | 91.1 |
| linear | attn1.to_v | down | 24 | 92.9 | 2.3 | 2% | compute | 90.6 |
| linear | net.proj | down | 24 | 108.6 | 18.5 | 17% | compute | 90.1 |
| feedforward | transformer_blocks.ff | up | 36 | 100.8 | 27.4 | 27% | memory | 73.4 |
| feedforward | transformer_blocks.ff | down | 24 | 79.7 | 18.2 | 23% | memory | 61.5 |
| linear | attn2.to_out | mid | 10 | 53.7 | 1.0 | 2% | compute | 52.7 |
| attention | transformer_blocks.attn1 | mid | 10 | 52.4 | 1.2 | 2% | memory | 51.2 |
| attention | transformer_blocks.attn2 | mid | 10 | 51.5 | 1.2 | 2% | memory | 50.3 |
| linear | attn1.to_v | mid | 10 | 44.7 | 1.0 | 2% | compute | 43.8 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 15.6 | 0.0 | 68.3 | 43.4 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 15.2 | 0.0 | 56.8 | 32.8 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 15.2 | 0.0 | 59.1 | 32.1 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 15.2 | 0.0 | 56.6 | 31.9 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 15.1 | 0.0 | 56.8 | 31.9 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 15.1 | 0.0 | 55.6 | 31.5 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 15.1 | 0.0 | 54.4 | 31.0 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 15.0 | 0.0 | 55.0 | 31.0 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 15.0 | 0.0 | 54.8 | 30.8 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 14.9 | 0.0 | 55.2 | 30.7 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 15.1 | 0.0 | 54.0 | 30.7 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 15.2 | 0.0 | 53.6 | 30.6 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 15.0 | 0.0 | 54.6 | 30.5 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 15.1 | 0.0 | 54.3 | 30.4 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 15.2 | 0.0 | 54.2 | 30.4 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 15.1 | 0.0 | 53.9 | 30.3 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 15.0 | 0.0 | 53.5 | 30.2 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 15.3 | 0.0 | 52.8 | 29.9 | - | - |
| middle_block.1.transformer_blocks.0 | block | 1 | 15.6 | 0.0 | 52.5 | 29.8 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 15.1 | 0.0 | 53.0 | 29.7 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 15.2 | 0.0 | 53.4 | 29.5 | - | - |
| middle_block.1.transformer_blocks.7 | block | 1 | 16.0 | 0.0 | 51.0 | 29.5 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 15.2 | 0.0 | 53.2 | 29.3 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 15.2 | 0.0 | 51.9 | 29.2 | - | - |
| middle_block.1.transformer_blocks.3 | block | 1 | 21.0 | 0.0 | 51.4 | 29.1 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 15.2 | 0.0 | 51.7 | 29.0 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 16.8 | 0.0 | 52.0 | 29.0 | - | - |
| middle_block.1.transformer_blocks.1 | block | 1 | 15.5 | 0.0 | 65.1 | 28.8 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 15.5 | 0.0 | 50.9 | 28.6 | - | - |
| middle_block.1.transformer_blocks.9 | block | 1 | 15.6 | 0.0 | 50.2 | 28.6 | - | - |

## Reconciliation

device 708 ms; modules - ms (unrated - ms). Module times need eager steps: a window with replayed steps has no module table.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 8296 | 279.25 | 34 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 73.68 | 34 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2141 | 66.62 | 31 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 55.60 | 18 |
| `gen_conv` | convolution | 131 | 46.20 | 353 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 93 | 29.46 | 316 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 17.52 | 83 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 14.39 | 206 |
| `conv_reorder` | convolution | 241 | 13.65 | 57 |
| `at::native::xpu::impl::DispatchSoftmaxForwardKernelFunctor<16, 1, 32, c10::BFloat16, c10::` | attention | 47 | 9.83 | 211 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.25 | 22 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 6.90 | 49 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.32 | 2 |
| `micro_sdpa` | attention | 93 | 6.31 | 68 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.24 | 89 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.00 | 86 |
| `at::native::xpu::impl::DispatchSoftmaxBackwardKernelFunctor<8, 1, 32, c10::BFloat16, c10::` | attention | 23 | 5.97 | 256 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 93 | 5.27 | 56 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.34 | 21 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::CopyScalarFunc<c10::BFloa` | copy | 520 | 3.60 | 7 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.30 | 2 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 3.07 | 41 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.71 | 37 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | copy | 183 | 2.55 | 14 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 39 | 2.24 | 57 |
