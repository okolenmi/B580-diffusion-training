# Performance report (xpu)

Window: 3 steps (2 eager, 1 replayed). All times are per step.

## Step budget

- wall 9636 ms (best 8165); eager 8170 ms, replayed 12567 ms
- device busy 620 ms, utilisation 6%
- 25434 kernel launches, 541647 operators, 373 us host time per launch
- step peak memory 6126 MB
- phases: loss 0 ms

## Findings, largest first

1. **The step is launch bound** (9016 ms)
   - the device is busy 6% of the step (620 of 9636 ms) over 25434 launches and 541647 operators per step, 373 us of host time per launch
   - lever: graph replay is already running for some steps; compare replayed and eager step times above
2. **Slowest pieces by recoverable time: conv output_blocks.in_layers (up)** (689 ms)
   - 709.6 ms per step against an ideal 20.7 ms (3% efficient, compute-bound), 9 layers; all rated pieces together leave 11113 ms
   - lever: see the module table for the individual layers
3. **Slowest pieces by recoverable time: conv output_blocks.skip_connection (up)** (474 ms)
   - 476.6 ms per step against an ideal 2.5 ms (1% efficient, compute-bound), 9 layers; all rated pieces together leave 11113 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (428 ms)
   - 437.6 ms per step against an ideal 9.8 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 11113 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (414 ms)
   - 423.6 ms per step against an ideal 9.8 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 11113 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (down)** (378 ms)
   - 384.0 ms per step against an ideal 6.5 ms (2% efficient, compute-bound), 24 layers; all rated pieces together leave 11113 ms
   - lever: see the module table for the individual layers
7. **Dtype and copy kernels take a large share** (151 ms)
   - 24% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
8. **Many small memory-bound kernels** (139 ms)
   - 45% of device time (16075 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 224.2 | 36% |
| copy | 4610 | 150.9 | 24% |
| elementwise | 9038 | 104.8 | 17% |
| convolution | 533 | 73.0 | 12% |
| other | 623 | 36.5 | 6% |
| norm | 782 | 18.5 | 3% |
| attention | 140 | 8.3 | 1% |
| memory | 1627 | 2.0 | 0% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 95.1 TFLOPS, int8 184.9 TOPS, copy 366.2 GB/s, dispatch 11.0 us, queued launch 11.0 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| conv | output_blocks.in_layers | up | 9 | 709.6 | 20.7 | 3% | compute | 688.9 |
| conv | output_blocks.skip_connection | up | 9 | 476.6 | 2.5 | 1% | compute | 474.1 |
| attention | transformer_blocks.attn1 | up | 36 | 437.6 | 9.8 | 2% | compute | 427.9 |
| attention | transformer_blocks.attn2 | up | 36 | 423.6 | 9.8 | 2% | compute | 413.8 |
| attention | transformer_blocks.attn1 | down | 24 | 384.0 | 6.5 | 2% | compute | 377.5 |
| attention | transformer_blocks.attn2 | down | 24 | 376.1 | 6.5 | 2% | compute | 369.6 |
| conv | input_blocks.op | down | 2 | 294.2 | 0.4 | 0% | compute | 293.8 |
| linear | attn2.to_q | up | 36 | 293.4 | 4.4 | 1% | compute | 289.0 |
| linear | attn2.to_out | up | 36 | 283.5 | 4.4 | 2% | compute | 279.1 |
| linear | attn1.to_out | up | 36 | 278.2 | 4.4 | 2% | compute | 273.8 |
| linear | attn2.to_v | up | 36 | 276.2 | 2.6 | 1% | memory | 273.6 |
| linear | attn1.to_q | up | 36 | 277.4 | 4.4 | 2% | compute | 273.1 |
| linear | attn1.to_k | up | 36 | 275.5 | 4.4 | 2% | compute | 271.1 |
| linear | attn2.to_k | up | 36 | 273.2 | 2.6 | 1% | memory | 270.7 |
| linear | ff.net | up | 36 | 285.9 | 17.5 | 6% | compute | 268.5 |
| linear | attn1.to_v | up | 36 | 268.9 | 4.4 | 2% | compute | 264.5 |
| linear | net.proj | up | 36 | 291.6 | 34.9 | 12% | compute | 256.7 |
| feedforward | transformer_blocks.ff | up | 36 | 257.4 | 33.1 | 13% | memory | 224.4 |
| conv | input_blocks.in_layers | down | 6 | 228.5 | 4.5 | 2% | compute | 223.9 |
| linear | attn2.to_out | down | 24 | 204.1 | 2.9 | 1% | compute | 201.2 |
| feedforward | transformer_blocks.ff | down | 24 | 222.3 | 22.0 | 10% | memory | 200.3 |
| linear | attn1.to_v | down | 24 | 202.6 | 2.9 | 1% | compute | 199.6 |
| linear | attn1.to_out | down | 24 | 201.3 | 2.9 | 1% | compute | 198.4 |
| linear | attn2.to_k | down | 24 | 199.9 | 1.7 | 1% | memory | 198.2 |
| linear | attn1.to_q | down | 24 | 200.7 | 2.9 | 1% | compute | 197.8 |
| linear | attn1.to_k | down | 24 | 200.4 | 2.9 | 1% | compute | 197.5 |
| linear | attn2.to_q | down | 24 | 199.3 | 2.9 | 1% | compute | 196.4 |
| linear | attn2.to_v | down | 24 | 197.0 | 1.7 | 1% | memory | 195.3 |
| linear | ff.net | down | 24 | 202.6 | 11.6 | 6% | compute | 191.0 |
| linear | net.proj | down | 24 | 211.6 | 23.3 | 11% | compute | 188.3 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.6.0.op | conv | 2 | 47.4 | 0.0 | 182.3 | 229.7 | 0.2 | 0% |
| output_blocks.3.0.in_layers.2 | conv | 2 | 62.6 | 2.1 | 42.2 | 106.9 | 3.3 | 3% |
| output_blocks.6.0.in_layers.2 | conv | 2 | 59.4 | 2.0 | 43.9 | 105.4 | 3.3 | 3% |
| output_blocks.4.0.in_layers.2 | conv | 2 | 61.4 | 1.5 | 41.5 | 104.4 | 2.2 | 2% |
| output_blocks.5.0.in_layers.2 | conv | 2 | 61.1 | 1.3 | 39.3 | 101.7 | 1.6 | 2% |
| output_blocks.2.0.in_layers.2 | conv | 2 | 59.6 | 2.4 | 37.5 | 99.5 | 1.6 | 2% |
| input_blocks.7.0.in_layers.2 | conv | 2 | 57.4 | 1.8 | 39.7 | 98.8 | 0.5 | 1% |
| input_blocks.7.1.transformer_blocks.2 | block | 2 | 25.9 | 0.0 | 139.5 | 81.9 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 2 | 32.9 | 0.0 | 140.9 | 80.9 | - | - |
| output_blocks.5.2.conv | conv | 2 | 39.8 | 0.0 | 40.5 | 80.3 | 2.9 | 4% |
| input_blocks.4.1.transformer_blocks.1 | block | 2 | 25.2 | 0.0 | 139.1 | 79.5 | - | - |
| output_blocks.2.0.skip_connection | conv | 2 | 44.7 | 1.8 | 32.1 | 78.6 | 0.2 | 0% |
| input_blocks.7.1.transformer_blocks.9 | block | 2 | 26.7 | 0.0 | 133.8 | 78.6 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 2 | 26.0 | 0.0 | 136.6 | 77.5 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 2 | 34.3 | 0.0 | 135.2 | 77.4 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 2 | 26.0 | 0.0 | 135.9 | 77.3 | - | - |
| output_blocks.1.1.transformer_blocks.6 | block | 2 | 29.9 | 0.0 | 120.5 | 75.8 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 2 | 25.9 | 0.0 | 133.8 | 75.7 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 2 | 25.2 | 0.0 | 133.8 | 75.6 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 2 | 26.6 | 0.0 | 131.2 | 74.7 | - | - |
| input_blocks.7.0.skip_connection | conv | 2 | 43.3 | 2.6 | 28.6 | 74.4 | 0.1 | 0% |
| input_blocks.7.1.transformer_blocks.4 | block | 2 | 25.8 | 0.0 | 131.7 | 74.3 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 2 | 25.3 | 0.0 | 130.3 | 73.8 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 2 | 27.6 | 0.0 | 129.9 | 73.3 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 2 | 27.7 | 0.0 | 128.7 | 72.8 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 2 | 27.2 | 0.0 | 126.7 | 72.4 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 2 | 27.0 | 0.0 | 125.9 | 72.1 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 2 | 26.4 | 0.0 | 125.0 | 71.5 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 2 | 26.5 | 0.0 | 124.9 | 71.4 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 2 | 27.1 | 0.0 | 125.2 | 71.2 | - | - |

## Reconciliation

device 620 ms; modules - ms (unrated - ms). Module times need eager steps: a window with replayed steps has no module table.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 217.25 | 27 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 68.34 | 32 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2141 | 62.31 | 29 |
| `gen_conv` | convolution | 131 | 57.57 | 439 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 49.29 | 16 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 31.93 | 228 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 14.50 | 69 |
| `conv_reorder` | convolution | 262 | 13.95 | 53 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 12.33 | 176 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 8.95 | 21 |
| `micro_sdpa` | attention | 140 | 8.32 | 59 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 6.98 | 50 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.90 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.24 | 89 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 5.98 | 43 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.62 | 80 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.93 | 23 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.66 | 2 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 3.46 | 105 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.71 | 37 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 2.67 | 36 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 210 | 1.99 | 9 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 1.98 | 14 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 39 | 1.89 | 48 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::SiluFunctor<c10::BFloat16` | elementwise | 96 | 1.75 | 18 |
