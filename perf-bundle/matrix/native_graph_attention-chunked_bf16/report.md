# Performance report (xpu)

Window: 3 steps (2 eager, 1 replayed). All times are per step.

## Step budget

- wall 7382 ms (best 5943); eager 6002 ms, replayed 10142 ms
- device busy 595 ms, utilisation 8%
- 25434 kernel launches, 541647 operators, 284 us host time per launch
- step peak memory 6126 MB
- phases: loss 0 ms

## Findings, largest first

1. **The step is launch bound** (6787 ms)
   - the device is busy 8% of the step (595 of 7382 ms) over 25434 launches and 541647 operators per step, 284 us of host time per launch
   - lever: graph replay is already running for some steps; compare replayed and eager step times above
2. **Slowest pieces by recoverable time: conv output_blocks.in_layers (up)** (647 ms)
   - 667.5 ms per step against an ideal 20.7 ms (3% efficient, compute-bound), 9 layers; all rated pieces together leave 8544 ms
   - lever: see the module table for the individual layers
3. **Slowest pieces by recoverable time: conv output_blocks.skip_connection (up)** (445 ms)
   - 447.4 ms per step against an ideal 2.5 ms (1% efficient, compute-bound), 9 layers; all rated pieces together leave 8544 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (291 ms)
   - 300.6 ms per step against an ideal 9.8 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 8544 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: conv input_blocks.op (down)** (282 ms)
   - 282.0 ms per step against an ideal 0.4 ms (0% efficient, compute-bound), 2 layers; all rated pieces together leave 8544 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (269 ms)
   - 278.3 ms per step against an ideal 9.8 ms (4% efficient, compute-bound), 36 layers; all rated pieces together leave 8544 ms
   - lever: see the module table for the individual layers
7. **Dtype and copy kernels take a large share** (141 ms)
   - 24% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
8. **Many small memory-bound kernels** (130 ms)
   - 44% of device time (16075 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 222.8 | 37% |
| copy | 4610 | 141.4 | 24% |
| elementwise | 9038 | 98.1 | 16% |
| convolution | 533 | 69.8 | 12% |
| other | 623 | 34.8 | 6% |
| norm | 782 | 17.0 | 3% |
| attention | 140 | 7.8 | 1% |
| memory | 1627 | 2.0 | 0% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 95.1 TFLOPS, int8 184.9 TOPS, copy 366.2 GB/s, dispatch 11.0 us, queued launch 11.0 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| conv | output_blocks.in_layers | up | 9 | 667.5 | 20.7 | 3% | compute | 646.7 |
| conv | output_blocks.skip_connection | up | 9 | 447.4 | 2.5 | 1% | compute | 445.0 |
| attention | transformer_blocks.attn1 | up | 36 | 300.6 | 9.8 | 3% | compute | 290.9 |
| conv | input_blocks.op | down | 2 | 282.0 | 0.4 | 0% | compute | 281.6 |
| attention | transformer_blocks.attn2 | up | 36 | 278.3 | 9.8 | 4% | compute | 268.6 |
| attention | transformer_blocks.attn1 | down | 24 | 249.2 | 6.5 | 3% | compute | 242.7 |
| attention | transformer_blocks.attn2 | down | 24 | 241.0 | 6.5 | 3% | compute | 234.5 |
| linear | attn1.to_q | up | 36 | 230.3 | 4.4 | 2% | compute | 226.0 |
| linear | attn1.to_out | up | 36 | 221.0 | 4.4 | 2% | compute | 216.7 |
| linear | attn2.to_out | up | 36 | 215.1 | 4.4 | 2% | compute | 210.7 |
| linear | attn2.to_q | up | 36 | 214.8 | 4.4 | 2% | compute | 210.4 |
| conv | input_blocks.in_layers | down | 6 | 214.8 | 4.5 | 2% | compute | 210.3 |
| linear | attn2.to_v | up | 36 | 211.3 | 2.6 | 1% | memory | 208.8 |
| linear | attn2.to_k | up | 36 | 210.8 | 2.6 | 1% | memory | 208.2 |
| linear | attn1.to_k | up | 36 | 210.1 | 4.4 | 2% | compute | 205.7 |
| linear | ff.net | up | 36 | 220.8 | 17.5 | 8% | compute | 203.3 |
| linear | attn1.to_v | up | 36 | 203.1 | 4.4 | 2% | compute | 198.7 |
| linear | net.proj | up | 36 | 231.9 | 34.9 | 15% | compute | 197.0 |
| linear | attn2.to_out | down | 24 | 158.6 | 2.9 | 2% | compute | 155.6 |
| linear | attn2.to_k | down | 24 | 156.6 | 1.7 | 1% | memory | 154.9 |
| linear | attn1.to_out | down | 24 | 157.3 | 2.9 | 2% | compute | 154.4 |
| linear | attn1.to_q | down | 24 | 157.1 | 2.9 | 2% | compute | 154.2 |
| linear | attn2.to_q | down | 24 | 155.6 | 2.9 | 2% | compute | 152.7 |
| linear | attn2.to_v | down | 24 | 154.3 | 1.7 | 1% | memory | 152.6 |
| linear | attn1.to_k | down | 24 | 155.5 | 2.9 | 2% | compute | 152.6 |
| linear | attn1.to_v | down | 24 | 151.5 | 2.9 | 2% | compute | 148.6 |
| linear | ff.net | down | 24 | 157.2 | 11.6 | 7% | compute | 145.5 |
| linear | net.proj | down | 24 | 165.3 | 23.3 | 14% | compute | 142.0 |
| feedforward | transformer_blocks.ff | up | 36 | 169.9 | 33.1 | 19% | memory | 136.8 |
| conv | output_blocks.conv | up | 2 | 136.5 | 5.8 | 4% | compute | 130.7 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.6.0.op | conv | 2 | 46.1 | 0.0 | 174.9 | 221.0 | 0.2 | 0% |
| output_blocks.3.0.in_layers.2 | conv | 2 | 59.0 | 2.0 | 41.0 | 102.0 | 3.3 | 3% |
| output_blocks.4.0.in_layers.2 | conv | 2 | 59.0 | 1.4 | 38.9 | 99.3 | 2.2 | 2% |
| output_blocks.6.0.in_layers.2 | conv | 2 | 54.8 | 1.7 | 40.8 | 97.3 | 3.3 | 3% |
| output_blocks.5.0.in_layers.2 | conv | 2 | 58.1 | 1.2 | 37.3 | 96.6 | 1.6 | 2% |
| input_blocks.7.0.in_layers.2 | conv | 2 | 54.4 | 1.1 | 37.6 | 93.1 | 0.5 | 1% |
| output_blocks.2.0.in_layers.2 | conv | 2 | 54.8 | 2.4 | 35.8 | 92.9 | 1.6 | 2% |
| output_blocks.5.2.conv | conv | 2 | 37.7 | 0.0 | 37.8 | 75.4 | 2.9 | 4% |
| output_blocks.2.0.skip_connection | conv | 2 | 42.5 | 1.0 | 30.0 | 73.5 | 0.2 | 0% |
| input_blocks.7.0.skip_connection | conv | 2 | 42.2 | 1.5 | 27.2 | 70.9 | 0.1 | 0% |
| output_blocks.5.0.skip_connection | conv | 2 | 29.9 | 0.7 | 34.0 | 64.6 | 0.2 | 0% |
| out.2 | conv | 2 | 18.0 | 0.0 | 44.8 | 62.7 | 0.1 | 0% |
| output_blocks.3.0.skip_connection | conv | 2 | 30.2 | 0.8 | 31.2 | 62.2 | 0.4 | 1% |
| output_blocks.2.2.conv | conv | 2 | 29.8 | 0.0 | 31.2 | 61.1 | 2.9 | 5% |
| input_blocks.3.0.op | conv | 2 | 61.0 | 0.0 | 0.0 | 61.0 | 0.2 | 0% |
| output_blocks.4.0.skip_connection | conv | 2 | 30.5 | 0.7 | 29.5 | 60.8 | 0.2 | 0% |
| output_blocks.0.0.in_layers.2 | conv | 2 | 55.3 | 2.9 | 2.6 | 60.7 | 2.2 | 4% |
| input_blocks.4.0.out_layers.3 | conv | 2 | 58.5 | 0.0 | 0.0 | 58.5 | 0.7 | 1% |
| input_blocks.7.0.out_layers.3 | conv | 2 | 55.4 | 1.4 | 1.8 | 58.5 | 1.1 | 2% |
| input_blocks.4.0.in_layers.2 | conv | 2 | 57.9 | 0.0 | 0.0 | 57.9 | 0.4 | 1% |
| input_blocks.4.1.transformer_blocks.1 | block | 2 | 22.2 | 0.0 | 101.4 | 56.7 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 2 | 24.0 | 0.0 | 100.4 | 56.4 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 2 | 29.2 | 0.0 | 98.1 | 56.1 | - | - |
| input_blocks.0.0 | conv | 2 | 54.9 | 0.0 | 0.0 | 54.9 | 0.1 | 0% |
| input_blocks.1.0.in_layers.2 | conv | 2 | 54.6 | 0.0 | 0.0 | 54.6 | 0.7 | 1% |
| input_blocks.7.1.transformer_blocks.2 | block | 2 | 22.1 | 0.0 | 94.1 | 53.3 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 2 | 23.2 | 0.0 | 94.7 | 52.2 | - | - |
| output_blocks.6.0.skip_connection | conv | 2 | 22.4 | 0.7 | 28.7 | 51.8 | 0.4 | 1% |
| input_blocks.7.1.transformer_blocks.3 | block | 2 | 23.3 | 0.0 | 91.8 | 51.2 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 2 | 29.7 | 0.0 | 91.6 | 51.2 | - | - |

## Reconciliation

device 595 ms; modules - ms (unrated - ms). Module times need eager steps: a window with replayed steps has no module table.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 215.98 | 27 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 64.67 | 30 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2141 | 57.99 | 27 |
| `gen_conv` | convolution | 131 | 54.62 | 417 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 47.17 | 15 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 30.19 | 216 |
| `conv_reorder` | convolution | 262 | 13.94 | 53 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 13.27 | 63 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 12.12 | 173 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 8.31 | 20 |
| `micro_sdpa` | attention | 140 | 7.85 | 56 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 6.80 | 49 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.44 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 5.93 | 42 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.36 | 77 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 4.98 | 71 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.16 | 20 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.22 | 2 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 2.69 | 82 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 2.63 | 36 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.46 | 33 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 2.12 | 15 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 210 | 2.00 | 10 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 39 | 1.88 | 48 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::SiluFunctor<c10::BFloat16` | elementwise | 96 | 1.72 | 18 |
