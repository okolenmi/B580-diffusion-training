# Performance report (xpu)

Window: 3 steps (2 eager, 1 replayed). All times are per step.

## Step budget

- wall 7381 ms (best 5986); eager 6003 ms, replayed 10136 ms
- device busy 598 ms, utilisation 8%
- 25434 kernel launches, 541647 operators, 284 us host time per launch
- step peak memory 6126 MB
- phases: loss 0 ms

## Findings, largest first

1. **The step is launch bound** (6783 ms)
   - the device is busy 8% of the step (598 of 7381 ms) over 25434 launches and 541647 operators per step, 284 us of host time per launch
   - lever: graph replay is already running for some steps; compare replayed and eager step times above
2. **Slowest pieces by recoverable time: conv output_blocks.in_layers (up)** (651 ms)
   - 671.5 ms per step against an ideal 20.7 ms (3% efficient, compute-bound), 9 layers; all rated pieces together leave 8543 ms
   - lever: see the module table for the individual layers
3. **Slowest pieces by recoverable time: conv output_blocks.skip_connection (up)** (447 ms)
   - 449.8 ms per step against an ideal 2.5 ms (1% efficient, compute-bound), 9 layers; all rated pieces together leave 8543 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (286 ms)
   - 295.9 ms per step against an ideal 9.8 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 8543 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: conv input_blocks.op (down)** (280 ms)
   - 280.5 ms per step against an ideal 0.4 ms (0% efficient, compute-bound), 2 layers; all rated pieces together leave 8543 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (269 ms)
   - 278.9 ms per step against an ideal 9.8 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 8543 ms
   - lever: see the module table for the individual layers
7. **Dtype and copy kernels take a large share** (143 ms)
   - 24% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
8. **Many small memory-bound kernels** (131 ms)
   - 44% of device time (16075 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 222.6 | 37% |
| copy | 4610 | 143.2 | 24% |
| elementwise | 9038 | 98.2 | 16% |
| convolution | 533 | 70.7 | 12% |
| other | 623 | 34.7 | 6% |
| norm | 782 | 17.1 | 3% |
| attention | 140 | 7.9 | 1% |
| memory | 1627 | 2.1 | 0% |
| reduce | 18 | 1.6 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 95.1 TFLOPS, int8 184.9 TOPS, copy 366.2 GB/s, dispatch 11.0 us, queued launch 11.0 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| conv | output_blocks.in_layers | up | 9 | 671.5 | 20.7 | 3% | compute | 650.8 |
| conv | output_blocks.skip_connection | up | 9 | 449.8 | 2.5 | 1% | compute | 447.4 |
| attention | transformer_blocks.attn1 | up | 36 | 295.9 | 9.8 | 3% | compute | 286.2 |
| conv | input_blocks.op | down | 2 | 280.5 | 0.4 | 0% | compute | 280.2 |
| attention | transformer_blocks.attn2 | up | 36 | 278.9 | 9.8 | 3% | compute | 269.2 |
| attention | transformer_blocks.attn1 | down | 24 | 249.8 | 6.5 | 3% | compute | 243.3 |
| attention | transformer_blocks.attn2 | down | 24 | 242.5 | 6.5 | 3% | compute | 236.0 |
| linear | ff.net | up | 36 | 236.6 | 17.5 | 7% | compute | 219.2 |
| linear | attn1.to_out | up | 36 | 215.6 | 4.4 | 2% | compute | 211.3 |
| conv | input_blocks.in_layers | down | 6 | 213.6 | 4.5 | 2% | compute | 209.0 |
| linear | attn1.to_q | up | 36 | 213.4 | 4.4 | 2% | compute | 209.0 |
| linear | attn2.to_out | up | 36 | 213.3 | 4.4 | 2% | compute | 209.0 |
| linear | attn2.to_q | up | 36 | 212.7 | 4.4 | 2% | compute | 208.3 |
| linear | attn2.to_v | up | 36 | 209.9 | 2.6 | 1% | memory | 207.3 |
| linear | attn2.to_k | up | 36 | 208.5 | 2.6 | 1% | memory | 206.0 |
| linear | attn1.to_k | up | 36 | 208.8 | 4.4 | 2% | compute | 204.4 |
| linear | attn1.to_v | up | 36 | 204.9 | 4.4 | 2% | compute | 200.5 |
| linear | net.proj | up | 36 | 234.6 | 34.9 | 15% | compute | 199.6 |
| linear | attn2.to_out | down | 24 | 162.5 | 2.9 | 2% | compute | 159.6 |
| linear | attn1.to_q | down | 24 | 157.0 | 2.9 | 2% | compute | 154.1 |
| linear | ff.net | down | 24 | 164.5 | 11.6 | 7% | compute | 152.8 |
| linear | attn1.to_out | down | 24 | 155.4 | 2.9 | 2% | compute | 152.5 |
| linear | attn2.to_v | down | 24 | 153.0 | 1.7 | 1% | memory | 151.3 |
| linear | attn1.to_k | down | 24 | 154.1 | 2.9 | 2% | compute | 151.2 |
| linear | attn2.to_k | down | 24 | 152.7 | 1.7 | 1% | memory | 151.0 |
| linear | attn2.to_q | down | 24 | 153.8 | 2.9 | 2% | compute | 150.9 |
| linear | attn1.to_v | down | 24 | 147.7 | 2.9 | 2% | compute | 144.8 |
| linear | net.proj | down | 24 | 164.3 | 23.3 | 14% | compute | 141.0 |
| feedforward | transformer_blocks.ff | up | 36 | 170.6 | 33.1 | 19% | memory | 137.5 |
| conv | output_blocks.out_layers | up | 9 | 139.8 | 9.8 | 7% | compute | 130.0 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.6.0.op | conv | 2 | 45.5 | 0.0 | 175.0 | 220.5 | 0.2 | 0% |
| output_blocks.3.0.in_layers.2 | conv | 2 | 59.6 | 2.0 | 40.1 | 101.7 | 3.3 | 3% |
| output_blocks.4.0.in_layers.2 | conv | 2 | 58.4 | 1.5 | 39.1 | 99.0 | 2.2 | 2% |
| output_blocks.6.0.in_layers.2 | conv | 2 | 55.6 | 1.8 | 41.3 | 98.7 | 3.3 | 3% |
| output_blocks.5.0.in_layers.2 | conv | 2 | 57.7 | 1.2 | 38.6 | 97.5 | 1.6 | 2% |
| output_blocks.2.0.in_layers.2 | conv | 2 | 55.9 | 2.2 | 35.9 | 94.1 | 1.6 | 2% |
| input_blocks.7.0.in_layers.2 | conv | 2 | 54.2 | 1.2 | 37.8 | 93.1 | 0.5 | 1% |
| output_blocks.5.2.conv | conv | 2 | 37.8 | 0.0 | 38.0 | 75.8 | 2.9 | 4% |
| output_blocks.2.0.skip_connection | conv | 2 | 43.4 | 1.0 | 29.7 | 74.1 | 0.2 | 0% |
| input_blocks.7.0.skip_connection | conv | 2 | 42.1 | 1.6 | 27.5 | 71.2 | 0.1 | 0% |
| output_blocks.5.0.skip_connection | conv | 2 | 29.7 | 0.8 | 35.1 | 65.5 | 0.2 | 0% |
| out.2 | conv | 2 | 17.8 | 0.0 | 45.8 | 63.6 | 0.1 | 0% |
| output_blocks.3.0.skip_connection | conv | 2 | 30.5 | 0.8 | 30.2 | 61.5 | 0.4 | 1% |
| output_blocks.4.0.skip_connection | conv | 2 | 30.4 | 0.8 | 29.9 | 61.1 | 0.2 | 0% |
| output_blocks.0.0.in_layers.2 | conv | 2 | 55.2 | 3.0 | 2.5 | 60.7 | 2.2 | 4% |
| input_blocks.3.0.op | conv | 2 | 60.1 | 0.0 | 0.0 | 60.1 | 0.2 | 0% |
| output_blocks.2.2.conv | conv | 2 | 29.7 | 0.0 | 30.0 | 59.7 | 2.9 | 5% |
| input_blocks.7.0.out_layers.3 | conv | 2 | 55.5 | 1.4 | 1.9 | 58.9 | 1.1 | 2% |
| input_blocks.4.0.out_layers.3 | conv | 2 | 58.3 | 0.0 | 0.0 | 58.3 | 0.7 | 1% |
| input_blocks.7.1.transformer_blocks.2 | block | 2 | 21.8 | 0.0 | 102.0 | 57.4 | - | - |
| input_blocks.4.0.in_layers.2 | conv | 2 | 57.2 | 0.0 | 0.0 | 57.2 | 0.4 | 1% |
| input_blocks.4.1.transformer_blocks.0 | block | 2 | 29.7 | 0.0 | 96.0 | 55.3 | - | - |
| input_blocks.0.0 | conv | 2 | 54.5 | 0.0 | 0.0 | 54.5 | 0.1 | 0% |
| input_blocks.7.1.transformer_blocks.9 | block | 2 | 23.2 | 0.0 | 93.2 | 54.2 | - | - |
| input_blocks.1.0.in_layers.2 | conv | 2 | 53.8 | 0.0 | 0.0 | 53.8 | 0.7 | 1% |
| input_blocks.4.1.transformer_blocks.1 | block | 2 | 23.5 | 0.0 | 91.7 | 52.2 | - | - |
| output_blocks.6.0.skip_connection | conv | 2 | 22.5 | 0.7 | 28.8 | 52.1 | 0.4 | 1% |
| input_blocks.7.1.transformer_blocks.4 | block | 2 | 22.8 | 0.0 | 91.5 | 51.7 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 2 | 22.7 | 0.0 | 91.4 | 51.3 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 2 | 22.4 | 0.0 | 90.5 | 51.2 | - | - |

## Reconciliation

device 598 ms; modules - ms (unrated - ms). Module times need eager steps: a window with replayed steps has no module table.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 216.05 | 27 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 65.85 | 31 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2141 | 58.64 | 27 |
| `gen_conv` | convolution | 131 | 55.97 | 427 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 47.05 | 15 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 30.11 | 215 |
| `conv_reorder` | convolution | 262 | 13.57 | 52 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 13.27 | 63 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 12.26 | 175 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 8.33 | 20 |
| `micro_sdpa` | attention | 140 | 7.93 | 57 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 6.58 | 47 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.54 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 5.95 | 42 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.33 | 76 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 4.86 | 69 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.19 | 20 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.21 | 2 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 2.65 | 36 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 2.57 | 78 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.51 | 34 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 210 | 2.07 | 10 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 2.02 | 14 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 39 | 1.91 | 49 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::SiluFunctor<c10::BFloat16` | elementwise | 96 | 1.76 | 18 |
