# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- wall 11973 ms (best 10466); eager 11973 ms, replayed - ms
- device busy 617 ms, utilisation 5%
- 42013 kernel launches, 927899 operators, 273 us host time per launch
- step peak memory 7182 MB
- phases: loss 0 ms

## Findings, largest first

1. **The step is launch bound** (11357 ms)
   - the device is busy 5% of the step (617 of 11973 ms) over 42013 launches and 927899 operators per step, 273 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
2. **Much of the module time has no cost model** (2989 ms)
   - 2989 of 12451 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
3. **Slowest pieces by recoverable time: conv output_blocks.in_layers (up)** (419 ms)
   - 431.5 ms per step against an ideal 12.5 ms (3% efficient, compute-bound), 9 layers; all rated pieces together leave 9254 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (374 ms)
   - 380.6 ms per step against an ideal 6.2 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 9254 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (346 ms)
   - 351.9 ms per step against an ideal 6.2 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 9254 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: conv output_blocks.skip_connection (up)** (294 ms)
   - 295.5 ms per step against an ideal 1.5 ms (1% efficient, memory-bound), 9 layers; all rated pieces together leave 9254 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (down)** (285 ms)
   - 289.2 ms per step against an ideal 4.1 ms (1% efficient, compute-bound), 24 layers; all rated pieces together leave 9254 ms
   - lever: see the module table for the individual layers
8. **Dtype and copy kernels take a large share** (134 ms)
   - 22% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
9. **Many small memory-bound kernels** (123 ms)
   - 40% of device time (25639 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
10. **Module times do not add up to the device time** (0 ms)
   - modules account for 12451 ms of 617 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 14054 | 254.0 | 41% |
| copy | 7808 | 133.8 | 22% |
| elementwise | 14654 | 90.9 | 15% |
| convolution | 946 | 71.6 | 12% |
| other | 1214 | 37.5 | 6% |
| norm | 1352 | 18.7 | 3% |
| attention | 160 | 6.6 | 1% |
| memory | 1806 | 2.1 | 0% |
| reduce | 19 | 1.4 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 95.1 TFLOPS, int8 184.9 TOPS, copy 366.2 GB/s, dispatch 11.0 us, queued launch 11.0 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| conv | output_blocks.in_layers | up | 9 | 431.5 | 12.5 | 3% | compute | 419.1 |
| attention | transformer_blocks.attn2 | up | 36 | 380.6 | 6.2 | 2% | compute | 374.3 |
| attention | transformer_blocks.attn1 | up | 36 | 351.9 | 6.2 | 2% | compute | 345.7 |
| conv | output_blocks.skip_connection | up | 9 | 295.5 | 1.5 | 1% | memory | 294.0 |
| attention | transformer_blocks.attn2 | down | 24 | 289.2 | 4.1 | 1% | compute | 285.1 |
| linear | attn1.to_out | up | 36 | 284.1 | 2.7 | 1% | memory | 281.3 |
| linear | attn1.to_q | up | 36 | 283.8 | 2.7 | 1% | memory | 281.1 |
| linear | attn2.to_out | up | 36 | 279.4 | 2.7 | 1% | memory | 276.7 |
| linear | attn1.to_k | up | 36 | 278.8 | 2.7 | 1% | memory | 276.1 |
| linear | attn2.to_q | up | 36 | 278.6 | 2.7 | 1% | memory | 275.9 |
| linear | ff.net | up | 36 | 283.2 | 10.3 | 4% | compute | 272.9 |
| linear | attn1.to_v | up | 36 | 273.4 | 2.7 | 1% | memory | 270.7 |
| linear | net.proj | up | 36 | 287.2 | 20.7 | 7% | compute | 266.5 |
| attention | transformer_blocks.attn1 | down | 24 | 248.9 | 4.1 | 2% | compute | 244.8 |
| linear | attn2.to_k | up | 36 | 232.5 | 2.8 | 1% | memory | 229.7 |
| linear | attn2.to_v | up | 36 | 231.7 | 2.8 | 1% | memory | 229.0 |
| linear | attn2.to_out | down | 24 | 199.0 | 1.7 | 1% | memory | 197.3 |
| feedforward | transformer_blocks.ff | up | 36 | 206.0 | 20.5 | 10% | memory | 185.5 |
| linear | attn1.to_q | down | 24 | 186.5 | 1.7 | 1% | memory | 184.8 |
| linear | attn1.to_out | down | 24 | 186.3 | 1.7 | 1% | memory | 184.6 |
| linear | ff.net | down | 24 | 190.3 | 6.5 | 3% | compute | 183.8 |
| linear | attn1.to_v | down | 24 | 184.0 | 1.7 | 1% | memory | 182.3 |
| linear | attn2.to_q | down | 24 | 184.0 | 1.7 | 1% | memory | 182.3 |
| linear | attn1.to_k | down | 24 | 183.8 | 1.7 | 1% | memory | 182.1 |
| linear | net.proj | down | 24 | 192.2 | 12.9 | 7% | compute | 179.3 |
| conv | input_blocks.op | down | 2 | 167.9 | 0.2 | 0% | compute | 167.6 |
| linear | output_blocks.proj_out | up | 6 | 145.1 | 0.3 | 0% | memory | 144.7 |
| linear | attn2.to_v | down | 24 | 145.0 | 1.7 | 1% | memory | 143.3 |
| conv | input_blocks.in_layers | down | 6 | 143.0 | 2.4 | 2% | compute | 140.6 |
| linear | attn2.to_k | down | 24 | 139.7 | 1.7 | 1% | memory | 138.0 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.6.0.op | conv | 2 | 35.7 | 0.0 | 116.1 | 151.7 | 0.1 | 0% |
| input_blocks.8.1.transformer_blocks.6 | block | 2 | 32.3 | 0.0 | 121.9 | 72.1 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 2 | 31.5 | 0.0 | 121.1 | 68.3 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 2 | 36.2 | 0.0 | 123.0 | 68.2 | - | - |
| output_blocks.3.0.in_layers.2 | conv | 2 | 39.2 | 1.9 | 26.7 | 67.8 | 2.2 | 3% |
| input_blocks.5.1.transformer_blocks.0 | block | 2 | 31.3 | 0.0 | 120.8 | 67.8 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 2 | 32.0 | 0.0 | 120.9 | 67.7 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 2 | 36.6 | 0.0 | 120.3 | 67.1 | - | - |
| output_blocks.2.2.conv | conv | 2 | 39.6 | 0.0 | 26.9 | 66.5 | 1.9 | 3% |
| output_blocks.6.0.in_layers.2 | conv | 2 | 36.0 | 1.6 | 28.2 | 65.7 | 2.2 | 3% |
| output_blocks.5.0.in_layers.2 | conv | 2 | 39.7 | 1.2 | 24.8 | 65.7 | 1.1 | 2% |
| input_blocks.8.1.transformer_blocks.0 | block | 2 | 32.0 | 0.0 | 117.1 | 65.7 | - | - |
| output_blocks.5.1.proj_out | linear | 2 | 2.5 | 0.0 | 62.8 | 65.4 | 0.1 | 0% |
| output_blocks.4.0.in_layers.2 | conv | 2 | 37.9 | 1.4 | 26.0 | 65.3 | 1.5 | 2% |
| input_blocks.8.1.transformer_blocks.2 | block | 2 | 33.6 | 0.0 | 114.2 | 65.1 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 2 | 32.5 | 0.0 | 116.2 | 64.5 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 2 | 32.0 | 0.0 | 114.8 | 64.1 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 2 | 32.7 | 0.0 | 111.2 | 62.6 | - | - |
| middle_block.1.transformer_blocks.2 | block | 2 | 31.8 | 0.0 | 111.7 | 62.1 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 2 | 32.5 | 0.0 | 110.5 | 61.8 | - | - |
| input_blocks.7.0.in_layers.2 | conv | 2 | 29.3 | 0.0 | 32.2 | 61.5 | 0.2 | 0% |
| middle_block.1.transformer_blocks.1 | block | 2 | 32.5 | 0.0 | 110.0 | 61.4 | - | - |
| middle_block.1.transformer_blocks.3 | block | 2 | 32.5 | 0.0 | 109.5 | 61.2 | - | - |
| middle_block.1.transformer_blocks.4 | block | 2 | 31.4 | 0.0 | 109.3 | 60.8 | - | - |
| output_blocks.0.1.transformer_blocks.8 | block | 2 | 39.8 | 0.0 | 108.0 | 60.6 | - | - |
| output_blocks.0.1.transformer_blocks.3 | block | 2 | 33.5 | 0.0 | 106.6 | 60.4 | - | - |
| output_blocks.0.1.transformer_blocks.5 | block | 2 | 32.9 | 0.0 | 105.9 | 59.7 | - | - |
| output_blocks.0.1.transformer_blocks.6 | block | 2 | 32.6 | 0.0 | 109.3 | 59.5 | - | - |
| output_blocks.0.1.transformer_blocks.4 | block | 2 | 33.8 | 0.0 | 106.0 | 59.4 | - | - |
| output_blocks.1.1.transformer_blocks.4 | block | 2 | 34.5 | 0.0 | 104.1 | 59.2 | - | - |

## Reconciliation

device 617 ms; modules' own time 12451 ms, of which 2989 ms is in modules with no cost model; wall 11973 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 13774 | 241.05 | 18 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 3690 | 58.34 | 16 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 3574 | 58.09 | 16 |
| `gen_conv` | convolution | 222 | 53.23 | 240 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 5134 | 41.03 | 8 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 280 | 31.92 | 114 |
| `conv_reorder` | convolution | 444 | 17.40 | 39 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 140 | 10.89 | 78 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 240 | 10.14 | 42 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 5207 | 8.93 | 2 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 660 | 8.29 | 13 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 360 | 8.28 | 23 |
| `micro_sdpa` | attention | 160 | 6.60 | 41 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 420 | 5.78 | 14 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 2912 | 5.25 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 140 | 5.23 | 37 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 220 | 4.89 | 22 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 116 | 4.25 | 37 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 140 | 3.74 | 27 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 3.33 | 101 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 40 | 2.86 | 72 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 420 | 2.66 | 6 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 116 | 2.56 | 22 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 280 | 2.11 | 8 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 78 | 1.91 | 24 |
