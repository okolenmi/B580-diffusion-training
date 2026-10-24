# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- wall 7627 ms (best 6342); eager 7627 ms, replayed - ms
- device busy 642 ms, utilisation 8%
- 25434 kernel launches, 541506 operators, 297 us host time per launch
- step peak memory 6125 MB
- phases: loss 0 ms

## Findings, largest first

1. **The step is launch bound** (6985 ms)
   - the device is busy 8% of the step (642 of 7627 ms) over 25434 launches and 541506 operators per step, 297 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
2. **Much of the module time has no cost model** (2458 ms)
   - 2458 of 8599 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
3. **Slowest pieces by recoverable time: conv output_blocks.in_layers (up)** (438 ms)
   - 451.4 ms per step against an ideal 13.8 ms (3% efficient, compute-bound), 9 layers; all rated pieces together leave 5915 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: conv output_blocks.skip_connection (up)** (294 ms)
   - 295.6 ms per step against an ideal 1.6 ms (1% efficient, compute-bound), 9 layers; all rated pieces together leave 5915 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (194 ms)
   - 200.5 ms per step against an ideal 6.5 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 5915 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: conv input_blocks.op (down)** (183 ms)
   - 182.9 ms per step against an ideal 0.2 ms (0% efficient, compute-bound), 2 layers; all rated pieces together leave 5915 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (180 ms)
   - 186.7 ms per step against an ideal 6.5 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 5915 ms
   - lever: see the module table for the individual layers
8. **Dtype and copy kernels take a large share** (161 ms)
   - 25% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
9. **Many small memory-bound kernels** (149 ms)
   - 46% of device time (16075 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
10. **Module times do not add up to the device time** (0 ms)
   - modules account for 8599 ms of 642 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 223.1 | 35% |
| copy | 4610 | 160.8 | 25% |
| elementwise | 9038 | 105.9 | 16% |
| convolution | 533 | 76.8 | 12% |
| other | 623 | 35.6 | 6% |
| norm | 782 | 27.2 | 4% |
| attention | 140 | 9.4 | 1% |
| memory | 1627 | 2.1 | 0% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 95.1 TFLOPS, int8 184.9 TOPS, copy 366.2 GB/s, dispatch 11.0 us, queued launch 11.0 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| conv | output_blocks.in_layers | up | 9 | 451.4 | 13.8 | 3% | compute | 437.6 |
| conv | output_blocks.skip_connection | up | 9 | 295.6 | 1.6 | 1% | compute | 294.0 |
| attention | transformer_blocks.attn1 | up | 36 | 200.5 | 6.5 | 3% | compute | 194.0 |
| conv | input_blocks.op | down | 2 | 182.9 | 0.2 | 0% | compute | 182.7 |
| attention | transformer_blocks.attn2 | up | 36 | 186.7 | 6.5 | 3% | compute | 180.2 |
| linear | attn2.to_k | up | 36 | 168.9 | 1.7 | 1% | memory | 167.2 |
| attention | transformer_blocks.attn1 | down | 24 | 168.8 | 4.3 | 3% | compute | 164.5 |
| attention | transformer_blocks.attn2 | down | 24 | 164.2 | 4.3 | 3% | compute | 159.9 |
| linear | attn2.to_v | up | 36 | 154.6 | 1.7 | 1% | memory | 152.9 |
| linear | ff.net | up | 36 | 162.2 | 11.6 | 7% | compute | 150.6 |
| linear | attn2.to_q | up | 36 | 152.5 | 2.9 | 2% | compute | 149.6 |
| linear | attn2.to_out | up | 36 | 152.1 | 2.9 | 2% | compute | 149.2 |
| linear | attn1.to_q | up | 36 | 151.1 | 2.9 | 2% | compute | 148.2 |
| linear | attn1.to_out | up | 36 | 150.2 | 2.9 | 2% | compute | 147.3 |
| linear | attn1.to_k | up | 36 | 147.7 | 2.9 | 2% | compute | 144.8 |
| linear | attn1.to_v | up | 36 | 144.1 | 2.9 | 2% | compute | 141.2 |
| conv | input_blocks.in_layers | down | 6 | 141.5 | 3.0 | 2% | compute | 138.5 |
| linear | net.proj | up | 36 | 159.0 | 23.3 | 15% | compute | 135.7 |
| linear | attn2.to_out | down | 24 | 114.0 | 1.9 | 2% | compute | 112.0 |
| linear | attn2.to_q | down | 24 | 112.3 | 1.9 | 2% | compute | 110.4 |
| linear | attn2.to_k | down | 24 | 106.9 | 1.1 | 1% | memory | 105.8 |
| linear | ff.net | down | 24 | 112.6 | 7.8 | 7% | compute | 104.8 |
| linear | attn1.to_out | down | 24 | 105.7 | 1.9 | 2% | compute | 103.8 |
| linear | attn2.to_v | down | 24 | 104.6 | 1.1 | 1% | memory | 103.5 |
| linear | attn1.to_q | down | 24 | 104.7 | 1.9 | 2% | compute | 102.7 |
| linear | attn1.to_k | down | 24 | 104.5 | 1.9 | 2% | compute | 102.6 |
| linear | attn1.to_v | down | 24 | 101.7 | 1.9 | 2% | compute | 99.8 |
| linear | net.proj | down | 24 | 115.0 | 15.5 | 13% | compute | 99.5 |
| feedforward | transformer_blocks.ff | up | 36 | 117.9 | 22.0 | 19% | memory | 95.8 |
| conv | output_blocks.out_layers | up | 9 | 94.9 | 6.5 | 7% | compute | 88.3 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.6.0.op | conv | 1 | 30.0 | 0.0 | 112.9 | 142.9 | 0.1 | 0% |
| output_blocks.3.0.in_layers.2 | conv | 1 | 39.3 | 1.4 | 27.0 | 67.7 | 2.2 | 3% |
| output_blocks.4.0.in_layers.2 | conv | 1 | 38.5 | 1.1 | 26.2 | 65.7 | 1.5 | 2% |
| output_blocks.6.0.in_layers.2 | conv | 1 | 35.8 | 1.3 | 27.3 | 64.3 | 2.2 | 3% |
| output_blocks.5.0.in_layers.2 | conv | 1 | 38.3 | 0.7 | 25.2 | 64.2 | 1.1 | 2% |
| output_blocks.2.0.in_layers.2 | conv | 1 | 37.9 | 1.6 | 23.9 | 63.4 | 1.1 | 2% |
| input_blocks.7.0.in_layers.2 | conv | 1 | 35.7 | 0.8 | 24.4 | 60.9 | 0.4 | 1% |
| output_blocks.5.2.conv | conv | 1 | 24.6 | 0.0 | 25.3 | 49.9 | 1.9 | 4% |
| output_blocks.2.0.skip_connection | conv | 1 | 27.8 | 0.6 | 20.3 | 48.7 | 0.1 | 0% |
| input_blocks.7.0.skip_connection | conv | 1 | 27.4 | 0.8 | 17.9 | 46.1 | 0.0 | 0% |
| out.2 | conv | 1 | 12.0 | 0.0 | 30.9 | 42.9 | 0.0 | 0% |
| output_blocks.5.0.skip_connection | conv | 1 | 19.4 | 0.4 | 22.5 | 42.2 | 0.1 | 0% |
| output_blocks.0.0.in_layers.2 | conv | 1 | 36.9 | 3.3 | 1.6 | 41.7 | 1.5 | 3% |
| output_blocks.3.0.skip_connection | conv | 1 | 20.2 | 0.7 | 20.5 | 41.4 | 0.2 | 1% |
| output_blocks.4.0.skip_connection | conv | 1 | 20.2 | 0.5 | 20.2 | 40.8 | 0.2 | 0% |
| output_blocks.1.1.transformer_blocks.6 | block | 1 | 15.7 | 0.0 | 63.2 | 40.5 | - | - |
| output_blocks.2.2.conv | conv | 1 | 19.9 | 0.0 | 20.2 | 40.1 | 1.9 | 5% |
| input_blocks.3.0.op | conv | 1 | 40.0 | 0.0 | 0.0 | 40.0 | 0.1 | 0% |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 16.1 | 0.0 | 66.6 | 39.3 | - | - |
| input_blocks.4.0.out_layers.3 | conv | 1 | 38.8 | 0.0 | 0.0 | 38.8 | 0.5 | 1% |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 14.9 | 0.0 | 66.3 | 38.4 | - | - |
| input_blocks.7.0.out_layers.3 | conv | 1 | 36.3 | 1.0 | 1.1 | 38.4 | 0.7 | 2% |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 15.0 | 0.0 | 68.5 | 38.0 | - | - |
| input_blocks.4.0.in_layers.2 | conv | 1 | 37.9 | 0.0 | 0.0 | 37.9 | 0.2 | 1% |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 14.0 | 0.0 | 65.8 | 37.6 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 14.7 | 0.0 | 64.5 | 37.0 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 15.3 | 0.0 | 65.9 | 36.8 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 14.4 | 0.0 | 64.6 | 36.7 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 14.7 | 0.0 | 64.1 | 36.6 | - | - |
| input_blocks.1.0.in_layers.2 | conv | 1 | 36.5 | 0.0 | 0.0 | 36.5 | 0.5 | 1% |

## Reconciliation

device 642 ms; modules' own time 8599 ms, of which 2458 ms is in modules with no cost model; wall 7627 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 216.26 | 27 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 74.44 | 35 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2141 | 62.85 | 29 |
| `gen_conv` | convolution | 131 | 57.81 | 441 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 52.58 | 17 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 31.11 | 222 |
| `conv_reorder` | convolution | 262 | 17.76 | 68 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 13.64 | 65 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 13.42 | 32 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 12.24 | 175 |
| `micro_sdpa` | attention | 140 | 9.37 | 67 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 7.48 | 101 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 7.07 | 2 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 6.85 | 49 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 5.80 | 41 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.38 | 77 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | copy | 43 | 5.01 | 116 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 4.93 | 70 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.32 | 21 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.26 | 2 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.53 | 34 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 2.38 | 72 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 2.05 | 15 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 210 | 1.95 | 9 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::SiluFunctor<c10::BFloat16` | elementwise | 96 | 1.95 | 20 |
