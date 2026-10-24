# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- wall 8442 ms (best 6987); eager 8442 ms, replayed - ms
- device busy 614 ms, utilisation 7%
- 25434 kernel launches, 541506 operators, 328 us host time per launch
- step peak memory 6125 MB
- phases: loss 0 ms

## Findings, largest first

1. **The step is launch bound** (7828 ms)
   - the device is busy 7% of the step (614 of 8442 ms) over 25434 launches and 541506 operators per step, 328 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
2. **Much of the module time has no cost model** (2743 ms)
   - 2743 of 9431 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
3. **Slowest pieces by recoverable time: conv output_blocks.in_layers (up)** (445 ms)
   - 459.0 ms per step against an ideal 13.8 ms (3% efficient, compute-bound), 9 layers; all rated pieces together leave 6461 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: conv output_blocks.skip_connection (up)** (306 ms)
   - 308.0 ms per step against an ideal 1.6 ms (1% efficient, compute-bound), 9 layers; all rated pieces together leave 6461 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (220 ms)
   - 226.4 ms per step against an ideal 6.5 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 6461 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (209 ms)
   - 215.7 ms per step against an ideal 6.5 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 6461 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: conv input_blocks.op (down)** (196 ms)
   - 196.3 ms per step against an ideal 0.2 ms (0% efficient, compute-bound), 2 layers; all rated pieces together leave 6461 ms
   - lever: see the module table for the individual layers
8. **Dtype and copy kernels take a large share** (147 ms)
   - 24% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
9. **Many small memory-bound kernels** (135 ms)
   - 44% of device time (16075 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
10. **Module times do not add up to the device time** (0 ms)
   - modules account for 9431 ms of 614 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 223.9 | 36% |
| copy | 4610 | 146.7 | 24% |
| elementwise | 9038 | 102.0 | 17% |
| convolution | 533 | 74.1 | 12% |
| other | 623 | 37.0 | 6% |
| norm | 782 | 18.1 | 3% |
| attention | 140 | 8.4 | 1% |
| memory | 1627 | 2.0 | 0% |
| reduce | 18 | 1.6 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 95.1 TFLOPS, int8 184.9 TOPS, copy 366.2 GB/s, dispatch 11.0 us, queued launch 11.0 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| conv | output_blocks.in_layers | up | 9 | 459.0 | 13.8 | 3% | compute | 445.2 |
| conv | output_blocks.skip_connection | up | 9 | 308.0 | 1.6 | 1% | compute | 306.4 |
| attention | transformer_blocks.attn1 | up | 36 | 226.4 | 6.5 | 3% | compute | 219.9 |
| attention | transformer_blocks.attn2 | up | 36 | 215.7 | 6.5 | 3% | compute | 209.2 |
| conv | input_blocks.op | down | 2 | 196.3 | 0.2 | 0% | compute | 196.1 |
| attention | transformer_blocks.attn1 | down | 24 | 198.7 | 4.3 | 2% | compute | 194.4 |
| attention | transformer_blocks.attn2 | down | 24 | 191.1 | 4.3 | 2% | compute | 186.7 |
| linear | attn2.to_out | up | 36 | 180.9 | 2.9 | 2% | compute | 178.0 |
| linear | attn1.to_q | up | 36 | 166.5 | 2.9 | 2% | compute | 163.5 |
| linear | attn1.to_out | up | 36 | 166.3 | 2.9 | 2% | compute | 163.4 |
| linear | attn2.to_q | up | 36 | 163.6 | 2.9 | 2% | compute | 160.7 |
| linear | attn2.to_v | up | 36 | 161.7 | 1.7 | 1% | memory | 160.0 |
| linear | attn1.to_k | up | 36 | 162.9 | 2.9 | 2% | compute | 160.0 |
| linear | attn2.to_k | up | 36 | 160.9 | 1.7 | 1% | memory | 159.2 |
| linear | ff.net | up | 36 | 168.7 | 11.6 | 7% | compute | 157.0 |
| linear | attn1.to_v | up | 36 | 157.3 | 2.9 | 2% | compute | 154.4 |
| linear | net.proj | up | 36 | 175.2 | 23.3 | 13% | compute | 151.9 |
| conv | input_blocks.in_layers | down | 6 | 149.6 | 3.0 | 2% | compute | 146.6 |
| linear | attn2.to_v | down | 24 | 121.4 | 1.1 | 1% | memory | 120.3 |
| linear | attn2.to_k | down | 24 | 120.9 | 1.1 | 1% | memory | 119.8 |
| linear | attn1.to_q | down | 24 | 121.6 | 1.9 | 2% | compute | 119.7 |
| linear | attn2.to_q | down | 24 | 121.6 | 1.9 | 2% | compute | 119.6 |
| linear | attn2.to_out | down | 24 | 120.7 | 1.9 | 2% | compute | 118.8 |
| linear | attn1.to_v | down | 24 | 119.1 | 1.9 | 2% | compute | 117.1 |
| linear | attn1.to_out | down | 24 | 118.3 | 1.9 | 2% | compute | 116.3 |
| linear | attn1.to_k | down | 24 | 117.7 | 1.9 | 2% | compute | 115.8 |
| linear | ff.net | down | 24 | 121.9 | 7.8 | 6% | compute | 114.2 |
| linear | net.proj | down | 24 | 129.6 | 15.5 | 12% | compute | 114.1 |
| feedforward | transformer_blocks.ff | up | 36 | 130.4 | 22.0 | 17% | memory | 108.3 |
| feedforward | transformer_blocks.ff | down | 24 | 113.2 | 14.7 | 13% | memory | 98.5 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.6.0.op | conv | 1 | 33.1 | 0.0 | 120.9 | 154.0 | 0.1 | 0% |
| output_blocks.3.0.in_layers.2 | conv | 1 | 40.2 | 1.7 | 27.2 | 69.1 | 2.2 | 3% |
| output_blocks.4.0.in_layers.2 | conv | 1 | 39.5 | 1.1 | 27.6 | 68.3 | 1.5 | 2% |
| output_blocks.6.0.in_layers.2 | conv | 1 | 37.6 | 1.4 | 28.3 | 67.3 | 2.2 | 3% |
| output_blocks.5.0.in_layers.2 | conv | 1 | 39.3 | 0.8 | 25.5 | 65.6 | 1.1 | 2% |
| input_blocks.7.0.in_layers.2 | conv | 1 | 37.8 | 1.1 | 26.3 | 65.2 | 0.4 | 1% |
| output_blocks.2.0.in_layers.2 | conv | 1 | 38.6 | 1.5 | 24.8 | 65.0 | 1.1 | 2% |
| output_blocks.5.2.conv | conv | 1 | 25.6 | 0.0 | 26.3 | 52.0 | 1.9 | 4% |
| output_blocks.2.0.skip_connection | conv | 1 | 29.2 | 0.7 | 20.7 | 50.7 | 0.1 | 0% |
| input_blocks.7.0.skip_connection | conv | 1 | 29.1 | 0.9 | 19.5 | 49.5 | 0.0 | 0% |
| output_blocks.5.0.skip_connection | conv | 1 | 20.6 | 0.6 | 22.7 | 43.9 | 0.1 | 0% |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 15.9 | 0.0 | 75.6 | 43.7 | - | - |
| out.2 | conv | 1 | 12.1 | 0.0 | 30.7 | 42.8 | 0.0 | 0% |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 21.5 | 0.0 | 74.6 | 42.4 | - | - |
| input_blocks.3.0.op | conv | 1 | 42.3 | 0.0 | 0.0 | 42.3 | 0.1 | 0% |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 16.5 | 0.0 | 73.6 | 42.2 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 22.3 | 0.0 | 73.9 | 42.1 | - | - |
| output_blocks.1.1.transformer_blocks.6 | block | 1 | 18.4 | 0.0 | 66.8 | 42.1 | - | - |
| output_blocks.0.0.in_layers.2 | conv | 1 | 38.1 | 1.9 | 1.9 | 41.9 | 1.5 | 3% |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 17.3 | 0.0 | 72.6 | 41.9 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 16.3 | 0.0 | 74.6 | 41.9 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 16.2 | 0.0 | 74.0 | 41.9 | - | - |
| output_blocks.4.0.skip_connection | conv | 1 | 20.7 | 0.6 | 20.5 | 41.8 | 0.2 | 0% |
| output_blocks.3.0.skip_connection | conv | 1 | 20.6 | 0.6 | 20.6 | 41.7 | 0.2 | 1% |
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 16.3 | 0.0 | 73.6 | 41.6 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 16.7 | 0.0 | 73.2 | 41.5 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 16.4 | 0.0 | 73.8 | 41.3 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 16.9 | 0.0 | 73.4 | 41.2 | - | - |
| input_blocks.4.0.out_layers.3 | conv | 1 | 40.9 | 0.0 | 0.0 | 40.9 | 0.5 | 1% |
| input_blocks.7.0.out_layers.3 | conv | 1 | 38.2 | 1.2 | 1.4 | 40.8 | 0.7 | 2% |

## Reconciliation

device 614 ms; modules' own time 9431 ms, of which 2743 ms is in modules with no cost model; wall 8442 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 216.94 | 27 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 67.17 | 31 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2141 | 60.11 | 28 |
| `gen_conv` | convolution | 131 | 58.83 | 449 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 49.42 | 16 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 32.28 | 231 |
| `conv_reorder` | convolution | 262 | 13.87 | 53 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 13.52 | 64 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 12.37 | 177 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 8.89 | 21 |
| `micro_sdpa` | attention | 140 | 8.38 | 60 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 6.92 | 49 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.71 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 5.90 | 42 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.67 | 81 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 4.97 | 71 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.53 | 22 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.25 | 2 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 3.13 | 95 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.80 | 38 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 2.68 | 36 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 210 | 2.10 | 10 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 2.09 | 15 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 39 | 1.91 | 49 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::SiluFunctor<c10::BFloat16` | elementwise | 96 | 1.89 | 20 |
