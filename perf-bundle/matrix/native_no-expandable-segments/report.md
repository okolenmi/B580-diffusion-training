# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- wall 7399 ms (best 5901); eager 7399 ms, replayed - ms
- device busy 603 ms, utilisation 8%
- 25434 kernel launches, 541454 operators, 288 us host time per launch
- step peak memory 6244 MB
- phases: loss 0 ms

## Findings, largest first

1. **The step is launch bound** (6796 ms)
   - the device is busy 8% of the step (603 of 7399 ms) over 25434 launches and 541454 operators per step, 288 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
2. **Much of the module time has no cost model** (2308 ms)
   - 2308 of 8243 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
3. **Slowest pieces by recoverable time: conv output_blocks.in_layers (up)** (436 ms)
   - 449.5 ms per step against an ideal 13.8 ms (3% efficient, compute-bound), 9 layers; all rated pieces together leave 5708 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: conv output_blocks.skip_connection (up)** (297 ms)
   - 299.1 ms per step against an ideal 1.6 ms (1% efficient, compute-bound), 9 layers; all rated pieces together leave 5708 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: conv input_blocks.op (down)** (186 ms)
   - 186.7 ms per step against an ideal 0.2 ms (0% efficient, compute-bound), 2 layers; all rated pieces together leave 5708 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (185 ms)
   - 192.0 ms per step against an ideal 6.5 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 5708 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (177 ms)
   - 183.9 ms per step against an ideal 6.5 ms (4% efficient, compute-bound), 36 layers; all rated pieces together leave 5708 ms
   - lever: see the module table for the individual layers
8. **Dtype and copy kernels take a large share** (145 ms)
   - 24% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
9. **Many small memory-bound kernels** (133 ms)
   - 44% of device time (16075 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
10. **Module times do not add up to the device time** (0 ms)
   - modules account for 8243 ms of 603 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 225.1 | 37% |
| copy | 4610 | 144.9 | 24% |
| elementwise | 9038 | 100.0 | 17% |
| convolution | 533 | 71.2 | 12% |
| other | 623 | 32.5 | 5% |
| norm | 782 | 17.4 | 3% |
| attention | 140 | 8.2 | 1% |
| memory | 1627 | 2.0 | 0% |
| reduce | 18 | 1.4 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 95.1 TFLOPS, int8 184.9 TOPS, copy 366.2 GB/s, dispatch 11.0 us, queued launch 11.0 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| conv | output_blocks.in_layers | up | 9 | 449.5 | 13.8 | 3% | compute | 435.7 |
| conv | output_blocks.skip_connection | up | 9 | 299.1 | 1.6 | 1% | compute | 297.4 |
| conv | input_blocks.op | down | 2 | 186.7 | 0.2 | 0% | compute | 186.4 |
| attention | transformer_blocks.attn1 | up | 36 | 192.0 | 6.5 | 3% | compute | 185.5 |
| attention | transformer_blocks.attn2 | up | 36 | 183.9 | 6.5 | 4% | compute | 177.3 |
| attention | transformer_blocks.attn1 | down | 24 | 162.8 | 4.3 | 3% | compute | 158.4 |
| attention | transformer_blocks.attn2 | down | 24 | 156.6 | 4.3 | 3% | compute | 152.2 |
| linear | attn2.to_q | up | 36 | 154.1 | 2.9 | 2% | compute | 151.2 |
| linear | attn2.to_v | up | 36 | 146.7 | 1.7 | 1% | memory | 144.9 |
| linear | attn1.to_q | up | 36 | 147.0 | 2.9 | 2% | compute | 144.1 |
| conv | input_blocks.in_layers | down | 6 | 145.5 | 3.0 | 2% | compute | 142.4 |
| linear | attn1.to_out | up | 36 | 145.2 | 2.9 | 2% | compute | 142.3 |
| linear | attn2.to_out | up | 36 | 144.0 | 2.9 | 2% | compute | 141.1 |
| linear | attn2.to_k | up | 36 | 141.6 | 1.7 | 1% | memory | 139.9 |
| linear | attn1.to_k | up | 36 | 142.5 | 2.9 | 2% | compute | 139.6 |
| linear | attn1.to_v | up | 36 | 136.7 | 2.9 | 2% | compute | 133.7 |
| linear | ff.net | up | 36 | 145.3 | 11.6 | 8% | compute | 133.6 |
| linear | net.proj | up | 36 | 154.1 | 23.3 | 15% | compute | 130.8 |
| linear | attn1.to_k | down | 24 | 107.0 | 1.9 | 2% | compute | 105.0 |
| linear | attn2.to_q | down | 24 | 106.3 | 1.9 | 2% | compute | 104.3 |
| linear | attn2.to_out | down | 24 | 105.6 | 1.9 | 2% | compute | 103.7 |
| linear | attn2.to_v | down | 24 | 103.7 | 1.1 | 1% | memory | 102.6 |
| linear | attn1.to_q | down | 24 | 104.5 | 1.9 | 2% | compute | 102.6 |
| linear | attn1.to_out | down | 24 | 104.5 | 1.9 | 2% | compute | 102.6 |
| linear | attn2.to_k | down | 24 | 103.0 | 1.1 | 1% | memory | 101.9 |
| linear | attn1.to_v | down | 24 | 102.2 | 1.9 | 2% | compute | 100.2 |
| linear | ff.net | down | 24 | 105.2 | 7.8 | 7% | compute | 97.5 |
| linear | net.proj | down | 24 | 111.2 | 15.5 | 14% | compute | 95.7 |
| feedforward | transformer_blocks.ff | up | 36 | 110.6 | 22.0 | 20% | memory | 88.5 |
| conv | output_blocks.conv | up | 2 | 90.4 | 3.9 | 4% | compute | 86.5 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.6.0.op | conv | 1 | 30.3 | 0.0 | 115.6 | 145.8 | 0.1 | 0% |
| output_blocks.3.0.in_layers.2 | conv | 1 | 39.2 | 1.3 | 27.3 | 67.9 | 2.2 | 3% |
| output_blocks.4.0.in_layers.2 | conv | 1 | 38.4 | 1.1 | 27.6 | 67.0 | 1.5 | 2% |
| output_blocks.6.0.in_layers.2 | conv | 1 | 36.7 | 1.1 | 26.8 | 64.6 | 2.2 | 3% |
| output_blocks.5.0.in_layers.2 | conv | 1 | 38.6 | 0.9 | 25.0 | 64.5 | 1.1 | 2% |
| output_blocks.2.0.in_layers.2 | conv | 1 | 36.8 | 1.4 | 25.9 | 64.1 | 1.1 | 2% |
| input_blocks.7.0.in_layers.2 | conv | 1 | 35.6 | 0.8 | 24.6 | 61.0 | 0.4 | 1% |
| output_blocks.5.2.conv | conv | 1 | 24.7 | 0.0 | 25.2 | 49.9 | 1.9 | 4% |
| output_blocks.2.0.skip_connection | conv | 1 | 27.6 | 0.5 | 21.3 | 49.4 | 0.1 | 0% |
| input_blocks.7.0.skip_connection | conv | 1 | 27.6 | 0.8 | 17.8 | 46.3 | 0.0 | 0% |
| output_blocks.5.0.skip_connection | conv | 1 | 19.9 | 0.5 | 22.2 | 42.6 | 0.1 | 0% |
| out.2 | conv | 1 | 11.9 | 0.0 | 30.5 | 42.4 | 0.0 | 0% |
| output_blocks.3.0.skip_connection | conv | 1 | 20.0 | 0.5 | 20.7 | 41.2 | 0.2 | 1% |
| output_blocks.4.0.skip_connection | conv | 1 | 20.2 | 0.5 | 20.4 | 41.1 | 0.2 | 0% |
| output_blocks.0.0.in_layers.2 | conv | 1 | 37.4 | 1.9 | 1.7 | 41.0 | 1.5 | 4% |
| input_blocks.3.0.op | conv | 1 | 40.8 | 0.0 | 0.0 | 40.8 | 0.1 | 0% |
| output_blocks.2.2.conv | conv | 1 | 20.0 | 0.0 | 20.5 | 40.5 | 1.9 | 5% |
| input_blocks.4.0.out_layers.3 | conv | 1 | 40.3 | 0.0 | 0.0 | 40.3 | 0.5 | 1% |
| input_blocks.4.0.in_layers.2 | conv | 1 | 40.1 | 0.0 | 0.0 | 40.1 | 0.2 | 1% |
| input_blocks.0.0 | conv | 1 | 38.8 | 0.0 | 0.0 | 38.8 | 0.0 | 0% |
| input_blocks.7.0.out_layers.3 | conv | 1 | 36.4 | 0.9 | 1.1 | 38.4 | 0.7 | 2% |
| input_blocks.1.0.in_layers.2 | conv | 1 | 38.4 | 0.0 | 0.0 | 38.4 | 0.5 | 1% |
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 15.4 | 0.0 | 63.5 | 35.6 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 15.5 | 0.0 | 61.5 | 35.3 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 20.5 | 0.0 | 63.5 | 35.3 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 16.4 | 0.0 | 62.8 | 35.1 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 14.9 | 0.0 | 60.7 | 35.1 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 15.7 | 0.0 | 62.2 | 34.9 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 14.9 | 0.0 | 65.1 | 34.7 | - | - |
| output_blocks.6.0.skip_connection | conv | 1 | 15.0 | 0.6 | 19.1 | 34.7 | 0.3 | 1% |

## Reconciliation

device 603 ms; modules' own time 8243 ms, of which 2308 ms is in modules with no cost model; wall 7399 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 218.37 | 28 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 67.31 | 31 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2141 | 59.89 | 28 |
| `gen_conv` | convolution | 131 | 56.19 | 429 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 48.06 | 16 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 28.14 | 201 |
| `conv_reorder` | convolution | 262 | 13.79 | 53 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 13.42 | 64 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 12.26 | 175 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 8.25 | 20 |
| `micro_sdpa` | attention | 140 | 8.24 | 59 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 7.02 | 2 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 6.75 | 48 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 5.87 | 42 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.38 | 77 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.13 | 73 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.36 | 21 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 2.98 | 2 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.71 | 37 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 2.70 | 36 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 39 | 2.03 | 52 |
| `at::native::xpu::GammaBetaBackwardSimpleKernelFunctor<c10::BFloat16, float, float, c10::BF` | other | 210 | 1.99 | 9 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::SiluFunctor<c10::BFloat16` | elementwise | 96 | 1.82 | 19 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 1.79 | 13 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 1.73 | 52 |
