# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- step 1607 ms, median of 3 unprofiled steps (best 1603); eager 1607 ms, replayed - ms
- the profiled window's steps took 8184 ms each (hooks and profiler); device and module times come from them
- device busy 776 ms, utilisation 48%
- 38080 kernel launches, 232658 operators, 212 us host time per launch
- step peak memory 4251 MB
- phases: loss 0 ms

## Findings, largest first

1. **Much of the module time has no cost model** (2969 ms)
   - 2969 of 9517 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
2. **The step is launch bound** (831 ms)
   - the device is busy 48% of the step (776 of 1607 ms) over 38080 launches and 232658 operators per step, 212 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
3. **Slowest pieces by recoverable time: linear ff.net (up)** (246 ms)
   - 251.9 ms per step against an ideal 5.6 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 6379 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: linear attn2.to_k (up)** (240 ms)
   - 240.7 ms per step against an ideal 0.9 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 6379 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: linear attn2.to_v (up)** (239 ms)
   - 240.2 ms per step against an ideal 0.9 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 6379 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (210 ms)
   - 216.5 ms per step against an ideal 6.0 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 6379 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: linear attn2.to_q (up)** (210 ms)
   - 211.4 ms per step against an ideal 1.3 ms (1% efficient, memory-bound), 30 layers; all rated pieces together leave 6379 ms
   - lever: see the module table for the individual layers
8. **Dtype and copy kernels take a large share** (175 ms)
   - 22% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
9. **Many small memory-bound kernels** (168 ms)
   - 43% of device time (24268 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
10. **The int8 path's own kernels are a large share** (126 ms)
   - rotate, quantize and dequantize kernels are 16% of device time against 197 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows
11. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5906 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up
12. **Module times do not add up to the device time** (0 ms)
   - modules account for 9517 ms of 776 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 197.0 | 25% |
| copy | 5764 | 174.7 | 22% |
| elementwise | 9693 | 132.4 | 17% |
| int8 | 4487 | 126.0 | 16% |
| convolution | 512 | 62.4 | 8% |
| other | 610 | 45.8 | 6% |
| norm | 782 | 20.2 | 3% |
| attention | 140 | 9.4 | 1% |
| memory | 8011 | 7.2 | 1% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 104.6 TFLOPS, int8 199.4 TOPS, copy 386.2 GB/s, dispatch 10.5 us, queued launch 10.5 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| linear int8 | ff.net | up | 36 | 251.9 | 5.6 | 2% | compute | 246.2 |
| linear int8 | attn2.to_k | up | 36 | 240.7 | 0.9 | 0% | memory | 239.8 |
| linear int8 | attn2.to_v | up | 36 | 240.2 | 0.9 | 0% | memory | 239.3 |
| attention | transformer_blocks.attn1 | up | 36 | 216.5 | 6.0 | 3% | compute | 210.4 |
| linear int8 | attn2.to_q | up | 30 | 211.4 | 1.3 | 1% | memory | 210.1 |
| linear int8 | attn2.to_out | up | 30 | 210.6 | 1.3 | 1% | memory | 209.3 |
| linear int8 | attn1.to_out | up | 30 | 210.3 | 1.3 | 1% | memory | 209.0 |
| linear int8 | net.proj | up | 30 | 217.4 | 9.3 | 4% | compute | 208.1 |
| linear int8 | attn1.to_q | up | 30 | 208.3 | 1.3 | 1% | memory | 207.0 |
| linear int8 | attn1.to_k | up | 30 | 202.7 | 1.3 | 1% | memory | 201.4 |
| linear int8 | attn1.to_v | up | 30 | 201.1 | 1.3 | 1% | memory | 199.8 |
| attention | transformer_blocks.attn2 | up | 36 | 204.5 | 6.0 | 3% | compute | 198.5 |
| linear int8 | ff.net | down | 24 | 188.7 | 3.8 | 2% | compute | 184.9 |
| attention | transformer_blocks.attn1 | down | 24 | 182.6 | 4.0 | 2% | compute | 178.6 |
| linear int8 | attn2.to_v | down | 24 | 177.8 | 0.6 | 0% | memory | 177.2 |
| attention | transformer_blocks.attn2 | down | 24 | 173.3 | 4.0 | 2% | compute | 169.3 |
| linear int8 | attn2.to_k | down | 24 | 169.8 | 0.6 | 0% | memory | 169.2 |
| linear int8 | attn1.to_out | down | 20 | 147.5 | 0.9 | 1% | memory | 146.6 |
| linear int8 | attn2.to_out | down | 20 | 146.6 | 0.9 | 1% | memory | 145.8 |
| linear int8 | net.proj | down | 20 | 151.6 | 6.2 | 4% | compute | 145.4 |
| linear int8 | attn1.to_q | down | 20 | 144.4 | 0.9 | 1% | memory | 143.5 |
| linear int8 | attn1.to_v | down | 20 | 142.9 | 0.9 | 1% | memory | 142.0 |
| linear int8 | attn2.to_q | down | 20 | 142.4 | 0.9 | 1% | memory | 141.5 |
| linear int8 | attn1.to_k | down | 20 | 141.4 | 0.9 | 1% | memory | 140.5 |
| feedforward | transformer_blocks.ff | up | 36 | 121.6 | 20.9 | 17% | memory | 100.7 |
| feedforward | transformer_blocks.ff | down | 24 | 102.7 | 13.9 | 14% | memory | 88.8 |
| linear int8 | attn1.to_out | mid | 10 | 72.2 | 0.4 | 1% | memory | 71.7 |
| linear int8 | net.proj | mid | 10 | 74.7 | 3.1 | 4% | compute | 71.6 |
| linear int8 | attn1.to_k | mid | 10 | 71.8 | 0.4 | 1% | memory | 71.4 |
| linear int8 | attn2.to_out | mid | 10 | 71.1 | 0.4 | 1% | memory | 70.6 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 17.2 | 0.0 | 83.5 | 49.3 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 17.5 | 0.0 | 80.1 | 47.3 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 22.8 | 0.0 | 88.4 | 46.0 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 22.4 | 0.0 | 87.6 | 45.7 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 22.8 | 0.0 | 86.2 | 44.6 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 22.9 | 0.0 | 85.1 | 44.4 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 22.3 | 0.0 | 85.3 | 44.2 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 23.1 | 0.0 | 85.6 | 44.1 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 23.4 | 0.0 | 84.2 | 43.8 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 23.6 | 0.0 | 83.9 | 43.7 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 22.5 | 0.0 | 86.0 | 43.7 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 22.7 | 0.0 | 83.7 | 43.0 | - | - |
| middle_block.1.transformer_blocks.6 | block | 1 | 22.6 | 0.0 | 82.6 | 42.9 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 22.4 | 0.0 | 81.9 | 42.5 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 22.3 | 0.0 | 81.8 | 42.4 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 22.6 | 0.0 | 81.6 | 42.2 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 24.2 | 0.0 | 80.9 | 42.2 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 22.1 | 0.0 | 81.2 | 41.8 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 17.3 | 0.0 | 76.9 | 41.7 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 23.5 | 0.0 | 80.6 | 41.6 | - | - |
| middle_block.1.transformer_blocks.7 | block | 1 | 23.2 | 0.0 | 79.4 | 41.5 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 23.4 | 0.0 | 80.2 | 41.5 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 22.2 | 0.0 | 81.9 | 41.3 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 22.7 | 0.0 | 80.2 | 41.1 | - | - |
| middle_block.1.transformer_blocks.5 | block | 1 | 22.6 | 0.0 | 79.5 | 41.1 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 23.7 | 0.0 | 80.8 | 41.0 | - | - |
| middle_block.1.transformer_blocks.3 | block | 1 | 22.6 | 0.0 | 79.2 | 40.7 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 17.7 | 0.0 | 77.7 | 40.7 | - | - |
| output_blocks.0.1.transformer_blocks.3 | block | 1 | 22.6 | 0.0 | 78.4 | 40.6 | - | - |
| middle_block.1.transformer_blocks.4 | block | 1 | 22.4 | 0.0 | 79.1 | 40.5 | - | - |

## Reconciliation

device 776 ms; modules' own time 9517 ms, of which 2969 ms is in modules with no cost model; wall 1607 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 189.41 | 24 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2789 | 77.39 | 28 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2653 | 75.13 | 28 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 61.34 | 20 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1904 | 54.47 | 29 |
| `gen_conv` | convolution | 131 | 47.48 | 362 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 40.46 | 289 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 846 | 28.37 | 34 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1272 | 27.90 | 22 |
| `(anonymous namespace)::launch_rotate_bf16<1>(sycl::_V1::queue&, unsigned short const*, syc` | int8 | 582 | 18.76 | 32 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 14.64 | 209 |
| `conv_reorder` | convolution | 241 | 13.29 | 55 |
| `(anonymous namespace)::launch_rotate_bf16<2>(sycl::_V1::queue&, unsigned short const*, syc` | int8 | 60 | 12.93 | 216 |
| `_quantize_row_kernel` | int8 | 642 | 10.18 | 16 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.61 | 23 |
| `micro_sdpa` | attention | 140 | 9.41 | 67 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 7.63 | 55 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 7.29 | 52 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.72 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.38 | 91 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::FillFunctor<float>, at::d` | memory | 7969 | 6.29 | 1 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.94 | 85 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 5.03 | 24 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.29 | 2 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 3.21 | 43 |
