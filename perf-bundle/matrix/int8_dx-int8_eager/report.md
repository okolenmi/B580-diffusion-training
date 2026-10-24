# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- step 1500 ms, median of 3 unprofiled steps (best 1496); eager 1500 ms, replayed - ms
- the profiled window's steps took 7395 ms each (hooks and profiler); device and module times come from them
- device busy 758 ms, utilisation 51%
- 38080 kernel launches, 232658 operators, 191 us host time per launch
- step peak memory 4251 MB
- phases: loss 0 ms

## Findings, largest first

1. **Much of the module time has no cost model** (2623 ms)
   - 2623 of 8489 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
2. **The step is launch bound** (741 ms)
   - the device is busy 51% of the step (758 of 1500 ms) over 38080 launches and 232658 operators per step, 191 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
3. **Slowest pieces by recoverable time: linear ff.net (up)** (224 ms)
   - 229.4 ms per step against an ideal 5.6 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 5695 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: linear attn2.to_k (up)** (218 ms)
   - 218.7 ms per step against an ideal 0.9 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 5695 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: linear attn2.to_v (up)** (217 ms)
   - 218.0 ms per step against an ideal 0.9 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 5695 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: linear attn1.to_out (up)** (191 ms)
   - 192.4 ms per step against an ideal 1.3 ms (1% efficient, memory-bound), 30 layers; all rated pieces together leave 5695 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: linear attn2.to_out (up)** (190 ms)
   - 191.7 ms per step against an ideal 1.3 ms (1% efficient, memory-bound), 30 layers; all rated pieces together leave 5695 ms
   - lever: see the module table for the individual layers
8. **Dtype and copy kernels take a large share** (165 ms)
   - 22% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
9. **Many small memory-bound kernels** (161 ms)
   - 43% of device time (24268 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
10. **The int8 path's own kernels are a large share** (121 ms)
   - rotate, quantize and dequantize kernels are 16% of device time against 200 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows
11. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5580 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up
12. **Module times do not add up to the device time** (0 ms)
   - modules account for 8489 ms of 758 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 200.0 | 26% |
| copy | 5764 | 165.2 | 22% |
| elementwise | 9693 | 130.4 | 17% |
| int8 | 4487 | 120.6 | 16% |
| convolution | 512 | 61.4 | 8% |
| other | 610 | 44.5 | 6% |
| norm | 782 | 19.3 | 3% |
| attention | 140 | 8.8 | 1% |
| memory | 8011 | 6.4 | 1% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 97.7 TFLOPS, int8 200.6 TOPS, copy 396.5 GB/s, dispatch 10.3 us, queued launch 10.3 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| linear int8 | ff.net | up | 36 | 229.4 | 5.6 | 2% | compute | 223.8 |
| linear int8 | attn2.to_k | up | 36 | 218.7 | 0.9 | 0% | memory | 217.8 |
| linear int8 | attn2.to_v | up | 36 | 218.0 | 0.9 | 0% | memory | 217.1 |
| linear int8 | attn1.to_out | up | 30 | 192.4 | 1.3 | 1% | memory | 191.1 |
| linear int8 | attn2.to_out | up | 30 | 191.7 | 1.3 | 1% | memory | 190.5 |
| linear int8 | attn2.to_q | up | 30 | 189.3 | 1.3 | 1% | memory | 188.0 |
| linear int8 | net.proj | up | 30 | 196.8 | 9.2 | 5% | compute | 187.6 |
| linear int8 | attn1.to_q | up | 30 | 187.8 | 1.3 | 1% | memory | 186.5 |
| linear int8 | attn1.to_k | up | 30 | 183.7 | 1.3 | 1% | memory | 182.4 |
| linear int8 | attn1.to_v | up | 30 | 182.5 | 1.3 | 1% | memory | 181.2 |
| attention | transformer_blocks.attn1 | up | 36 | 183.9 | 6.2 | 3% | compute | 177.8 |
| linear int8 | ff.net | down | 24 | 171.4 | 3.7 | 2% | compute | 167.7 |
| attention | transformer_blocks.attn2 | up | 36 | 171.7 | 6.2 | 4% | compute | 165.5 |
| linear int8 | attn2.to_k | down | 24 | 154.0 | 0.6 | 0% | memory | 153.4 |
| linear int8 | attn2.to_v | down | 24 | 153.5 | 0.6 | 0% | memory | 152.8 |
| attention | transformer_blocks.attn1 | down | 24 | 149.6 | 4.1 | 3% | compute | 145.5 |
| attention | transformer_blocks.attn2 | down | 24 | 141.9 | 4.1 | 3% | compute | 137.8 |
| linear int8 | attn1.to_out | down | 20 | 134.0 | 0.9 | 1% | memory | 133.1 |
| linear int8 | attn2.to_out | down | 20 | 133.6 | 0.9 | 1% | memory | 132.8 |
| linear int8 | net.proj | down | 20 | 137.9 | 6.1 | 4% | compute | 131.7 |
| linear int8 | attn1.to_q | down | 20 | 130.8 | 0.9 | 1% | memory | 129.9 |
| linear int8 | attn2.to_q | down | 20 | 129.6 | 0.9 | 1% | memory | 128.7 |
| linear int8 | attn1.to_v | down | 20 | 129.1 | 0.9 | 1% | memory | 128.3 |
| linear int8 | attn1.to_k | down | 20 | 129.0 | 0.9 | 1% | memory | 128.1 |
| feedforward | transformer_blocks.ff | up | 36 | 102.8 | 20.4 | 20% | memory | 82.4 |
| feedforward | transformer_blocks.ff | down | 24 | 84.7 | 13.6 | 16% | memory | 71.1 |
| linear int8 | attn1.to_q | mid | 10 | 66.2 | 0.4 | 1% | memory | 65.8 |
| linear int8 | attn1.to_out | mid | 10 | 65.8 | 0.4 | 1% | memory | 65.4 |
| linear int8 | ff.net | mid | 10 | 66.6 | 1.5 | 2% | compute | 65.1 |
| linear int8 | attn2.to_out | mid | 10 | 65.5 | 0.4 | 1% | memory | 65.0 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 16.2 | 0.0 | 69.5 | 41.1 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 20.5 | 0.0 | 75.7 | 39.1 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 20.7 | 0.0 | 75.5 | 38.9 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 21.4 | 0.0 | 75.1 | 38.9 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 21.2 | 0.0 | 74.5 | 38.6 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 21.3 | 0.0 | 74.4 | 38.5 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 21.0 | 0.0 | 74.7 | 38.5 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 21.2 | 0.0 | 74.3 | 38.3 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 21.2 | 0.0 | 73.9 | 38.2 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 21.3 | 0.0 | 73.8 | 38.1 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 21.6 | 0.0 | 73.4 | 37.9 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 21.3 | 0.0 | 73.0 | 37.8 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 21.4 | 0.0 | 72.5 | 37.3 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 21.5 | 0.0 | 72.1 | 37.3 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 21.3 | 0.0 | 72.2 | 37.2 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 21.5 | 0.0 | 71.9 | 37.2 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 21.7 | 0.0 | 73.1 | 37.0 | - | - |
| middle_block.1.transformer_blocks.0 | block | 1 | 22.1 | 0.0 | 71.8 | 37.0 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 21.4 | 0.0 | 71.9 | 37.0 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 22.0 | 0.0 | 71.2 | 36.8 | - | - |
| middle_block.1.transformer_blocks.1 | block | 1 | 21.6 | 0.0 | 71.4 | 36.8 | - | - |
| middle_block.1.transformer_blocks.4 | block | 1 | 21.7 | 0.0 | 71.2 | 36.8 | - | - |
| middle_block.1.transformer_blocks.3 | block | 1 | 22.0 | 0.0 | 71.0 | 36.7 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 21.4 | 0.0 | 71.2 | 36.6 | - | - |
| middle_block.1.transformer_blocks.6 | block | 1 | 21.8 | 0.0 | 70.7 | 36.6 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 21.5 | 0.0 | 71.0 | 36.5 | - | - |
| middle_block.1.transformer_blocks.7 | block | 1 | 21.9 | 0.0 | 70.1 | 36.3 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 21.6 | 0.0 | 70.5 | 36.2 | - | - |
| middle_block.1.transformer_blocks.5 | block | 1 | 21.7 | 0.0 | 70.5 | 36.2 | - | - |
| middle_block.1.transformer_blocks.9 | block | 1 | 21.9 | 0.0 | 67.1 | 35.0 | - | - |

## Reconciliation

device 758 ms; modules' own time 8489 ms, of which 2623 ms is in modules with no cost model; wall 1500 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 192.71 | 24 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2789 | 74.54 | 27 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2653 | 70.02 | 26 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 60.47 | 20 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1904 | 51.33 | 27 |
| `gen_conv` | convolution | 131 | 46.02 | 351 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 39.18 | 280 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 846 | 28.22 | 33 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1272 | 25.58 | 20 |
| `(anonymous namespace)::launch_rotate_bf16<1>(sycl::_V1::queue&, unsigned short const*, syc` | int8 | 582 | 19.32 | 33 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 14.34 | 205 |
| `conv_reorder` | convolution | 241 | 13.83 | 57 |
| `(anonymous namespace)::launch_rotate_bf16<2>(sycl::_V1::queue&, unsigned short const*, syc` | int8 | 60 | 12.80 | 213 |
| `_quantize_row_kernel` | int8 | 642 | 9.83 | 15 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.21 | 22 |
| `micro_sdpa` | attention | 140 | 8.77 | 63 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 7.30 | 52 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 6.88 | 49 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.33 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.23 | 89 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.97 | 85 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::FillFunctor<float>, at::d` | memory | 7969 | 5.68 | 1 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.78 | 23 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.41 | 2 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 3.06 | 41 |
