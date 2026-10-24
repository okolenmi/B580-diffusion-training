# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- step 1371 ms, median of 3 unprofiled steps (best 1351); eager 1371 ms, replayed - ms
- the profiled window's steps took 7110 ms each (hooks and profiler); device and module times come from them
- device busy 794 ms, utilisation 58%
- 32938 kernel launches, 214712 operators, 213 us host time per launch
- step peak memory 4244 MB
- phases: loss 0 ms

## Findings, largest first

1. **Recomputing activations is a large part of the step** (3433 ms)
   - 3433 ms per step (16% of module time) is forward work done a second time by checkpointing
   - lever: keep more blocks un-checkpointed (trainer.checkpoint_by=measured with tune_checkpointing, attention_fraction) if memory allows
2. **Much of the module time has no cost model** (2775 ms)
   - 2775 of 8318 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
3. **The step is launch bound** (576 ms)
   - the device is busy 58% of the step (794 of 1371 ms) over 32938 launches and 214712 operators per step, 213 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
4. **Slowest pieces by recoverable time: linear ff.net (up)** (201 ms)
   - 206.3 ms per step against an ideal 5.6 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 5374 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: linear attn2.to_v (up)** (196 ms)
   - 197.0 ms per step against an ideal 0.9 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 5374 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: linear attn2.to_k (up)** (196 ms)
   - 196.5 ms per step against an ideal 0.9 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 5374 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (188 ms)
   - 194.0 ms per step against an ideal 6.0 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 5374 ms
   - lever: see the module table for the individual layers
8. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (174 ms)
   - 180.4 ms per step against an ideal 6.0 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 5374 ms
   - lever: see the module table for the individual layers
9. **Dtype and copy kernels take a large share** (168 ms)
   - 21% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
10. **Many small memory-bound kernels** (157 ms)
   - 40% of device time (20410 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
11. **The int8 path's own kernels are a large share** (148 ms)
   - rotate, quantize and dequantize kernels are 19% of device time against 213 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows
12. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5661 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up
13. **Module times do not add up to the device time** (0 ms)
   - modules account for 8318 ms of 794 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 213.5 | 27% |
| copy | 5116 | 168.2 | 21% |
| int8 | 3203 | 148.3 | 19% |
| elementwise | 9051 | 119.3 | 15% |
| convolution | 512 | 63.3 | 8% |
| other | 610 | 46.3 | 6% |
| norm | 782 | 20.1 | 3% |
| attention | 140 | 9.3 | 1% |
| memory | 5443 | 4.9 | 1% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 104.6 TFLOPS, int8 199.4 TOPS, copy 386.2 GB/s, dispatch 10.5 us, queued launch 10.5 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| linear int8 | ff.net | up | 36 | 206.3 | 5.6 | 3% | compute | 200.7 |
| linear int8 | attn2.to_v | up | 36 | 197.0 | 0.9 | 0% | memory | 196.1 |
| linear int8 | attn2.to_k | up | 36 | 196.5 | 0.9 | 0% | memory | 195.5 |
| attention | transformer_blocks.attn1 | up | 36 | 194.0 | 6.0 | 3% | compute | 187.9 |
| attention | transformer_blocks.attn2 | up | 36 | 180.4 | 6.0 | 3% | compute | 174.4 |
| linear int8 | attn1.to_out | up | 30 | 171.0 | 1.3 | 1% | memory | 169.7 |
| linear int8 | attn2.to_out | up | 30 | 170.6 | 1.3 | 1% | memory | 169.3 |
| linear int8 | net.proj | up | 30 | 176.7 | 9.3 | 5% | compute | 167.4 |
| linear int8 | attn1.to_q | up | 30 | 168.5 | 1.3 | 1% | memory | 167.2 |
| attention | transformer_blocks.attn1 | down | 24 | 170.9 | 4.0 | 2% | compute | 166.8 |
| linear int8 | attn1.to_v | up | 30 | 167.2 | 1.3 | 1% | memory | 165.9 |
| linear int8 | attn1.to_k | up | 30 | 167.0 | 1.3 | 1% | memory | 165.7 |
| linear int8 | attn2.to_q | up | 30 | 166.0 | 1.3 | 1% | memory | 164.7 |
| attention | transformer_blocks.attn2 | down | 24 | 155.5 | 4.0 | 3% | compute | 151.5 |
| linear int8 | ff.net | down | 24 | 147.3 | 3.8 | 3% | compute | 143.5 |
| linear int8 | attn2.to_k | down | 24 | 139.0 | 0.6 | 0% | memory | 138.3 |
| linear int8 | attn2.to_v | down | 24 | 138.9 | 0.6 | 0% | memory | 138.3 |
| linear int8 | net.proj | down | 20 | 130.6 | 6.2 | 5% | compute | 124.4 |
| linear int8 | attn1.to_out | down | 20 | 120.0 | 0.9 | 1% | memory | 119.1 |
| linear int8 | attn2.to_out | down | 20 | 119.7 | 0.9 | 1% | memory | 118.8 |
| linear int8 | attn1.to_q | down | 20 | 116.9 | 0.9 | 1% | memory | 116.0 |
| linear int8 | attn2.to_q | down | 20 | 116.4 | 0.9 | 1% | memory | 115.5 |
| linear int8 | attn1.to_v | down | 20 | 116.1 | 0.9 | 1% | memory | 115.2 |
| linear int8 | attn1.to_k | down | 20 | 115.1 | 0.9 | 1% | memory | 114.3 |
| feedforward | transformer_blocks.ff | up | 36 | 107.8 | 20.9 | 19% | memory | 86.9 |
| feedforward | transformer_blocks.ff | down | 24 | 92.8 | 13.9 | 15% | memory | 78.8 |
| linear int8 | ff.net | mid | 10 | 60.1 | 1.5 | 3% | compute | 58.6 |
| linear int8 | attn1.to_out | mid | 10 | 59.0 | 0.4 | 1% | memory | 58.5 |
| linear int8 | attn2.to_out | mid | 10 | 57.8 | 0.4 | 1% | memory | 57.3 |
| linear int8 | net.proj | mid | 10 | 60.2 | 3.1 | 5% | compute | 57.2 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 21.1 | 0.0 | 77.5 | 46.2 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 21.2 | 0.0 | 75.2 | 45.4 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 25.2 | 0.0 | 74.1 | 43.3 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 21.1 | 0.0 | 71.3 | 43.1 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 21.7 | 0.0 | 68.4 | 40.9 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 16.6 | 0.0 | 71.8 | 40.8 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 22.2 | 0.0 | 69.6 | 40.3 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 20.9 | 0.0 | 66.9 | 39.7 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 16.6 | 0.0 | 69.0 | 39.5 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 16.7 | 0.0 | 70.9 | 39.5 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 16.8 | 0.0 | 69.0 | 39.5 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 21.1 | 0.0 | 70.1 | 39.5 | - | - |
| middle_block.1.transformer_blocks.4 | block | 1 | 21.5 | 0.0 | 65.8 | 39.4 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 21.1 | 0.0 | 66.5 | 39.4 | - | - |
| output_blocks.1.1.transformer_blocks.0 | block | 1 | 22.6 | 0.0 | 63.5 | 39.3 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 31.9 | 0.0 | 66.7 | 39.1 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 21.6 | 0.0 | 65.7 | 39.1 | - | - |
| middle_block.1.transformer_blocks.7 | block | 1 | 21.6 | 0.0 | 64.5 | 38.9 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 21.0 | 0.0 | 66.4 | 38.8 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 23.4 | 0.0 | 68.4 | 38.8 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 20.9 | 0.0 | 67.1 | 38.7 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 21.8 | 0.0 | 65.3 | 38.5 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 21.9 | 0.0 | 65.6 | 38.2 | - | - |
| middle_block.1.transformer_blocks.0 | block | 1 | 22.0 | 0.0 | 64.6 | 38.2 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 21.4 | 0.0 | 64.0 | 38.0 | - | - |
| output_blocks.0.1.transformer_blocks.6 | block | 1 | 22.4 | 0.0 | 65.0 | 38.0 | - | - |
| middle_block.1.transformer_blocks.8 | block | 1 | 21.6 | 0.0 | 62.2 | 37.7 | - | - |
| output_blocks.0.1.transformer_blocks.5 | block | 1 | 23.2 | 0.0 | 62.6 | 37.6 | - | - |
| middle_block.1.transformer_blocks.3 | block | 1 | 21.4 | 0.0 | 62.7 | 37.6 | - | - |
| middle_block.1.transformer_blocks.6 | block | 1 | 22.0 | 0.0 | 63.1 | 37.6 | - | - |

## Reconciliation

device 794 ms; modules' own time 8318 ms, of which 2775 ms is in modules with no cost model; wall 1371 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 206.14 | 26 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 74.24 | 35 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2653 | 72.45 | 27 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 58.66 | 19 |
| `(anonymous namespace)::launch_dequant_weight<1>(sycl::_V1::queue&, signed char const*, flo` | int8 | 599 | 57.50 | 96 |
| `gen_conv` | convolution | 131 | 47.97 | 366 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 40.51 | 289 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1265 | 39.52 | 31 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1272 | 27.39 | 22 |
| `(anonymous namespace)::launch_dequant_weight<2>(sycl::_V1::queue&, signed char const*, flo` | int8 | 60 | 23.69 | 395 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 17.92 | 85 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 14.55 | 208 |
| `conv_reorder` | convolution | 241 | 13.73 | 57 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.52 | 23 |
| `micro_sdpa` | attention | 140 | 9.27 | 66 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 7.33 | 52 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 7.27 | 52 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.52 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.52 | 93 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.07 | 87 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.88 | 23 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::FillFunctor<float>, at::d` | memory | 5401 | 4.09 | 1 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.42 | 2 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 3.23 | 44 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 2.94 | 89 |
