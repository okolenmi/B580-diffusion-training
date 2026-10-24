# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- step 2109 ms, median of 3 unprofiled steps (best 2101); eager 2109 ms, replayed - ms
- the profiled window's steps took 11062 ms each (hooks and profiler); device and module times come from them
- device busy 777 ms, utilisation 37%
- 49315 kernel launches, 336300 operators, 203 us host time per launch
- step peak memory 6437 MB
- phases: loss 0 ms

## Findings, largest first

1. **Much of the module time has no cost model** (2448 ms)
   - 2448 of 11231 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
2. **The step is launch bound** (1333 ms)
   - the device is busy 37% of the step (777 of 2109 ms) over 49315 launches and 336300 operators per step, 203 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
3. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (404 ms)
   - 409.8 ms per step against an ideal 5.6 ms (1% efficient, compute-bound), 36 layers; all rated pieces together leave 8631 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: linear ff.net (up)** (340 ms)
   - 345.3 ms per step against an ideal 5.1 ms (1% efficient, memory-bound), 36 layers; all rated pieces together leave 8631 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (307 ms)
   - 312.1 ms per step against an ideal 5.6 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 8631 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (down)** (277 ms)
   - 281.0 ms per step against an ideal 3.7 ms (1% efficient, compute-bound), 24 layers; all rated pieces together leave 8631 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: linear net.proj (up)** (273 ms)
   - 280.4 ms per step against an ideal 7.7 ms (3% efficient, memory-bound), 30 layers; all rated pieces together leave 8631 ms
   - lever: see the module table for the individual layers
8. **The int8 path's own kernels are a large share** (183 ms)
   - rotate, quantize and dequantize kernels are 24% of device time against 208 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows
9. **Dtype and copy kernels take a large share** (147 ms)
   - 19% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
10. **Many small memory-bound kernels** (136 ms)
   - 35% of device time (29672 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
11. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 6361 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up
12. **Module times do not add up to the device time** (0 ms)
   - modules account for 11231 ms of 777 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 13034 | 208.2 | 27% |
| int8 | 4412 | 182.6 | 24% |
| copy | 7812 | 147.3 | 19% |
| elementwise | 13841 | 98.0 | 13% |
| convolution | 910 | 62.7 | 8% |
| other | 1187 | 45.7 | 6% |
| norm | 1262 | 19.4 | 2% |
| attention | 100 | 5.8 | 1% |
| memory | 6738 | 5.6 | 1% |
| reduce | 19 | 1.4 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 104.6 TFLOPS, int8 199.4 TOPS, copy 386.2 GB/s, dispatch 10.5 us, queued launch 10.5 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention | transformer_blocks.attn2 | up | 36 | 409.8 | 5.6 | 1% | compute | 404.3 |
| linear int8 | ff.net | up | 36 | 345.3 | 5.1 | 1% | memory | 340.3 |
| attention | transformer_blocks.attn1 | up | 36 | 312.1 | 5.6 | 2% | compute | 306.5 |
| attention | transformer_blocks.attn2 | down | 24 | 281.0 | 3.7 | 1% | compute | 277.3 |
| linear int8 | net.proj | up | 30 | 280.4 | 7.7 | 3% | memory | 272.8 |
| linear int8 | attn2.to_out | up | 30 | 270.4 | 1.3 | 0% | memory | 269.1 |
| linear int8 | attn1.to_out | up | 30 | 270.3 | 1.3 | 0% | memory | 269.0 |
| linear int8 | attn1.to_q | up | 30 | 263.7 | 1.3 | 0% | memory | 262.5 |
| linear int8 | attn2.to_q | up | 30 | 262.2 | 1.3 | 0% | memory | 260.9 |
| linear int8 | attn1.to_v | up | 30 | 261.7 | 1.3 | 0% | memory | 260.5 |
| linear int8 | attn1.to_k | up | 30 | 260.5 | 1.3 | 0% | memory | 259.3 |
| linear int8 | attn2.to_k | up | 36 | 253.4 | 1.2 | 0% | memory | 252.1 |
| linear int8 | attn2.to_v | up | 36 | 241.8 | 1.2 | 1% | memory | 240.6 |
| linear int8 | ff.net | down | 24 | 241.1 | 3.5 | 1% | memory | 237.6 |
| attention | transformer_blocks.attn1 | down | 24 | 233.3 | 3.7 | 2% | compute | 229.5 |
| linear int8 | attn1.to_out | down | 20 | 188.4 | 0.9 | 0% | memory | 187.5 |
| linear int8 | attn2.to_out | down | 20 | 188.4 | 0.9 | 0% | memory | 187.5 |
| linear int8 | net.proj | down | 20 | 190.0 | 5.3 | 3% | memory | 184.8 |
| linear int8 | attn1.to_q | down | 20 | 183.3 | 0.9 | 0% | memory | 182.4 |
| linear int8 | attn2.to_q | down | 20 | 183.2 | 0.9 | 0% | memory | 182.3 |
| linear int8 | attn1.to_v | down | 20 | 181.7 | 0.9 | 0% | memory | 180.8 |
| linear int8 | attn1.to_k | down | 20 | 180.8 | 0.9 | 0% | memory | 180.0 |
| linear int8 | attn2.to_k | down | 24 | 175.4 | 0.9 | 0% | memory | 174.6 |
| linear int8 | attn2.to_v | down | 24 | 173.9 | 0.9 | 0% | memory | 173.0 |
| feedforward | transformer_blocks.ff | up | 36 | 184.1 | 18.0 | 10% | memory | 166.1 |
| feedforward | transformer_blocks.ff | down | 24 | 137.2 | 12.2 | 9% | memory | 125.0 |
| linear int8 | ff.net | mid | 10 | 117.3 | 1.3 | 1% | memory | 116.0 |
| attention | transformer_blocks.attn2 | mid | 10 | 111.2 | 0.8 | 1% | memory | 110.4 |
| linear int8 | attn1.to_out | mid | 10 | 89.4 | 0.4 | 0% | memory | 89.0 |
| linear int8 | net.proj | mid | 10 | 91.4 | 2.5 | 3% | memory | 88.9 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.7.1.transformer_blocks.5 | block | 2 | 45.0 | 0.0 | 133.8 | 80.1 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 2 | 46.0 | 0.0 | 132.6 | 79.2 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 2 | 46.0 | 0.0 | 133.8 | 79.1 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 2 | 45.0 | 0.0 | 133.3 | 78.6 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 2 | 45.7 | 0.0 | 128.8 | 76.1 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 2 | 45.7 | 0.0 | 128.6 | 76.0 | - | - |
| output_blocks.0.1.transformer_blocks.5 | block | 2 | 47.8 | 0.0 | 125.5 | 75.6 | - | - |
| middle_block.1.transformer_blocks.4 | block | 2 | 45.8 | 0.0 | 125.9 | 75.5 | - | - |
| middle_block.1.transformer_blocks.2 | block | 2 | 46.0 | 0.0 | 126.4 | 74.5 | - | - |
| output_blocks.0.1.transformer_blocks.4 | block | 2 | 48.0 | 0.0 | 125.2 | 74.2 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 2 | 37.3 | 0.0 | 125.6 | 72.5 | - | - |
| output_blocks.0.1.transformer_blocks.1 | block | 2 | 46.9 | 0.0 | 122.2 | 72.2 | - | - |
| output_blocks.0.1.transformer_blocks.6 | block | 2 | 48.5 | 0.0 | 131.2 | 71.9 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 2 | 37.6 | 0.0 | 128.4 | 71.3 | - | - |
| output_blocks.1.1.transformer_blocks.8 | block | 2 | 48.6 | 0.0 | 119.2 | 71.2 | - | - |
| output_blocks.2.1.transformer_blocks.3 | block | 2 | 51.6 | 0.0 | 117.5 | 70.6 | - | - |
| output_blocks.2.1.transformer_blocks.4 | block | 2 | 51.0 | 0.0 | 114.3 | 68.7 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 2 | 36.8 | 0.0 | 121.2 | 68.2 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 2 | 35.7 | 0.0 | 126.9 | 67.8 | - | - |
| output_blocks.3.1.transformer_blocks.1 | block | 2 | 41.6 | 0.0 | 105.5 | 60.6 | - | - |
| output_blocks.4.1.transformer_blocks.0 | block | 2 | 42.1 | 0.0 | 104.5 | 60.3 | - | - |
| output_blocks.3.1.transformer_blocks.0 | block | 2 | 41.3 | 0.0 | 103.2 | 59.9 | - | - |
| output_blocks.4.1.transformer_blocks.1 | block | 2 | 63.5 | 0.0 | 103.2 | 59.2 | - | - |
| output_blocks.5.1.transformer_blocks.1 | block | 2 | 41.2 | 0.0 | 102.6 | 58.7 | - | - |
| output_blocks.5.1.transformer_blocks.0 | block | 2 | 41.5 | 0.0 | 102.1 | 58.4 | - | - |
| middle_block.1.transformer_blocks.1.ff.net.2 | linear | 2 | 30.7 | 0.0 | 4.6 | 35.3 | 0.1 | 0% |
| output_blocks.4.1.transformer_blocks.1.attn2 | attention | 2 | 37.3 | 16.9 | 18.0 | 32.6 | 0.5 | 2% |
| output_blocks.4.0 | block | 2 | 10.3 | 0.0 | 26.3 | 21.6 | - | - |
| output_blocks.0.1.transformer_blocks.6.attn2.to_k | linear | 2 | 3.5 | 3.9 | 13.5 | 20.9 | 0.0 | 0% |
| output_blocks.5.0 | block | 2 | 9.9 | 0.0 | 25.6 | 20.7 | - | - |

## Reconciliation

device 777 ms; modules' own time 11231 ms, of which 2448 ms is in modules with no cost model; wall 2109 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 12754 | 198.30 | 16 |
| `(anonymous namespace)::launch_dequant_weight<1>(sycl::_V1::queue&, signed char const*, flo` | int8 | 1004 | 90.24 | 90 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 4054 | 65.85 | 16 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 3214 | 61.53 | 19 |
| `(anonymous namespace)::launch_dequant_weight<2>(sycl::_V1::queue&, signed char const*, flo` | int8 | 120 | 47.43 | 395 |
| `gen_conv` | convolution | 222 | 44.73 | 201 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 4654 | 43.64 | 9 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 280 | 39.74 | 142 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1631 | 25.09 | 15 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1644 | 19.65 | 12 |
| `conv_reorder` | convolution | 408 | 16.93 | 41 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 140 | 13.92 | 99 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 330 | 10.16 | 31 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 4907 | 8.29 | 2 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 570 | 7.92 | 14 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 200 | 6.45 | 32 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 140 | 6.40 | 46 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 420 | 6.16 | 15 |
| `micro_sdpa` | attention | 100 | 5.76 | 58 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 2912 | 5.09 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 140 | 5.00 | 36 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 190 | 4.99 | 26 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::FillFunctor<float>, at::d` | memory | 6658 | 4.71 | 1 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 116 | 4.53 | 39 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 80 | 3.50 | 44 |
