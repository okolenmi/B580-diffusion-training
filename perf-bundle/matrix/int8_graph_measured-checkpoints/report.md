# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- step 2055 ms, median of 3 unprofiled steps (best 2049); eager 2055 ms, replayed - ms
- the profiled window's steps took 10276 ms each (hooks and profiler); device and module times come from them
- device busy 750 ms, utilisation 37%
- 49605 kernel launches, 337918 operators, 188 us host time per launch
- step peak memory 6381 MB
- phases: loss 0 ms

## Findings, largest first

1. **Much of the module time has no cost model** (2325 ms)
   - 2325 of 10476 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
2. **The step is launch bound** (1305 ms)
   - the device is busy 37% of the step (750 of 2055 ms) over 49605 launches and 337918 operators per step, 188 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
3. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (345 ms)
   - 350.6 ms per step against an ideal 5.7 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 7997 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: linear ff.net (up)** (331 ms)
   - 335.9 ms per step against an ideal 4.8 ms (1% efficient, memory-bound), 36 layers; all rated pieces together leave 7997 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (262 ms)
   - 268.1 ms per step against an ideal 5.7 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 7997 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (down)** (243 ms)
   - 246.4 ms per step against an ideal 3.9 ms (2% efficient, compute-bound), 24 layers; all rated pieces together leave 7997 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: linear net.proj (up)** (242 ms)
   - 249.2 ms per step against an ideal 7.1 ms (3% efficient, memory-bound), 30 layers; all rated pieces together leave 7997 ms
   - lever: see the module table for the individual layers
8. **The int8 path's own kernels are a large share** (167 ms)
   - rotate, quantize and dequantize kernels are 22% of device time against 211 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows
9. **Dtype and copy kernels take a large share** (138 ms)
   - 18% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
10. **Many small memory-bound kernels** (130 ms)
   - 35% of device time (29846 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
11. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 6200 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up
12. **Module times do not add up to the device time** (0 ms)
   - modules account for 10476 ms of 750 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 13102 | 211.1 | 28% |
| int8 | 4456 | 167.0 | 22% |
| copy | 7864 | 138.4 | 18% |
| elementwise | 13897 | 96.7 | 13% |
| convolution | 910 | 61.7 | 8% |
| other | 1187 | 45.3 | 6% |
| norm | 1268 | 18.2 | 2% |
| attention | 104 | 5.4 | 1% |
| memory | 6798 | 5.0 | 1% |
| reduce | 19 | 1.4 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 97.7 TFLOPS, int8 200.6 TOPS, copy 396.5 GB/s, dispatch 10.3 us, queued launch 10.3 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention | transformer_blocks.attn2 | up | 36 | 350.6 | 5.7 | 2% | compute | 344.9 |
| linear int8 | ff.net | up | 36 | 335.9 | 4.8 | 1% | memory | 331.1 |
| attention | transformer_blocks.attn1 | up | 36 | 268.1 | 5.7 | 2% | compute | 262.4 |
| attention | transformer_blocks.attn2 | down | 24 | 246.4 | 3.9 | 2% | compute | 242.5 |
| linear int8 | net.proj | up | 30 | 249.2 | 7.1 | 3% | memory | 242.1 |
| linear int8 | attn1.to_out | up | 30 | 242.2 | 1.2 | 0% | memory | 241.0 |
| linear int8 | attn2.to_out | up | 30 | 241.9 | 1.2 | 0% | memory | 240.8 |
| linear int8 | attn1.to_k | up | 30 | 241.3 | 1.2 | 0% | memory | 240.1 |
| linear int8 | ff.net | down | 24 | 241.0 | 3.6 | 2% | memory | 237.4 |
| linear int8 | attn2.to_q | up | 30 | 235.4 | 1.2 | 0% | memory | 234.3 |
| linear int8 | attn1.to_v | up | 30 | 233.9 | 1.2 | 1% | memory | 232.7 |
| linear int8 | attn1.to_q | up | 30 | 233.0 | 1.2 | 1% | memory | 231.9 |
| attention | transformer_blocks.attn1 | down | 24 | 216.9 | 3.9 | 2% | compute | 213.0 |
| linear int8 | attn2.to_k | up | 36 | 206.6 | 1.2 | 1% | memory | 205.5 |
| linear int8 | attn2.to_v | up | 36 | 204.9 | 1.2 | 1% | memory | 203.7 |
| linear int8 | attn2.to_v | down | 24 | 196.6 | 0.9 | 0% | memory | 195.7 |
| linear int8 | attn2.to_k | down | 24 | 189.6 | 0.9 | 0% | memory | 188.7 |
| linear int8 | attn2.to_out | down | 20 | 189.3 | 0.9 | 0% | memory | 188.4 |
| linear int8 | attn1.to_out | down | 20 | 188.7 | 0.9 | 0% | memory | 187.8 |
| linear int8 | net.proj | down | 20 | 192.7 | 5.6 | 3% | memory | 187.2 |
| linear int8 | attn1.to_q | down | 20 | 184.8 | 0.9 | 0% | memory | 183.9 |
| linear int8 | attn2.to_q | down | 20 | 182.5 | 0.9 | 1% | memory | 181.5 |
| linear int8 | attn1.to_k | down | 20 | 181.4 | 0.9 | 1% | memory | 180.5 |
| linear int8 | attn1.to_v | down | 20 | 181.3 | 0.9 | 1% | memory | 180.4 |
| feedforward | transformer_blocks.ff | up | 36 | 158.5 | 17.2 | 11% | memory | 141.3 |
| feedforward | transformer_blocks.ff | down | 24 | 130.1 | 12.4 | 9% | memory | 117.8 |
| attention | transformer_blocks.attn2 | mid | 10 | 100.2 | 0.7 | 1% | memory | 99.5 |
| attention | transformer_blocks.attn1 | mid | 10 | 98.6 | 0.7 | 1% | memory | 97.9 |
| linear int8 | attn1.to_out | mid | 10 | 83.4 | 0.4 | 0% | memory | 83.0 |
| linear int8 | net.proj | mid | 10 | 84.5 | 2.5 | 3% | memory | 82.0 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.7.1.transformer_blocks.0 | block | 2 | 43.0 | 0.0 | 129.0 | 78.6 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 2 | 42.8 | 0.0 | 124.6 | 74.7 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 2 | 43.2 | 0.0 | 120.4 | 72.1 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 2 | 43.0 | 0.0 | 120.0 | 71.9 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 2 | 42.4 | 0.0 | 120.1 | 71.8 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 2 | 43.0 | 0.0 | 118.1 | 70.3 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 2 | 42.7 | 0.0 | 117.4 | 69.9 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 2 | 42.4 | 0.0 | 116.2 | 69.5 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 2 | 42.6 | 0.0 | 117.0 | 69.5 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 2 | 43.2 | 0.0 | 115.6 | 69.0 | - | - |
| middle_block.1.transformer_blocks.6 | block | 2 | 44.9 | 0.0 | 112.6 | 67.3 | - | - |
| middle_block.1.transformer_blocks.8 | block | 2 | 44.6 | 0.0 | 111.7 | 66.3 | - | - |
| output_blocks.1.1.transformer_blocks.0 | block | 2 | 45.5 | 0.0 | 109.8 | 66.2 | - | - |
| output_blocks.1.1.transformer_blocks.1 | block | 2 | 44.7 | 0.0 | 109.5 | 65.5 | - | - |
| output_blocks.2.1.transformer_blocks.7 | block | 2 | 47.6 | 0.0 | 106.8 | 64.6 | - | - |
| output_blocks.1.1.transformer_blocks.2 | block | 2 | 45.4 | 0.0 | 108.2 | 64.5 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 2 | 34.3 | 0.0 | 111.4 | 63.3 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 2 | 34.3 | 0.0 | 113.0 | 63.2 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 2 | 34.2 | 0.0 | 109.8 | 62.9 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 2 | 34.1 | 0.0 | 114.1 | 62.9 | - | - |
| output_blocks.4.1.transformer_blocks.1 | block | 2 | 39.2 | 0.0 | 100.4 | 56.6 | - | - |
| output_blocks.4.1.transformer_blocks.0 | block | 2 | 38.3 | 0.0 | 97.4 | 56.2 | - | - |
| output_blocks.3.1.transformer_blocks.1 | block | 2 | 38.8 | 0.0 | 96.8 | 55.8 | - | - |
| output_blocks.5.1.transformer_blocks.0 | block | 2 | 39.2 | 0.0 | 97.0 | 55.5 | - | - |
| output_blocks.3.1.transformer_blocks.0 | block | 2 | 39.2 | 0.0 | 96.4 | 55.5 | - | - |
| output_blocks.5.1.transformer_blocks.1 | block | 2 | 39.1 | 0.0 | 95.6 | 54.7 | - | - |
| middle_block.1.transformer_blocks.5.attn1 | attention | 2 | 41.9 | 0.0 | 19.7 | 32.0 | 0.1 | 0% |
| output_blocks.2.1.transformer_blocks.8.ff.net.2 | linear | 2 | 24.1 | 0.0 | 3.8 | 27.9 | 0.1 | 0% |
| output_blocks.4.0 | block | 2 | 10.0 | 0.0 | 24.9 | 19.9 | - | - |
| output_blocks.5.0 | block | 2 | 9.6 | 0.0 | 24.2 | 19.8 | - | - |

## Reconciliation

device 750 ms; modules' own time 10476 ms, of which 2325 ms is in modules with no cost model; wall 2055 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 12822 | 201.80 | 16 |
| `(anonymous namespace)::launch_dequant_weight<1>(sycl::_V1::queue&, signed char const*, flo` | int8 | 1008 | 79.22 | 79 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 4082 | 61.30 | 15 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 3238 | 58.95 | 18 |
| `(anonymous namespace)::launch_dequant_weight<2>(sycl::_V1::queue&, signed char const*, flo` | int8 | 120 | 46.49 | 387 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 4686 | 43.19 | 9 |
| `gen_conv` | convolution | 222 | 42.62 | 192 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 280 | 39.39 | 141 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1651 | 23.52 | 14 |
| `conv_reorder` | convolution | 408 | 18.12 | 44 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1664 | 17.54 | 11 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 140 | 13.13 | 94 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 332 | 10.44 | 31 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 4927 | 8.12 | 2 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 576 | 7.57 | 13 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 140 | 6.36 | 45 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 200 | 5.99 | 30 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 420 | 5.64 | 13 |
| `micro_sdpa` | attention | 104 | 5.39 | 52 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 140 | 5.16 | 37 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 2912 | 4.99 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 192 | 4.78 | 25 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::FillFunctor<float>, at::d` | memory | 6718 | 4.13 | 1 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 116 | 4.13 | 36 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 80 | 3.29 | 41 |
