# Performance report (xpu)

Window: 3 steps (2 eager, 1 replayed). All times are per step.

## Step budget

- step 1560 ms, median of 3 unprofiled steps (best 1558); eager 1559 ms, replayed 5760 ms
- the profiled window's steps took 8359 ms each (hooks and profiler); device and module times come from them
- device busy 853 ms, utilisation 55%
- 40295 kernel launches, 244568 operators, 147 us host time per launch
- step peak memory 4469 MB
- phases: loss 0 ms

## Findings, largest first

1. **The step is launch bound** (707 ms)
   - the device is busy 55% of the step (853 of 1560 ms) over 40295 launches and 244568 operators per step, 147 us of host time per launch
   - lever: graph replay is already running for some steps; compare replayed and eager step times above
2. **Slowest pieces by recoverable time: linear ff.net (up)** (249 ms)
   - 256.4 ms per step against an ideal 7.4 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 6309 ms
   - lever: see the module table for the individual layers
3. **Slowest pieces by recoverable time: linear attn2.to_v (up)** (241 ms)
   - 242.3 ms per step against an ideal 0.9 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 6309 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: linear attn2.to_k (up)** (241 ms)
   - 242.1 ms per step against an ideal 0.9 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 6309 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: linear attn2.to_out (up)** (211 ms)
   - 212.3 ms per step against an ideal 1.6 ms (1% efficient, memory-bound), 30 layers; all rated pieces together leave 6309 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (210 ms)
   - 218.8 ms per step against an ideal 9.2 ms (4% efficient, compute-bound), 36 layers; all rated pieces together leave 6309 ms
   - lever: see the module table for the individual layers
7. **Dtype and copy kernels take a large share** (197 ms)
   - 23% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
8. **Many small memory-bound kernels** (180 ms)
   - 42% of device time (26203 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
9. **The int8 path's own kernels are a large share** (125 ms)
   - rotate, quantize and dequantize kernels are 15% of device time against 237 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows
10. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5760 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8390 | 237.3 | 28% |
| copy | 6983 | 196.6 | 23% |
| elementwise | 9973 | 135.1 | 16% |
| int8 | 4487 | 125.2 | 15% |
| convolution | 466 | 61.1 | 7% |
| other | 516 | 35.1 | 4% |
| attention | 233 | 33.7 | 4% |
| norm | 782 | 20.4 | 2% |
| memory | 8447 | 6.7 | 1% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 104.6 TFLOPS, int8 199.4 TOPS, copy 386.2 GB/s, dispatch 10.5 us, queued launch 10.5 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| linear int8 | ff.net | up | 36 | 256.4 | 7.4 | 3% | compute | 249.0 |
| linear int8 | attn2.to_v | up | 36 | 242.3 | 0.9 | 0% | memory | 241.3 |
| linear int8 | attn2.to_k | up | 36 | 242.1 | 0.9 | 0% | memory | 241.1 |
| linear int8 | attn2.to_out | up | 30 | 212.3 | 1.6 | 1% | memory | 210.7 |
| attention | transformer_blocks.attn1 | up | 36 | 218.8 | 9.2 | 4% | compute | 209.6 |
| linear int8 | attn1.to_out | up | 30 | 211.0 | 1.6 | 1% | memory | 209.3 |
| linear int8 | attn1.to_q | up | 30 | 207.9 | 1.6 | 1% | memory | 206.3 |
| linear int8 | net.proj | up | 30 | 217.5 | 12.1 | 6% | compute | 205.3 |
| linear int8 | attn2.to_q | up | 30 | 205.3 | 1.6 | 1% | memory | 203.7 |
| linear int8 | attn1.to_k | up | 30 | 203.3 | 1.6 | 1% | memory | 201.7 |
| linear int8 | attn1.to_v | up | 30 | 202.6 | 1.6 | 1% | memory | 201.0 |
| attention | transformer_blocks.attn2 | up | 36 | 204.7 | 9.2 | 4% | compute | 195.5 |
| attention | transformer_blocks.attn1 | down | 24 | 179.7 | 6.1 | 3% | compute | 173.5 |
| linear int8 | ff.net | down | 24 | 176.2 | 4.9 | 3% | compute | 171.3 |
| linear int8 | attn2.to_v | down | 24 | 170.5 | 0.6 | 0% | memory | 169.9 |
| linear int8 | attn2.to_k | down | 24 | 169.4 | 0.6 | 0% | memory | 168.8 |
| attention | transformer_blocks.attn2 | down | 24 | 170.2 | 6.1 | 4% | compute | 164.1 |
| linear int8 | attn1.to_out | down | 20 | 156.7 | 1.1 | 1% | memory | 155.6 |
| linear int8 | attn2.to_out | down | 20 | 147.2 | 1.1 | 1% | memory | 146.2 |
| linear int8 | net.proj | down | 20 | 151.2 | 8.1 | 5% | compute | 143.1 |
| linear int8 | attn1.to_v | down | 20 | 142.0 | 1.1 | 1% | memory | 141.0 |
| linear int8 | attn1.to_q | down | 20 | 142.0 | 1.1 | 1% | memory | 140.9 |
| linear int8 | attn2.to_q | down | 20 | 142.0 | 1.1 | 1% | memory | 140.9 |
| linear int8 | attn1.to_k | down | 20 | 141.2 | 1.1 | 1% | memory | 140.1 |
| feedforward | transformer_blocks.ff | up | 36 | 121.6 | 27.4 | 23% | memory | 94.2 |
| feedforward | transformer_blocks.ff | down | 24 | 98.9 | 18.2 | 18% | memory | 80.7 |
| linear int8 | attn1.to_out | mid | 10 | 72.4 | 0.5 | 1% | memory | 71.9 |
| linear int8 | ff.net | mid | 10 | 73.5 | 2.0 | 3% | compute | 71.5 |
| linear int8 | attn2.to_out | mid | 10 | 71.8 | 0.5 | 1% | memory | 71.3 |
| linear int8 | attn1.to_q | mid | 10 | 71.1 | 0.5 | 1% | memory | 70.6 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 22.3 | 0.0 | 87.1 | 46.2 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 22.6 | 0.0 | 88.6 | 45.4 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 23.9 | 0.0 | 85.8 | 45.1 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 22.5 | 0.0 | 84.8 | 44.2 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 22.7 | 0.0 | 94.5 | 43.8 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 22.2 | 0.0 | 83.7 | 43.0 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 22.0 | 0.0 | 82.8 | 42.9 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 24.2 | 0.0 | 81.2 | 42.9 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 22.5 | 0.0 | 82.0 | 42.7 | - | - |
| output_blocks.0.1.transformer_blocks.8 | block | 1 | 24.3 | 0.0 | 81.7 | 42.6 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 22.1 | 0.0 | 82.5 | 42.4 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 22.5 | 0.0 | 81.2 | 42.3 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 22.8 | 0.0 | 82.6 | 42.2 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 22.8 | 0.0 | 81.1 | 42.2 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 22.2 | 0.0 | 81.9 | 42.1 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 22.8 | 0.0 | 81.0 | 41.7 | - | - |
| output_blocks.0.1.transformer_blocks.7 | block | 1 | 25.2 | 0.0 | 80.1 | 41.6 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 22.2 | 0.0 | 81.4 | 41.5 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 24.1 | 0.0 | 79.4 | 41.2 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 23.5 | 0.0 | 80.3 | 41.1 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 22.5 | 0.0 | 80.0 | 41.0 | - | - |
| middle_block.1.transformer_blocks.3 | block | 1 | 23.5 | 0.0 | 78.5 | 40.8 | - | - |
| middle_block.1.transformer_blocks.1 | block | 1 | 24.2 | 0.0 | 77.8 | 40.7 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 22.8 | 0.0 | 78.5 | 40.6 | - | - |
| output_blocks.0.1.transformer_blocks.6 | block | 1 | 24.4 | 0.0 | 77.6 | 40.5 | - | - |
| output_blocks.2.1.transformer_blocks.2 | block | 1 | 25.9 | 0.0 | 76.5 | 40.5 | - | - |
| middle_block.1.transformer_blocks.0 | block | 1 | 24.0 | 0.0 | 77.9 | 40.1 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 23.2 | 0.0 | 78.0 | 40.1 | - | - |
| middle_block.1.transformer_blocks.6 | block | 1 | 24.2 | 0.0 | 78.2 | 40.0 | - | - |
| middle_block.1.transformer_blocks.5 | block | 1 | 24.2 | 0.0 | 77.4 | 39.9 | - | - |

## Reconciliation

device 853 ms; modules - ms (unrated - ms). Module times need eager steps: a window with replayed steps has no module table.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 8296 | 232.02 | 28 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 3116 | 90.75 | 29 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2980 | 77.13 | 26 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 60.95 | 20 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1904 | 54.10 | 28 |
| `gen_conv` | convolution | 131 | 46.55 | 355 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 93 | 30.51 | 327 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 803 | 28.52 | 36 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1272 | 27.56 | 22 |
| `(anonymous namespace)::launch_rotate_bf16<1>(sycl::_V1::queue&, unsigned short const*, syc` | int8 | 582 | 18.71 | 32 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 15.32 | 219 |
| `conv_reorder` | convolution | 241 | 13.44 | 56 |
| `(anonymous namespace)::launch_rotate_bf16<2>(sycl::_V1::queue&, unsigned short const*, syc` | int8 | 60 | 13.06 | 218 |
| `_quantize_row_kernel` | int8 | 642 | 10.17 | 16 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.78 | 23 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3937 | 9.54 | 2 |
| `at::native::xpu::impl::DispatchSoftmaxForwardKernelFunctor<8, 4, 32, float, float, float, ` | attention | 7 | 7.62 | 1143 |
| `at::native::xpu::impl::DispatchSoftmaxForwardKernelFunctor<8, 1, 32, float, float, float, ` | attention | 40 | 7.13 | 178 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 7.00 | 50 |
| `micro_sdpa` | attention | 93 | 6.37 | 68 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.19 | 88 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.00 | 86 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::FillFunctor<float>, at::d` | memory | 8404 | 5.96 | 1 |
| `at::native::xpu::impl::DispatchSoftmaxBackwardKernelFunctor<4, 1, 32, float, float, float,` | attention | 43 | 5.60 | 129 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 93 | 5.29 | 57 |
