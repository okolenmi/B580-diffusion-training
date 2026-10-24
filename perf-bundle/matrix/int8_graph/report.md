# Performance report (xpu)

Window: 3 steps (2 eager, 1 replayed). All times are per step.

## Step budget

- step 1325 ms, median of 3 unprofiled steps (best 1325); eager 1325 ms, replayed 5422 ms
- the profiled window's steps took 7372 ms each (hooks and profiler); device and module times come from them
- device busy 853 ms, utilisation 64%
- 35109 kernel launches, 226839 operators, 147 us host time per launch
- step peak memory 4469 MB
- phases: loss 0 ms

## Findings, largest first

1. **Recomputing activations is a large part of the step** (3421 ms)
   - 3421 ms per step (16% of module time) is forward work done a second time by checkpointing
   - lever: keep more blocks un-checkpointed (trainer.checkpoint_by=measured with tune_checkpointing, attention_fraction) if memory allows
2. **Slowest pieces by recoverable time: linear ff.net (up)** (202 ms)
   - 209.6 ms per step against an ideal 7.4 ms (4% efficient, compute-bound), 36 layers; all rated pieces together leave 5372 ms
   - lever: see the module table for the individual layers
3. **Slowest pieces by recoverable time: linear attn2.to_k (up)** (198 ms)
   - 198.7 ms per step against an ideal 0.9 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 5372 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: linear attn2.to_v (up)** (198 ms)
   - 198.6 ms per step against an ideal 0.9 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 5372 ms
   - lever: see the module table for the individual layers
5. **Dtype and copy kernels take a large share** (192 ms)
   - 23% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (184 ms)
   - 193.5 ms per step against an ideal 9.2 ms (5% efficient, compute-bound), 36 layers; all rated pieces together leave 5372 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: linear attn2.to_out (up)** (181 ms)
   - 183.0 ms per step against an ideal 1.6 ms (1% efficient, memory-bound), 30 layers; all rated pieces together leave 5372 ms
   - lever: see the module table for the individual layers
8. **Many small memory-bound kernels** (170 ms)
   - 40% of device time (22301 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
9. **The int8 path's own kernels are a large share** (132 ms)
   - rotate, quantize and dequantize kernels are 15% of device time against 253 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows
10. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5422 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8390 | 252.8 | 30% |
| copy | 6291 | 192.0 | 23% |
| int8 | 3203 | 131.6 | 15% |
| elementwise | 9331 | 122.4 | 14% |
| convolution | 466 | 61.5 | 7% |
| other | 516 | 34.3 | 4% |
| attention | 233 | 32.6 | 4% |
| norm | 782 | 19.7 | 2% |
| memory | 5879 | 4.9 | 1% |
| reduce | 18 | 1.6 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 104.6 TFLOPS, int8 199.4 TOPS, copy 386.2 GB/s, dispatch 10.5 us, queued launch 10.5 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| linear int8 | ff.net | up | 36 | 209.6 | 7.4 | 4% | compute | 202.2 |
| linear int8 | attn2.to_k | up | 36 | 198.7 | 0.9 | 0% | memory | 197.7 |
| linear int8 | attn2.to_v | up | 36 | 198.6 | 0.9 | 0% | memory | 197.6 |
| attention | transformer_blocks.attn1 | up | 36 | 193.5 | 9.2 | 5% | compute | 184.3 |
| linear int8 | attn2.to_out | up | 30 | 183.0 | 1.6 | 1% | memory | 181.4 |
| attention | transformer_blocks.attn2 | up | 36 | 180.6 | 9.2 | 5% | compute | 171.4 |
| linear int8 | attn1.to_out | up | 30 | 172.1 | 1.6 | 1% | memory | 170.4 |
| linear int8 | net.proj | up | 30 | 180.3 | 12.1 | 7% | compute | 168.2 |
| linear int8 | attn1.to_q | up | 30 | 168.0 | 1.6 | 1% | memory | 166.4 |
| linear int8 | attn2.to_q | up | 30 | 167.2 | 1.6 | 1% | memory | 165.6 |
| linear int8 | attn1.to_v | up | 30 | 164.8 | 1.6 | 1% | memory | 163.2 |
| linear int8 | attn1.to_k | up | 30 | 164.0 | 1.6 | 1% | memory | 162.4 |
| attention | transformer_blocks.attn1 | down | 24 | 162.3 | 6.1 | 4% | compute | 156.2 |
| attention | transformer_blocks.attn2 | down | 24 | 153.3 | 6.1 | 4% | compute | 147.2 |
| linear int8 | ff.net | down | 24 | 147.7 | 4.9 | 3% | compute | 142.8 |
| linear int8 | attn2.to_k | down | 24 | 141.2 | 0.6 | 0% | memory | 140.5 |
| linear int8 | attn2.to_v | down | 24 | 140.3 | 0.6 | 0% | memory | 139.7 |
| linear int8 | attn1.to_out | down | 20 | 123.5 | 1.1 | 1% | memory | 122.4 |
| linear int8 | attn2.to_out | down | 20 | 122.3 | 1.1 | 1% | memory | 121.3 |
| linear int8 | net.proj | down | 20 | 127.5 | 8.1 | 6% | compute | 119.4 |
| linear int8 | attn1.to_k | down | 20 | 119.6 | 1.1 | 1% | memory | 118.5 |
| linear int8 | attn1.to_q | down | 20 | 119.5 | 1.1 | 1% | memory | 118.5 |
| linear int8 | attn2.to_q | down | 20 | 119.1 | 1.1 | 1% | memory | 118.1 |
| linear int8 | attn1.to_v | down | 20 | 116.4 | 1.1 | 1% | memory | 115.3 |
| feedforward | transformer_blocks.ff | up | 36 | 107.9 | 27.4 | 25% | memory | 80.6 |
| feedforward | transformer_blocks.ff | down | 24 | 89.4 | 18.2 | 20% | memory | 71.1 |
| linear int8 | attn1.to_v | mid | 10 | 71.1 | 0.5 | 1% | memory | 70.5 |
| linear int8 | ff.net | mid | 10 | 63.5 | 2.0 | 3% | compute | 61.5 |
| linear int8 | attn2.to_out | mid | 10 | 60.2 | 0.5 | 1% | memory | 59.6 |
| linear int8 | attn1.to_out | mid | 10 | 60.1 | 0.5 | 1% | memory | 59.6 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 21.3 | 0.0 | 74.0 | 44.1 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 21.0 | 0.0 | 72.2 | 43.2 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 24.7 | 0.0 | 72.4 | 42.3 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 24.6 | 0.0 | 69.7 | 42.3 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 24.0 | 0.0 | 69.9 | 41.4 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 21.3 | 0.0 | 68.5 | 41.4 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 23.4 | 0.0 | 69.4 | 41.2 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 23.2 | 0.0 | 67.8 | 40.7 | - | - |
| middle_block.1.transformer_blocks.1 | block | 1 | 22.1 | 0.0 | 69.7 | 40.4 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 22.9 | 0.0 | 66.7 | 40.1 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 22.1 | 0.0 | 67.2 | 40.0 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 23.5 | 0.0 | 67.0 | 39.9 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 21.9 | 0.0 | 67.6 | 39.7 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 24.6 | 0.0 | 68.2 | 39.5 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 24.0 | 0.0 | 65.7 | 39.4 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 22.1 | 0.0 | 65.6 | 39.3 | - | - |
| middle_block.1.transformer_blocks.7 | block | 1 | 23.8 | 0.0 | 66.3 | 39.3 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 22.4 | 0.0 | 65.3 | 39.1 | - | - |
| middle_block.1.transformer_blocks.4 | block | 1 | 24.1 | 0.0 | 64.5 | 39.0 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 21.8 | 0.0 | 65.8 | 38.8 | - | - |
| output_blocks.0.1.transformer_blocks.2 | block | 1 | 24.1 | 0.0 | 64.7 | 38.7 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 17.2 | 0.0 | 68.5 | 38.7 | - | - |
| middle_block.1.transformer_blocks.9 | block | 1 | 21.9 | 0.0 | 65.5 | 38.7 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 23.9 | 0.0 | 63.9 | 38.4 | - | - |
| middle_block.1.transformer_blocks.5 | block | 1 | 25.6 | 0.0 | 63.9 | 38.2 | - | - |
| middle_block.1.transformer_blocks.0 | block | 1 | 22.8 | 0.0 | 64.2 | 38.1 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 24.1 | 0.0 | 64.2 | 37.9 | - | - |
| middle_block.1.transformer_blocks.3 | block | 1 | 22.7 | 0.0 | 63.5 | 37.8 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 23.0 | 0.0 | 66.7 | 37.8 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 24.2 | 0.0 | 65.9 | 37.7 | - | - |

## Reconciliation

device 853 ms; modules - ms (unrated - ms). Module times need eager steps: a window with replayed steps has no module table.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 8296 | 247.47 | 30 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2474 | 88.77 | 36 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2980 | 76.23 | 26 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 58.94 | 19 |
| `gen_conv` | convolution | 131 | 46.39 | 354 |
| `(anonymous namespace)::launch_dequant_weight<1>(sycl::_V1::queue&, signed char const*, flo` | int8 | 599 | 41.12 | 69 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1265 | 38.57 | 30 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 93 | 29.79 | 319 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1272 | 26.64 | 21 |
| `(anonymous namespace)::launch_dequant_weight<2>(sycl::_V1::queue&, signed char const*, flo` | int8 | 60 | 25.10 | 418 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 18.04 | 86 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 14.40 | 206 |
| `conv_reorder` | convolution | 241 | 13.93 | 58 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3937 | 9.71 | 2 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.52 | 23 |
| `at::native::xpu::impl::DispatchSoftmaxForwardKernelFunctor<8, 4, 32, float, float, float, ` | attention | 7 | 7.30 | 1096 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 7.03 | 50 |
| `at::native::xpu::impl::DispatchSoftmaxForwardKernelFunctor<8, 1, 32, float, float, float, ` | attention | 40 | 6.80 | 170 |
| `micro_sdpa` | attention | 93 | 6.33 | 68 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.23 | 89 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.96 | 85 |
| `at::native::xpu::impl::DispatchSoftmaxBackwardKernelFunctor<4, 1, 32, float, float, float,` | attention | 43 | 5.40 | 125 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 93 | 5.32 | 57 |
| `at::native::xpu::impl::DispatchSoftmaxBackwardKernelFunctor<4, 4, 32, float, float, float,` | attention | 3 | 5.20 | 1560 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.59 | 22 |
