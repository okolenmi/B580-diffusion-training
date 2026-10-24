# Performance report (xpu)

Window: 3 steps (2 eager, 1 replayed). All times are per step.

## Step budget

- step 1328 ms, median of 3 unprofiled steps (best 1325); eager 1326 ms, replayed 5535 ms
- the profiled window's steps took 7325 ms each (hooks and profiler); device and module times come from them
- device busy 900 ms, utilisation 68%
- 35739 kernel launches, 228152 operators, 142 us host time per launch
- step peak memory 4417 MB
- phases: loss 0 ms

## Findings, largest first

1. **Recomputing activations is a large part of the step** (3220 ms)
   - 3220 ms per step (16% of module time) is forward work done a second time by checkpointing
   - lever: keep more blocks un-checkpointed (trainer.checkpoint_by=measured with tune_checkpointing, attention_fraction) if memory allows
2. **Dtype and copy kernels take a large share** (222 ms)
   - 25% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
3. **Slowest pieces by recoverable time: linear ff.net (up)** (194 ms)
   - 201.1 ms per step against an ideal 7.4 ms (4% efficient, compute-bound), 36 layers; all rated pieces together leave 5107 ms
   - lever: see the module table for the individual layers
4. **Many small memory-bound kernels** (193 ms)
   - 43% of device time (22845 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
5. **Slowest pieces by recoverable time: linear attn2.to_k (up)** (190 ms)
   - 191.4 ms per step against an ideal 0.9 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 5107 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: linear attn2.to_v (up)** (189 ms)
   - 189.8 ms per step against an ideal 0.9 ms (1% efficient, memory-bound), 36 layers; all rated pieces together leave 5107 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (176 ms)
   - 185.2 ms per step against an ideal 9.2 ms (5% efficient, compute-bound), 36 layers; all rated pieces together leave 5107 ms
   - lever: see the module table for the individual layers
8. **Slowest pieces by recoverable time: linear attn2.to_out (up)** (166 ms)
   - 167.5 ms per step against an ideal 1.6 ms (1% efficient, memory-bound), 30 layers; all rated pieces together leave 5107 ms
   - lever: see the module table for the individual layers
9. **The int8 path's own kernels are a large share** (136 ms)
   - rotate, quantize and dequantize kernels are 15% of device time against 253 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows
10. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5535 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8466 | 253.2 | 28% |
| copy | 6661 | 222.1 | 25% |
| elementwise | 9455 | 137.3 | 15% |
| int8 | 3203 | 135.7 | 15% |
| convolution | 466 | 61.2 | 7% |
| other | 516 | 33.9 | 4% |
| attention | 243 | 29.4 | 3% |
| norm | 782 | 19.1 | 2% |
| memory | 5879 | 4.8 | 1% |
| reduce | 68 | 3.4 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 104.6 TFLOPS, int8 199.4 TOPS, copy 386.2 GB/s, dispatch 10.5 us, queued launch 10.5 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| linear int8 | ff.net | up | 36 | 201.1 | 7.4 | 4% | compute | 193.8 |
| linear int8 | attn2.to_k | up | 36 | 191.4 | 0.9 | 0% | memory | 190.5 |
| linear int8 | attn2.to_v | up | 36 | 189.8 | 0.9 | 1% | memory | 188.9 |
| attention | transformer_blocks.attn1 | up | 36 | 185.2 | 9.2 | 5% | compute | 176.0 |
| linear int8 | attn2.to_out | up | 30 | 167.5 | 1.6 | 1% | memory | 165.9 |
| linear int8 | attn1.to_out | up | 30 | 166.4 | 1.6 | 1% | memory | 164.8 |
| linear int8 | net.proj | up | 30 | 174.5 | 12.1 | 7% | compute | 162.4 |
| attention | transformer_blocks.attn2 | up | 36 | 171.1 | 9.2 | 5% | compute | 161.9 |
| linear int8 | attn1.to_q | up | 30 | 162.6 | 1.6 | 1% | memory | 161.0 |
| linear int8 | attn2.to_q | up | 30 | 161.4 | 1.6 | 1% | memory | 159.8 |
| linear int8 | attn1.to_k | up | 30 | 159.4 | 1.6 | 1% | memory | 157.8 |
| linear int8 | attn1.to_v | up | 30 | 159.0 | 1.6 | 1% | memory | 157.4 |
| attention | transformer_blocks.attn1 | down | 24 | 163.0 | 6.1 | 4% | compute | 156.8 |
| linear int8 | ff.net | down | 24 | 140.8 | 4.9 | 3% | compute | 135.8 |
| linear int8 | attn2.to_k | down | 24 | 134.6 | 0.6 | 0% | memory | 134.0 |
| attention | transformer_blocks.attn2 | down | 24 | 140.0 | 6.1 | 4% | compute | 133.9 |
| linear int8 | attn2.to_v | down | 24 | 133.7 | 0.6 | 0% | memory | 133.1 |
| linear int8 | net.proj | down | 20 | 130.8 | 8.1 | 6% | compute | 122.7 |
| linear int8 | attn2.to_out | down | 20 | 117.2 | 1.1 | 1% | memory | 116.1 |
| linear int8 | attn1.to_out | down | 20 | 116.5 | 1.1 | 1% | memory | 115.5 |
| linear int8 | attn1.to_q | down | 20 | 113.2 | 1.1 | 1% | memory | 112.1 |
| linear int8 | attn2.to_q | down | 20 | 112.9 | 1.1 | 1% | memory | 111.9 |
| linear int8 | attn1.to_v | down | 20 | 111.8 | 1.1 | 1% | memory | 110.7 |
| linear int8 | attn1.to_k | down | 20 | 111.3 | 1.1 | 1% | memory | 110.2 |
| feedforward | transformer_blocks.ff | up | 36 | 102.1 | 27.4 | 27% | memory | 74.8 |
| feedforward | transformer_blocks.ff | down | 24 | 82.9 | 18.2 | 22% | memory | 64.7 |
| linear int8 | attn1.to_out | mid | 10 | 57.0 | 0.5 | 1% | memory | 56.5 |
| linear int8 | ff.net | mid | 10 | 58.4 | 2.0 | 3% | compute | 56.3 |
| linear int8 | attn2.to_out | mid | 10 | 56.8 | 0.5 | 1% | memory | 56.3 |
| linear int8 | attn1.to_q | mid | 10 | 56.4 | 0.5 | 1% | memory | 55.8 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 22.7 | 0.0 | 64.9 | 39.2 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 22.3 | 0.0 | 64.8 | 38.9 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 22.7 | 0.0 | 65.0 | 38.9 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 22.3 | 0.0 | 64.6 | 38.3 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 22.2 | 0.0 | 64.1 | 38.2 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 22.4 | 0.0 | 64.6 | 38.1 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 22.8 | 0.0 | 63.0 | 38.1 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 22.4 | 0.0 | 63.7 | 37.8 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 23.0 | 0.0 | 62.6 | 37.8 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 22.5 | 0.0 | 63.4 | 37.7 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 23.2 | 0.0 | 62.0 | 37.7 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 22.7 | 0.0 | 62.5 | 37.6 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 23.1 | 0.0 | 61.9 | 37.1 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 23.7 | 0.0 | 61.8 | 37.0 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 22.6 | 0.0 | 62.0 | 37.0 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 22.9 | 0.0 | 61.7 | 36.8 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 22.8 | 0.0 | 61.5 | 36.8 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 23.3 | 0.0 | 63.8 | 36.7 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 23.1 | 0.0 | 61.6 | 36.7 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 36.6 | 0.0 | 61.7 | 36.6 | - | - |
| middle_block.1.transformer_blocks.1 | block | 1 | 23.2 | 0.0 | 60.3 | 36.4 | - | - |
| output_blocks.0.1.transformer_blocks.1 | block | 1 | 23.5 | 0.0 | 59.4 | 36.4 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 32.9 | 0.0 | 60.0 | 36.0 | - | - |
| middle_block.1.transformer_blocks.0 | block | 1 | 23.1 | 0.0 | 60.1 | 36.0 | - | - |
| middle_block.1.transformer_blocks.3 | block | 1 | 23.2 | 0.0 | 59.6 | 35.9 | - | - |
| output_blocks.0.1.transformer_blocks.2 | block | 1 | 23.4 | 0.0 | 59.3 | 35.5 | - | - |
| middle_block.1.transformer_blocks.4 | block | 1 | 25.3 | 0.0 | 58.9 | 35.4 | - | - |
| middle_block.1.transformer_blocks.6 | block | 1 | 23.3 | 0.0 | 58.8 | 35.3 | - | - |
| middle_block.1.transformer_blocks.5 | block | 1 | 24.0 | 0.0 | 59.0 | 35.2 | - | - |
| middle_block.1.transformer_blocks.9 | block | 1 | 23.3 | 0.0 | 58.7 | 35.2 | - | - |

## Reconciliation

device 900 ms; modules - ms (unrated - ms). Module times need eager steps: a window with replayed steps has no module table.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 8373 | 248.04 | 30 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2440 | 97.16 | 40 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 3040 | 95.55 | 31 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 58.00 | 19 |
| `(anonymous namespace)::launch_dequant_weight<1>(sycl::_V1::queue&, signed char const*, flo` | int8 | 599 | 48.93 | 82 |
| `gen_conv` | convolution | 131 | 46.19 | 353 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1265 | 37.73 | 30 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 93 | 29.40 | 315 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1272 | 25.75 | 20 |
| `(anonymous namespace)::launch_dequant_weight<2>(sycl::_V1::queue&, signed char const*, flo` | int8 | 60 | 23.12 | 385 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 17.80 | 85 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 14.38 | 205 |
| `conv_reorder` | convolution | 241 | 13.81 | 57 |
| `at::native::xpu::impl::DispatchSoftmaxForwardKernelFunctor<8, 4, 32, float, float, float, ` | attention | 20 | 10.95 | 547 |
| `at::native::xpu::impl::DispatchSoftmaxForwardKernelFunctor<8, 1, 32, float, float, float, ` | attention | 60 | 9.94 | 166 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 53 | 9.56 | 180 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.24 | 22 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 6.87 | 49 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3704 | 6.86 | 2 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<f` | elementwise | 50 | 6.76 | 135 |
| `micro_sdpa` | attention | 93 | 6.26 | 67 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::CopyScalarFunc<c10::BFloa` | copy | 784 | 6.25 | 8 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.21 | 89 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.02 | 86 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 93 | 5.20 | 56 |
