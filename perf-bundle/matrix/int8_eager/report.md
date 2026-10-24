# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- step 1360 ms, median of 3 unprofiled steps (best 1329); eager 1360 ms, replayed - ms
- the profiled window's steps took 6740 ms each (hooks and profiler); device and module times come from them
- device busy 778 ms, utilisation 57%
- 32938 kernel launches, 214712 operators, 201 us host time per launch
- step peak memory 4244 MB
- phases: loss 0 ms

## Findings, largest first

1. **Recomputing activations is a large part of the step** (3235 ms)
   - 3235 ms per step (16% of module time) is forward work done a second time by checkpointing
   - lever: keep more blocks un-checkpointed (trainer.checkpoint_by=measured with tune_checkpointing, attention_fraction) if memory allows
2. **Much of the module time has no cost model** (2600 ms)
   - 2600 of 7835 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
3. **The step is launch bound** (582 ms)
   - the device is busy 57% of the step (778 of 1360 ms) over 32938 launches and 214712 operators per step, 201 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
4. **Slowest pieces by recoverable time: linear ff.net (up)** (191 ms)
   - 196.7 ms per step against an ideal 5.6 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 5064 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: linear attn2.to_v (up)** (186 ms)
   - 187.0 ms per step against an ideal 0.9 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 5064 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: linear attn2.to_k (up)** (186 ms)
   - 187.0 ms per step against an ideal 0.9 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 5064 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (174 ms)
   - 180.5 ms per step against an ideal 6.2 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 5064 ms
   - lever: see the module table for the individual layers
8. **Dtype and copy kernels take a large share** (164 ms)
   - 21% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
9. **Slowest pieces by recoverable time: linear attn1.to_out (up)** (162 ms)
   - 163.5 ms per step against an ideal 1.3 ms (1% efficient, memory-bound), 30 layers; all rated pieces together leave 5064 ms
   - lever: see the module table for the individual layers
10. **Many small memory-bound kernels** (153 ms)
   - 39% of device time (20410 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
11. **The int8 path's own kernels are a large share** (143 ms)
   - rotate, quantize and dequantize kernels are 18% of device time against 214 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows
12. **Some steps in the window were much slower than the rest** (0 ms)
   - 1 step(s) took over twice the median: step 20 (2, 68, 92) 5531 ms
   - lever: a latent shape seen for the first time costs kernel and primitive creation (seconds); fewer distinct shapes (buckets.max_shapes) and a longer warm-up
13. **Module times do not add up to the device time** (0 ms)
   - modules account for 7835 ms of 778 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 214.1 | 28% |
| copy | 5116 | 164.0 | 21% |
| int8 | 3203 | 142.7 | 18% |
| elementwise | 9051 | 117.6 | 15% |
| convolution | 512 | 61.2 | 8% |
| other | 610 | 44.6 | 6% |
| norm | 782 | 19.0 | 2% |
| attention | 140 | 8.8 | 1% |
| memory | 5443 | 4.6 | 1% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 97.7 TFLOPS, int8 200.6 TOPS, copy 396.5 GB/s, dispatch 10.3 us, queued launch 10.3 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| linear int8 | ff.net | up | 36 | 196.7 | 5.6 | 3% | compute | 191.1 |
| linear int8 | attn2.to_v | up | 36 | 187.0 | 0.9 | 0% | memory | 186.1 |
| linear int8 | attn2.to_k | up | 36 | 187.0 | 0.9 | 0% | memory | 186.1 |
| attention | transformer_blocks.attn1 | up | 36 | 180.5 | 6.2 | 3% | compute | 174.4 |
| linear int8 | attn1.to_out | up | 30 | 163.5 | 1.3 | 1% | memory | 162.2 |
| linear int8 | attn2.to_out | up | 30 | 163.2 | 1.3 | 1% | memory | 161.9 |
| attention | transformer_blocks.attn2 | up | 36 | 168.1 | 6.2 | 4% | compute | 161.9 |
| linear int8 | net.proj | up | 30 | 170.9 | 9.2 | 5% | compute | 161.7 |
| linear int8 | attn2.to_q | up | 30 | 162.2 | 1.3 | 1% | memory | 160.9 |
| linear int8 | attn1.to_q | up | 30 | 159.8 | 1.3 | 1% | memory | 158.5 |
| linear int8 | attn1.to_v | up | 30 | 157.7 | 1.3 | 1% | memory | 156.4 |
| linear int8 | attn1.to_k | up | 30 | 156.7 | 1.3 | 1% | memory | 155.4 |
| attention | transformer_blocks.attn1 | down | 24 | 146.1 | 4.1 | 3% | compute | 142.0 |
| linear int8 | ff.net | down | 24 | 139.6 | 3.7 | 3% | compute | 135.8 |
| attention | transformer_blocks.attn2 | down | 24 | 137.2 | 4.1 | 3% | compute | 133.1 |
| linear int8 | attn2.to_k | down | 24 | 132.1 | 0.6 | 0% | memory | 131.5 |
| linear int8 | attn2.to_v | down | 24 | 132.0 | 0.6 | 0% | memory | 131.4 |
| linear int8 | attn1.to_out | down | 20 | 124.4 | 0.9 | 1% | memory | 123.5 |
| linear int8 | attn2.to_out | down | 20 | 116.6 | 0.9 | 1% | memory | 115.8 |
| linear int8 | attn1.to_q | down | 20 | 115.9 | 0.9 | 1% | memory | 115.1 |
| linear int8 | net.proj | down | 20 | 119.2 | 6.1 | 5% | compute | 113.1 |
| linear int8 | attn1.to_v | down | 20 | 110.9 | 0.9 | 1% | memory | 110.0 |
| linear int8 | attn2.to_q | down | 20 | 110.9 | 0.9 | 1% | memory | 110.0 |
| linear int8 | attn1.to_k | down | 20 | 110.3 | 0.9 | 1% | memory | 109.4 |
| feedforward | transformer_blocks.ff | up | 36 | 99.8 | 20.4 | 20% | memory | 79.4 |
| feedforward | transformer_blocks.ff | down | 24 | 81.7 | 13.6 | 17% | memory | 68.1 |
| linear int8 | ff.net | mid | 10 | 58.0 | 1.5 | 3% | compute | 56.4 |
| linear int8 | attn1.to_out | mid | 10 | 56.1 | 0.4 | 1% | memory | 55.7 |
| linear int8 | attn2.to_out | mid | 10 | 55.2 | 0.4 | 1% | memory | 54.8 |
| linear int8 | net.proj | mid | 10 | 57.5 | 3.1 | 5% | compute | 54.4 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 20.6 | 0.0 | 65.9 | 39.2 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 20.6 | 0.0 | 66.0 | 39.0 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 20.4 | 0.0 | 65.3 | 38.7 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 24.4 | 0.0 | 65.0 | 38.5 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 20.4 | 0.0 | 64.7 | 38.4 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 20.7 | 0.0 | 65.8 | 38.3 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 20.6 | 0.0 | 64.3 | 37.9 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 20.6 | 0.0 | 63.9 | 37.7 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 20.7 | 0.0 | 63.8 | 37.7 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 20.5 | 0.0 | 63.6 | 37.6 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 20.8 | 0.0 | 63.3 | 37.2 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 20.7 | 0.0 | 62.8 | 37.2 | - | - |
| output_blocks.1.1.transformer_blocks.0 | block | 1 | 22.7 | 0.0 | 59.2 | 36.8 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 20.9 | 0.0 | 62.0 | 36.6 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 1 | 20.8 | 0.0 | 62.3 | 36.6 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 20.9 | 0.0 | 61.8 | 36.5 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 21.5 | 0.0 | 61.4 | 36.5 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 31.3 | 0.0 | 61.4 | 36.5 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 21.0 | 0.0 | 62.1 | 36.4 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 21.2 | 0.0 | 62.9 | 36.3 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 21.0 | 0.0 | 60.8 | 36.2 | - | - |
| middle_block.1.transformer_blocks.0 | block | 1 | 21.1 | 0.0 | 60.7 | 35.8 | - | - |
| middle_block.1.transformer_blocks.5 | block | 1 | 21.0 | 0.0 | 59.1 | 35.5 | - | - |
| middle_block.1.transformer_blocks.6 | block | 1 | 21.1 | 0.0 | 60.1 | 35.4 | - | - |
| output_blocks.1.1.transformer_blocks.6 | block | 1 | 22.0 | 0.0 | 57.3 | 35.3 | - | - |
| middle_block.1.transformer_blocks.1 | block | 1 | 20.9 | 0.0 | 60.0 | 35.3 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 17.4 | 0.0 | 61.7 | 35.3 | - | - |
| middle_block.1.transformer_blocks.2 | block | 1 | 21.4 | 0.0 | 59.3 | 35.1 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 16.6 | 0.0 | 61.6 | 35.0 | - | - |
| output_blocks.0.1.transformer_blocks.1 | block | 1 | 21.7 | 0.0 | 58.4 | 34.9 | - | - |

## Reconciliation

device 778 ms; modules' own time 7835 ms, of which 2600 ms is in modules with no cost model; wall 1360 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 206.75 | 26 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 73.18 | 34 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2653 | 70.27 | 26 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 58.16 | 19 |
| `(anonymous namespace)::launch_dequant_weight<1>(sycl::_V1::queue&, signed char const*, flo` | int8 | 599 | 56.66 | 95 |
| `gen_conv` | convolution | 131 | 45.85 | 350 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 39.27 | 280 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1265 | 37.69 | 30 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1272 | 25.66 | 20 |
| `(anonymous namespace)::launch_dequant_weight<2>(sycl::_V1::queue&, signed char const*, flo` | int8 | 60 | 22.53 | 376 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 17.69 | 84 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 14.33 | 205 |
| `conv_reorder` | convolution | 241 | 13.84 | 57 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.20 | 22 |
| `micro_sdpa` | attention | 140 | 8.77 | 63 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 7.31 | 52 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 6.89 | 49 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.38 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.27 | 90 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.01 | 86 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.49 | 21 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::FillFunctor<float>, at::d` | memory | 5401 | 3.82 | 1 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.33 | 2 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 3.06 | 41 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 2.75 | 20 |
