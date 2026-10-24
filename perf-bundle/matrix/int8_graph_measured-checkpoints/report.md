# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- wall 12958 ms (best 11524); eager 12958 ms, replayed - ms
- device busy 727 ms, utilisation 6%
- 49641 kernel launches, 1024071 operators, 248 us host time per launch
- step peak memory 5728 MB
- phases: loss 0 ms

## Findings, largest first

1. **The step is launch bound** (12232 ms)
   - the device is busy 6% of the step (727 of 12958 ms) over 49641 launches and 1024071 operators per step, 248 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
2. **Much of the module time has no cost model** (2723 ms)
   - 2723 of 13451 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
3. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (441 ms)
   - 446.8 ms per step against an ideal 6.0 ms (1% efficient, compute-bound), 36 layers; all rated pieces together leave 10564 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: conv output_blocks.in_layers (up)** (417 ms)
   - 429.7 ms per step against an ideal 12.5 ms (3% efficient, compute-bound), 9 layers; all rated pieces together leave 10564 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: linear ff.net (up)** (352 ms)
   - 357.2 ms per step against an ideal 5.4 ms (2% efficient, memory-bound), 36 layers; all rated pieces together leave 10564 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (332 ms)
   - 338.3 ms per step against an ideal 6.0 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 10564 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (down)** (316 ms)
   - 320.2 ms per step against an ideal 4.0 ms (1% efficient, compute-bound), 24 layers; all rated pieces together leave 10564 ms
   - lever: see the module table for the individual layers
8. **The int8 path's own kernels are a large share** (167 ms)
   - rotate, quantize and dequantize kernels are 23% of device time against 202 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows
9. **Dtype and copy kernels take a large share** (131 ms)
   - 18% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
10. **Many small memory-bound kernels** (122 ms)
   - 34% of device time (29819 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
11. **Module times do not add up to the device time** (0 ms)
   - modules account for 13451 ms of 727 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 13102 | 202.1 | 28% |
| int8 | 4456 | 166.7 | 23% |
| copy | 7864 | 131.0 | 18% |
| elementwise | 13870 | 88.8 | 12% |
| convolution | 946 | 70.1 | 10% |
| other | 1214 | 38.0 | 5% |
| norm | 1268 | 17.6 | 2% |
| memory | 6798 | 5.7 | 1% |
| attention | 104 | 5.3 | 1% |
| reduce | 19 | 1.4 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 95.1 TFLOPS, int8 184.9 TOPS, copy 366.2 GB/s, dispatch 11.0 us, queued launch 11.0 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention | transformer_blocks.attn2 | up | 36 | 446.8 | 6.0 | 1% | compute | 440.8 |
| conv | output_blocks.in_layers | up | 9 | 429.7 | 12.5 | 3% | compute | 417.2 |
| linear int8 | ff.net | up | 36 | 357.2 | 5.4 | 2% | memory | 351.8 |
| attention | transformer_blocks.attn1 | up | 36 | 338.3 | 6.0 | 2% | compute | 332.3 |
| attention | transformer_blocks.attn2 | down | 24 | 320.2 | 4.0 | 1% | compute | 316.2 |
| conv | output_blocks.skip_connection | up | 9 | 297.2 | 1.5 | 1% | memory | 295.7 |
| linear int8 | attn1.to_out | up | 30 | 280.1 | 1.3 | 0% | memory | 278.7 |
| linear int8 | attn2.to_q | up | 30 | 280.0 | 1.3 | 0% | memory | 278.7 |
| linear int8 | attn2.to_out | up | 30 | 278.4 | 1.3 | 0% | memory | 277.0 |
| linear int8 | net.proj | up | 30 | 283.1 | 8.1 | 3% | memory | 275.0 |
| linear int8 | attn1.to_v | up | 30 | 274.1 | 1.3 | 0% | memory | 272.8 |
| linear int8 | attn2.to_v | up | 36 | 273.9 | 1.3 | 0% | memory | 272.6 |
| linear int8 | attn1.to_q | up | 30 | 272.2 | 1.3 | 0% | memory | 270.9 |
| linear int8 | attn1.to_k | up | 30 | 269.0 | 1.3 | 0% | memory | 267.7 |
| attention | transformer_blocks.attn1 | down | 24 | 267.8 | 4.0 | 2% | compute | 263.8 |
| linear int8 | attn2.to_k | up | 36 | 252.3 | 1.3 | 1% | memory | 251.0 |
| linear int8 | ff.net | down | 24 | 250.3 | 3.7 | 1% | memory | 246.6 |
| linear int8 | attn1.to_out | down | 20 | 195.8 | 0.9 | 0% | memory | 194.8 |
| linear int8 | attn2.to_out | down | 20 | 193.8 | 0.9 | 0% | memory | 192.9 |
| linear int8 | net.proj | down | 20 | 197.5 | 5.6 | 3% | memory | 192.0 |
| linear int8 | attn1.to_q | down | 20 | 190.5 | 0.9 | 0% | memory | 189.6 |
| linear int8 | attn1.to_k | down | 20 | 189.9 | 0.9 | 0% | memory | 188.9 |
| linear int8 | attn1.to_v | down | 20 | 188.4 | 0.9 | 0% | memory | 187.5 |
| linear int8 | attn2.to_q | down | 20 | 187.3 | 0.9 | 0% | memory | 186.4 |
| feedforward | transformer_blocks.ff | up | 36 | 201.8 | 19.0 | 9% | memory | 182.8 |
| linear int8 | attn2.to_v | down | 24 | 183.3 | 0.9 | 0% | memory | 182.4 |
| linear int8 | attn2.to_k | down | 24 | 183.2 | 0.9 | 0% | memory | 182.3 |
| conv | input_blocks.op | down | 2 | 165.0 | 0.2 | 0% | compute | 164.7 |
| feedforward | transformer_blocks.ff | down | 24 | 155.1 | 12.9 | 8% | memory | 142.3 |
| conv | input_blocks.in_layers | down | 6 | 144.4 | 2.4 | 2% | compute | 141.9 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.6.0.op | conv | 2 | 35.7 | 0.0 | 112.9 | 148.6 | 0.1 | 0% |
| input_blocks.7.1.transformer_blocks.2 | block | 2 | 49.2 | 0.0 | 153.2 | 90.8 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 2 | 43.3 | 0.0 | 155.2 | 90.5 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 2 | 44.5 | 0.0 | 147.6 | 86.7 | - | - |
| middle_block.1.transformer_blocks.1 | block | 2 | 47.0 | 0.0 | 141.4 | 84.8 | - | - |
| middle_block.1.transformer_blocks.9 | block | 2 | 47.6 | 0.0 | 141.9 | 84.5 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 2 | 46.4 | 0.0 | 141.2 | 83.5 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 2 | 44.5 | 0.0 | 142.8 | 82.8 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 2 | 45.9 | 0.0 | 139.2 | 82.1 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 2 | 36.4 | 0.0 | 141.6 | 81.7 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 2 | 35.9 | 0.0 | 143.1 | 80.0 | - | - |
| middle_block.1.transformer_blocks.6 | block | 2 | 46.1 | 0.0 | 134.6 | 79.6 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 2 | 40.5 | 0.0 | 140.0 | 78.9 | - | - |
| output_blocks.0.1.transformer_blocks.5 | block | 2 | 48.8 | 0.0 | 130.6 | 78.5 | - | - |
| output_blocks.0.1.transformer_blocks.1 | block | 2 | 46.1 | 0.0 | 129.4 | 76.7 | - | - |
| output_blocks.1.1.transformer_blocks.7 | block | 2 | 52.2 | 0.0 | 130.5 | 76.4 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 2 | 36.5 | 0.0 | 132.5 | 76.1 | - | - |
| output_blocks.2.1.transformer_blocks.3 | block | 2 | 51.8 | 0.0 | 124.7 | 74.7 | - | - |
| output_blocks.2.1.transformer_blocks.8 | block | 2 | 52.9 | 0.0 | 123.3 | 74.2 | - | - |
| output_blocks.2.1.transformer_blocks.2 | block | 2 | 49.6 | 0.0 | 125.4 | 73.4 | - | - |
| output_blocks.2.1.transformer_blocks.4 | block | 2 | 49.5 | 0.0 | 122.7 | 72.9 | - | - |
| output_blocks.3.0.in_layers.2 | conv | 2 | 38.2 | 1.9 | 27.2 | 67.4 | 2.2 | 3% |
| output_blocks.4.0.in_layers.2 | conv | 2 | 37.9 | 1.5 | 27.0 | 66.4 | 1.5 | 2% |
| output_blocks.2.2.conv | conv | 2 | 38.5 | 0.0 | 27.7 | 66.2 | 1.9 | 3% |
| output_blocks.4.1.transformer_blocks.0 | block | 2 | 41.1 | 0.0 | 112.9 | 65.9 | - | - |
| output_blocks.5.1.proj_out | linear | 2 | 2.7 | 0.0 | 62.6 | 65.3 | 0.1 | 0% |
| output_blocks.6.0.in_layers.2 | conv | 2 | 35.8 | 1.5 | 27.8 | 65.2 | 2.2 | 3% |
| output_blocks.5.0.in_layers.2 | conv | 2 | 37.4 | 1.3 | 24.6 | 63.3 | 1.1 | 2% |
| output_blocks.3.1.transformer_blocks.0 | block | 2 | 43.5 | 0.0 | 108.3 | 62.6 | - | - |
| output_blocks.5.1.transformer_blocks.0 | block | 2 | 45.4 | 0.0 | 106.8 | 62.1 | - | - |

## Reconciliation

device 727 ms; modules' own time 13451 ms, of which 2723 ms is in modules with no cost model; wall 12958 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 12822 | 188.77 | 15 |
| `(anonymous namespace)::launch_dequant_weight<1>(sycl::_V1::queue&, signed char const*, flo` | int8 | 1008 | 82.07 | 81 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 4082 | 59.62 | 15 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 3238 | 54.05 | 17 |
| `gen_conv` | convolution | 222 | 51.97 | 234 |
| `(anonymous namespace)::launch_dequant_weight<2>(sycl::_V1::queue&, signed char const*, flo` | int8 | 120 | 46.85 | 390 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 4686 | 39.13 | 8 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 280 | 32.41 | 116 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1624 | 19.85 | 12 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1664 | 17.14 | 10 |
| `conv_reorder` | convolution | 444 | 17.07 | 38 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 140 | 11.45 | 82 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 240 | 10.49 | 44 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 332 | 8.78 | 26 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 4927 | 8.37 | 2 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 576 | 7.14 | 12 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 420 | 5.77 | 14 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 140 | 5.48 | 39 |
| `micro_sdpa` | attention | 104 | 5.32 | 51 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 2912 | 5.31 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::FillFunctor<float>, at::d` | memory | 6718 | 4.91 | 1 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 192 | 4.47 | 23 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 140 | 4.08 | 29 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 116 | 3.95 | 34 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 40 | 2.83 | 71 |
