# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- wall 11086 ms (best 9369); eager 11086 ms, replayed - ms
- device busy 805 ms, utilisation 7%
- 32959 kernel launches, 656960 operators, 333 us host time per launch
- step peak memory 4066 MB
- phases: loss 0 ms

## Findings, largest first

1. **The step is launch bound** (10280 ms)
   - the device is busy 7% of the step (805 of 11086 ms) over 32959 launches and 656960 operators per step, 333 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
2. **Much of the module time has no cost model** (3955 ms)
   - 3955 of 12963 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
3. **Slowest pieces by recoverable time: conv output_blocks.in_layers (up)** (457 ms)
   - 471.0 ms per step against an ideal 13.8 ms (3% efficient, compute-bound), 9 layers; all rated pieces together leave 8827 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: conv output_blocks.skip_connection (up)** (314 ms)
   - 316.0 ms per step against an ideal 1.6 ms (1% efficient, compute-bound), 9 layers; all rated pieces together leave 8827 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (285 ms)
   - 291.5 ms per step against an ideal 6.5 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 8827 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (282 ms)
   - 288.8 ms per step against an ideal 6.5 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 8827 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (down)** (259 ms)
   - 263.5 ms per step against an ideal 4.3 ms (2% efficient, compute-bound), 24 layers; all rated pieces together leave 8827 ms
   - lever: see the module table for the individual layers
8. **Dtype and copy kernels take a large share** (184 ms)
   - 23% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
9. **The int8 path's own kernels are a large share** (169 ms)
   - rotate, quantize and dequantize kernels are 21% of device time against 195 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows
10. **Many small memory-bound kernels** (160 ms)
   - 40% of device time (20397 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
11. **Module times do not add up to the device time** (0 ms)
   - modules account for 12963 ms of 805 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 194.5 | 24% |
| copy | 5116 | 183.8 | 23% |
| int8 | 3203 | 169.3 | 21% |
| elementwise | 9038 | 109.0 | 14% |
| convolution | 533 | 75.0 | 9% |
| other | 623 | 38.2 | 5% |
| norm | 782 | 19.3 | 2% |
| attention | 140 | 9.0 | 1% |
| memory | 5443 | 5.4 | 1% |
| reduce | 18 | 1.6 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 95.1 TFLOPS, int8 184.9 TOPS, copy 366.2 GB/s, dispatch 11.0 us, queued launch 11.0 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| conv | output_blocks.in_layers | up | 9 | 471.0 | 13.8 | 3% | compute | 457.1 |
| conv | output_blocks.skip_connection | up | 9 | 316.0 | 1.6 | 1% | compute | 314.3 |
| attention | transformer_blocks.attn1 | up | 36 | 291.5 | 6.5 | 2% | compute | 285.0 |
| attention | transformer_blocks.attn2 | up | 36 | 288.8 | 6.5 | 2% | compute | 282.3 |
| attention | transformer_blocks.attn1 | down | 24 | 263.5 | 4.3 | 2% | compute | 259.2 |
| linear int8 | ff.net | up | 36 | 262.1 | 6.1 | 2% | compute | 256.0 |
| attention | transformer_blocks.attn2 | down | 24 | 252.6 | 4.3 | 2% | compute | 248.2 |
| linear int8 | attn2.to_k | up | 36 | 248.0 | 1.0 | 0% | memory | 247.0 |
| linear int8 | attn2.to_v | up | 36 | 246.0 | 1.0 | 0% | memory | 245.0 |
| linear int8 | net.proj | up | 30 | 235.0 | 10.0 | 4% | compute | 225.0 |
| linear int8 | attn2.to_out | up | 30 | 226.0 | 1.4 | 1% | memory | 224.6 |
| linear int8 | attn1.to_out | up | 30 | 214.9 | 1.4 | 1% | memory | 213.5 |
| linear int8 | attn1.to_q | up | 30 | 214.4 | 1.4 | 1% | memory | 213.1 |
| linear int8 | attn1.to_k | up | 30 | 210.0 | 1.4 | 1% | memory | 208.6 |
| linear int8 | attn2.to_q | up | 30 | 207.6 | 1.4 | 1% | memory | 206.2 |
| linear int8 | attn1.to_v | up | 30 | 207.1 | 1.4 | 1% | memory | 205.7 |
| conv | input_blocks.op | down | 2 | 203.1 | 0.2 | 0% | compute | 202.8 |
| linear int8 | attn2.to_v | down | 24 | 190.7 | 0.7 | 0% | memory | 190.1 |
| linear int8 | ff.net | down | 24 | 187.4 | 4.0 | 2% | compute | 183.4 |
| linear int8 | attn2.to_k | down | 24 | 177.3 | 0.7 | 0% | memory | 176.6 |
| linear int8 | attn2.to_out | down | 20 | 156.4 | 0.9 | 1% | memory | 155.5 |
| linear int8 | net.proj | down | 20 | 160.0 | 6.7 | 4% | compute | 153.4 |
| linear int8 | attn1.to_k | down | 20 | 154.3 | 0.9 | 1% | memory | 153.4 |
| linear int8 | attn1.to_out | down | 20 | 153.9 | 0.9 | 1% | memory | 153.0 |
| conv | input_blocks.in_layers | down | 6 | 155.4 | 3.0 | 2% | compute | 152.4 |
| linear int8 | attn2.to_q | down | 20 | 152.5 | 0.9 | 1% | memory | 151.6 |
| linear int8 | attn1.to_q | down | 20 | 151.6 | 0.9 | 1% | memory | 150.7 |
| linear int8 | attn1.to_v | down | 20 | 150.2 | 0.9 | 1% | memory | 149.2 |
| feedforward | transformer_blocks.ff | up | 36 | 169.1 | 22.0 | 13% | memory | 147.1 |
| feedforward | transformer_blocks.ff | down | 24 | 153.4 | 14.7 | 10% | memory | 138.7 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.6.0.op | conv | 1 | 31.4 | 0.0 | 129.4 | 160.7 | 0.1 | 0% |
| output_blocks.4.0.in_layers.2 | conv | 1 | 45.3 | 1.1 | 27.4 | 73.9 | 1.5 | 2% |
| output_blocks.3.0.in_layers.2 | conv | 1 | 41.5 | 1.5 | 27.9 | 70.9 | 2.2 | 3% |
| output_blocks.6.0.in_layers.2 | conv | 1 | 37.6 | 1.1 | 29.6 | 68.4 | 2.2 | 3% |
| output_blocks.5.0.in_layers.2 | conv | 1 | 41.0 | 0.9 | 26.4 | 68.3 | 1.1 | 2% |
| input_blocks.7.0.in_layers.2 | conv | 1 | 37.6 | 1.4 | 28.4 | 67.4 | 0.4 | 1% |
| output_blocks.2.0.in_layers.2 | conv | 1 | 38.9 | 1.6 | 24.8 | 65.2 | 1.1 | 2% |
| input_blocks.8.1.transformer_blocks.0 | block | 1 | 25.2 | 0.0 | 105.9 | 61.7 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 30.1 | 0.0 | 102.5 | 60.9 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 25.4 | 0.0 | 103.0 | 60.6 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 26.4 | 0.0 | 101.6 | 60.4 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 1 | 32.3 | 0.0 | 100.8 | 60.2 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 26.5 | 0.0 | 101.0 | 60.2 | - | - |
| middle_block.1.transformer_blocks.5 | block | 1 | 26.9 | 0.0 | 89.4 | 60.1 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 1 | 25.5 | 0.0 | 100.2 | 59.7 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 25.5 | 0.0 | 100.6 | 59.5 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 27.8 | 0.0 | 100.5 | 59.0 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 25.4 | 0.0 | 99.0 | 58.8 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 1 | 19.4 | 0.0 | 100.6 | 58.5 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 1 | 25.5 | 0.0 | 98.4 | 58.2 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 1 | 25.1 | 0.0 | 99.8 | 57.8 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 26.9 | 0.0 | 97.2 | 57.8 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 39.4 | 0.0 | 100.7 | 57.8 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 1 | 25.1 | 0.0 | 96.4 | 57.3 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 1 | 25.7 | 0.0 | 98.8 | 57.0 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 27.0 | 0.0 | 97.6 | 56.8 | - | - |
| input_blocks.5.1.transformer_blocks.1 | block | 1 | 20.0 | 0.0 | 96.5 | 56.3 | - | - |
| output_blocks.2.1.transformer_blocks.7 | block | 1 | 30.2 | 0.0 | 76.4 | 56.0 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 26.9 | 0.0 | 94.9 | 55.7 | - | - |
| input_blocks.8.1.transformer_blocks.9 | block | 1 | 26.3 | 0.0 | 94.6 | 55.6 | - | - |

## Reconciliation

device 805 ms; modules' own time 12963 ms, of which 3955 ms is in modules with no cost model; wall 11086 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 187.13 | 24 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 81.86 | 38 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2653 | 81.31 | 31 |
| `(anonymous namespace)::launch_dequant_weight<1>(sycl::_V1::queue&, signed char const*, flo` | int8 | 599 | 69.92 | 117 |
| `gen_conv` | convolution | 131 | 58.15 | 444 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 53.50 | 18 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1252 | 44.28 | 35 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 31.63 | 226 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1272 | 28.66 | 23 |
| `(anonymous namespace)::launch_dequant_weight<2>(sycl::_V1::queue&, signed char const*, flo` | int8 | 60 | 25.91 | 432 |
| `conv_reorder` | convolution | 262 | 15.46 | 59 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 13.97 | 67 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 12.92 | 185 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 9.16 | 22 |
| `micro_sdpa` | attention | 140 | 9.01 | 64 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 7.40 | 53 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 7.28 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 6.07 | 87 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 5.86 | 42 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 5.43 | 26 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.00 | 71 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::FillFunctor<float>, at::d` | memory | 5401 | 4.65 | 1 |
| `cute::MhaDotDoOName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 3.77 | 27 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 3.30 | 100 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.25 | 2 |
