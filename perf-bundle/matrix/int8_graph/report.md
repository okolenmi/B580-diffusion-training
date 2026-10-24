# Performance report (xpu)

Window: 3 steps (2 eager, 1 replayed). All times are per step.

## Step budget

- wall 10967 ms (best 9458); eager 9493 ms, replayed 13917 ms
- device busy 744 ms, utilisation 7%
- 32959 kernel launches, 657072 operators, 327 us host time per launch
- step peak memory 4067 MB
- phases: loss 0 ms

## Findings, largest first

1. **The step is launch bound** (10223 ms)
   - the device is busy 7% of the step (744 of 10967 ms) over 32959 launches and 657072 operators per step, 327 us of host time per launch
   - lever: graph replay is already running for some steps; compare replayed and eager step times above
2. **Slowest pieces by recoverable time: conv output_blocks.in_layers (up)** (702 ms)
   - 722.5 ms per step against an ideal 20.7 ms (3% efficient, compute-bound), 9 layers; all rated pieces together leave 13107 ms
   - lever: see the module table for the individual layers
3. **Slowest pieces by recoverable time: conv output_blocks.skip_connection (up)** (482 ms)
   - 484.2 ms per step against an ideal 2.5 ms (1% efficient, compute-bound), 9 layers; all rated pieces together leave 13107 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (448 ms)
   - 458.0 ms per step against an ideal 9.8 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 13107 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: attention transformer_blocks.attn2 (up)** (429 ms)
   - 438.9 ms per step against an ideal 9.8 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 13107 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: linear ff.net (up)** (385 ms)
   - 394.5 ms per step against an ideal 9.1 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 13107 ms
   - lever: see the module table for the individual layers
7. **Dtype and copy kernels take a large share** (157 ms)
   - 21% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
8. **Many small memory-bound kernels** (145 ms)
   - 39% of device time (20397 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
9. **The int8 path's own kernels are a large share** (139 ms)
   - rotate, quantize and dequantize kernels are 19% of device time against 194 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 194.1 | 26% |
| copy | 5116 | 157.5 | 21% |
| int8 | 3203 | 138.6 | 19% |
| elementwise | 9038 | 106.5 | 14% |
| convolution | 533 | 76.8 | 10% |
| other | 623 | 36.0 | 5% |
| norm | 782 | 18.7 | 3% |
| attention | 140 | 9.1 | 1% |
| memory | 5443 | 5.1 | 1% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 95.1 TFLOPS, int8 184.9 TOPS, copy 366.2 GB/s, dispatch 11.0 us, queued launch 11.0 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| conv | output_blocks.in_layers | up | 9 | 722.5 | 20.7 | 3% | compute | 701.8 |
| conv | output_blocks.skip_connection | up | 9 | 484.2 | 2.5 | 1% | compute | 481.8 |
| attention | transformer_blocks.attn1 | up | 36 | 458.0 | 9.8 | 2% | compute | 448.2 |
| attention | transformer_blocks.attn2 | up | 36 | 438.9 | 9.8 | 2% | compute | 429.1 |
| linear int8 | ff.net | up | 36 | 394.5 | 9.1 | 2% | compute | 385.4 |
| linear int8 | attn2.to_k | up | 36 | 384.4 | 1.5 | 0% | memory | 382.9 |
| attention | transformer_blocks.attn1 | down | 24 | 386.9 | 6.5 | 2% | compute | 380.4 |
| attention | transformer_blocks.attn2 | down | 24 | 380.0 | 6.5 | 2% | compute | 373.5 |
| linear int8 | attn2.to_v | up | 36 | 372.4 | 1.5 | 0% | memory | 370.9 |
| linear int8 | attn2.to_out | up | 30 | 328.0 | 2.1 | 1% | memory | 326.0 |
| linear int8 | attn1.to_out | up | 30 | 324.2 | 2.1 | 1% | memory | 322.1 |
| linear int8 | attn1.to_q | up | 30 | 319.6 | 2.1 | 1% | memory | 317.5 |
| linear int8 | net.proj | up | 30 | 331.7 | 15.0 | 5% | compute | 316.7 |
| linear int8 | attn1.to_k | up | 30 | 315.9 | 2.1 | 1% | memory | 313.8 |
| linear int8 | attn1.to_v | up | 30 | 315.5 | 2.1 | 1% | memory | 313.4 |
| linear int8 | attn2.to_q | up | 30 | 315.4 | 2.1 | 1% | memory | 313.3 |
| conv | input_blocks.op | down | 2 | 292.9 | 0.4 | 0% | compute | 292.5 |
| linear int8 | ff.net | down | 24 | 275.9 | 6.1 | 2% | compute | 269.8 |
| linear int8 | attn2.to_k | down | 24 | 263.2 | 1.0 | 0% | memory | 262.2 |
| linear int8 | attn2.to_v | down | 24 | 262.9 | 1.0 | 0% | memory | 261.9 |
| linear int8 | attn1.to_k | down | 20 | 245.3 | 1.4 | 1% | memory | 243.9 |
| linear int8 | attn2.to_out | down | 20 | 232.2 | 1.4 | 1% | memory | 230.8 |
| feedforward | transformer_blocks.ff | up | 36 | 261.6 | 33.1 | 13% | memory | 228.5 |
| linear int8 | attn1.to_out | down | 20 | 226.8 | 1.4 | 1% | memory | 225.5 |
| linear int8 | net.proj | down | 20 | 234.8 | 10.0 | 4% | compute | 224.8 |
| conv | input_blocks.in_layers | down | 6 | 227.0 | 4.5 | 2% | compute | 222.5 |
| linear int8 | attn1.to_q | down | 20 | 222.5 | 1.4 | 1% | memory | 221.1 |
| linear int8 | attn2.to_q | down | 20 | 221.5 | 1.4 | 1% | memory | 220.1 |
| linear int8 | attn1.to_v | down | 20 | 218.8 | 1.4 | 1% | memory | 217.4 |
| feedforward | transformer_blocks.ff | down | 24 | 224.3 | 22.0 | 10% | memory | 202.3 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.6.0.op | conv | 2 | 47.4 | 0.0 | 181.4 | 228.8 | 0.2 | 0% |
| output_blocks.3.0.in_layers.2 | conv | 2 | 64.1 | 2.0 | 44.4 | 110.6 | 3.3 | 3% |
| output_blocks.6.0.in_layers.2 | conv | 2 | 61.8 | 2.6 | 42.3 | 106.7 | 3.3 | 3% |
| output_blocks.4.0.in_layers.2 | conv | 2 | 61.7 | 1.5 | 42.6 | 105.8 | 2.2 | 2% |
| output_blocks.5.0.in_layers.2 | conv | 2 | 61.0 | 1.6 | 39.2 | 101.8 | 1.6 | 2% |
| output_blocks.2.0.in_layers.2 | conv | 2 | 60.5 | 2.3 | 38.0 | 100.8 | 1.6 | 2% |
| input_blocks.7.0.in_layers.2 | conv | 2 | 56.0 | 1.9 | 39.3 | 97.2 | 0.5 | 1% |
| input_blocks.7.1.transformer_blocks.0 | block | 2 | 44.5 | 0.0 | 156.8 | 91.7 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 2 | 39.5 | 0.0 | 150.1 | 91.3 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 2 | 39.3 | 0.0 | 150.0 | 91.2 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 2 | 37.2 | 0.0 | 151.8 | 90.2 | - | - |
| input_blocks.7.1.transformer_blocks.1 | block | 2 | 38.3 | 0.0 | 153.0 | 90.1 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 2 | 37.2 | 0.0 | 150.6 | 89.3 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 2 | 45.3 | 0.0 | 150.6 | 88.1 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 2 | 37.7 | 0.0 | 149.1 | 87.4 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 2 | 39.2 | 0.0 | 146.8 | 87.2 | - | - |
| input_blocks.7.1.transformer_blocks.9 | block | 2 | 39.8 | 0.0 | 146.3 | 85.5 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 2 | 38.5 | 0.0 | 146.2 | 85.5 | - | - |
| input_blocks.7.1.transformer_blocks.8 | block | 2 | 37.7 | 0.0 | 145.4 | 85.4 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 2 | 38.2 | 0.0 | 144.0 | 85.2 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 2 | 39.8 | 0.0 | 142.9 | 85.0 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 2 | 39.4 | 0.0 | 144.0 | 84.2 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 2 | 39.8 | 0.0 | 140.9 | 83.6 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 2 | 39.0 | 0.0 | 143.3 | 83.5 | - | - |
| input_blocks.4.1.transformer_blocks.1 | block | 2 | 30.0 | 0.0 | 142.8 | 82.9 | - | - |
| input_blocks.4.1.transformer_blocks.0 | block | 2 | 36.5 | 0.0 | 140.8 | 81.8 | - | - |
| output_blocks.5.2.conv | conv | 2 | 38.9 | 0.0 | 42.4 | 81.3 | 2.9 | 4% |
| input_blocks.8.1.transformer_blocks.7 | block | 2 | 39.7 | 0.0 | 136.8 | 81.2 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 2 | 39.9 | 0.0 | 139.6 | 81.1 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 2 | 29.6 | 0.0 | 140.5 | 81.0 | - | - |

## Reconciliation

device 744 ms; modules - ms (unrated - ms). Module times need eager steps: a window with replayed steps has no module table.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 187.07 | 24 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 69.68 | 32 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2653 | 67.17 | 25 |
| `gen_conv` | convolution | 131 | 60.75 | 464 |
| `(anonymous namespace)::launch_dequant_weight<1>(sycl::_V1::queue&, signed char const*, flo` | int8 | 599 | 53.57 | 89 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 52.53 | 17 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1252 | 33.65 | 27 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 31.35 | 224 |
| `(anonymous namespace)::launch_dequant_weight<2>(sycl::_V1::queue&, signed char const*, flo` | int8 | 60 | 25.99 | 433 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1272 | 24.87 | 20 |
| `conv_reorder` | convolution | 262 | 14.72 | 56 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 14.32 | 68 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 12.96 | 185 |
| `micro_sdpa` | attention | 140 | 9.08 | 65 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 8.68 | 21 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 7.63 | 2 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 6.99 | 50 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 6.09 | 44 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.49 | 78 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 5.18 | 25 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.07 | 72 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::FillFunctor<float>, at::d` | memory | 5401 | 4.40 | 1 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.40 | 2 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 3.15 | 95 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 2.85 | 39 |
