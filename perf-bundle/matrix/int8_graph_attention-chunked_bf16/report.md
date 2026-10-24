# Performance report (xpu)

Window: 3 steps (2 eager, 1 replayed). All times are per step.

## Step budget

- wall 8688 ms (best 7227); eager 7262 ms, replayed 11541 ms
- device busy 695 ms, utilisation 8%
- 32959 kernel launches, 657074 operators, 259 us host time per launch
- step peak memory 4067 MB
- phases: loss 0 ms

## Findings, largest first

1. **The step is launch bound** (7993 ms)
   - the device is busy 8% of the step (695 of 8688 ms) over 32959 launches and 657074 operators per step, 259 us of host time per launch
   - lever: graph replay is already running for some steps; compare replayed and eager step times above
2. **Slowest pieces by recoverable time: conv output_blocks.in_layers (up)** (666 ms)
   - 686.3 ms per step against an ideal 20.7 ms (3% efficient, compute-bound), 9 layers; all rated pieces together leave 10507 ms
   - lever: see the module table for the individual layers
3. **Slowest pieces by recoverable time: conv output_blocks.skip_connection (up)** (450 ms)
   - 452.2 ms per step against an ideal 2.5 ms (1% efficient, compute-bound), 9 layers; all rated pieces together leave 10507 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: linear attn2.to_v (up)** (306 ms)
   - 307.0 ms per step against an ideal 1.5 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 10507 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: linear ff.net (up)** (305 ms)
   - 313.6 ms per step against an ideal 9.1 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 10507 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up)** (302 ms)
   - 311.3 ms per step against an ideal 9.8 ms (3% efficient, compute-bound), 36 layers; all rated pieces together leave 10507 ms
   - lever: see the module table for the individual layers
7. **Dtype and copy kernels take a large share** (146 ms)
   - 21% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
8. **Many small memory-bound kernels** (135 ms)
   - 39% of device time (20397 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
9. **The int8 path's own kernels are a large share** (121 ms)
   - rotate, quantize and dequantize kernels are 17% of device time against 192 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 192.2 | 28% |
| copy | 5116 | 146.5 | 21% |
| int8 | 3203 | 120.5 | 17% |
| elementwise | 9038 | 100.1 | 14% |
| convolution | 533 | 69.7 | 10% |
| other | 623 | 34.7 | 5% |
| norm | 782 | 17.2 | 2% |
| attention | 140 | 8.0 | 1% |
| memory | 5443 | 4.8 | 1% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 95.1 TFLOPS, int8 184.9 TOPS, copy 366.2 GB/s, dispatch 11.0 us, queued launch 11.0 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| conv | output_blocks.in_layers | up | 9 | 686.3 | 20.7 | 3% | compute | 665.6 |
| conv | output_blocks.skip_connection | up | 9 | 452.2 | 2.5 | 1% | compute | 449.8 |
| linear int8 | attn2.to_v | up | 36 | 307.0 | 1.5 | 0% | memory | 305.5 |
| linear int8 | ff.net | up | 36 | 313.6 | 9.1 | 3% | compute | 304.5 |
| attention | transformer_blocks.attn1 | up | 36 | 311.3 | 9.8 | 3% | compute | 301.5 |
| linear int8 | attn2.to_k | up | 36 | 296.4 | 1.5 | 1% | memory | 294.9 |
| attention | transformer_blocks.attn2 | up | 36 | 296.5 | 9.8 | 3% | compute | 286.8 |
| conv | input_blocks.op | down | 2 | 279.7 | 0.4 | 0% | compute | 279.4 |
| attention | transformer_blocks.attn1 | down | 24 | 268.6 | 6.5 | 2% | compute | 262.1 |
| linear int8 | attn2.to_out | up | 30 | 259.6 | 2.1 | 1% | memory | 257.5 |
| attention | transformer_blocks.attn2 | down | 24 | 263.2 | 6.5 | 2% | compute | 256.7 |
| linear int8 | attn1.to_out | up | 30 | 258.8 | 2.1 | 1% | memory | 256.7 |
| linear int8 | net.proj | up | 30 | 271.0 | 15.0 | 6% | compute | 256.1 |
| linear int8 | attn1.to_q | up | 30 | 256.2 | 2.1 | 1% | memory | 254.1 |
| linear int8 | attn2.to_q | up | 30 | 251.4 | 2.1 | 1% | memory | 249.3 |
| linear int8 | attn1.to_v | up | 30 | 248.1 | 2.1 | 1% | memory | 246.0 |
| linear int8 | attn1.to_k | up | 30 | 247.7 | 2.1 | 1% | memory | 245.6 |
| linear int8 | ff.net | down | 24 | 227.4 | 6.1 | 3% | compute | 221.4 |
| linear int8 | attn2.to_v | down | 24 | 213.5 | 1.0 | 0% | memory | 212.5 |
| linear int8 | attn2.to_k | down | 24 | 212.0 | 1.0 | 0% | memory | 211.0 |
| conv | input_blocks.in_layers | down | 6 | 213.2 | 4.5 | 2% | compute | 208.7 |
| linear int8 | attn2.to_out | down | 20 | 200.8 | 1.4 | 1% | memory | 199.4 |
| linear int8 | attn1.to_out | down | 20 | 185.4 | 1.4 | 1% | memory | 184.1 |
| linear int8 | attn1.to_k | down | 20 | 184.4 | 1.4 | 1% | memory | 183.1 |
| linear int8 | net.proj | down | 20 | 191.9 | 10.0 | 5% | compute | 181.9 |
| linear int8 | attn1.to_q | down | 20 | 182.3 | 1.4 | 1% | memory | 180.9 |
| linear int8 | attn2.to_q | down | 20 | 179.8 | 1.4 | 1% | memory | 178.5 |
| linear int8 | attn1.to_v | down | 20 | 178.5 | 1.4 | 1% | memory | 177.1 |
| feedforward | transformer_blocks.ff | up | 36 | 176.1 | 33.1 | 19% | memory | 143.0 |
| conv | output_blocks.out_layers | up | 9 | 141.3 | 9.8 | 7% | compute | 131.4 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.6.0.op | conv | 2 | 44.9 | 0.0 | 174.9 | 219.7 | 0.2 | 0% |
| output_blocks.3.0.in_layers.2 | conv | 2 | 60.8 | 2.1 | 41.3 | 104.2 | 3.3 | 3% |
| output_blocks.4.0.in_layers.2 | conv | 2 | 59.5 | 1.5 | 40.4 | 101.4 | 2.2 | 2% |
| output_blocks.6.0.in_layers.2 | conv | 2 | 56.6 | 2.0 | 41.5 | 100.2 | 3.3 | 3% |
| output_blocks.5.0.in_layers.2 | conv | 2 | 59.3 | 1.2 | 38.1 | 98.5 | 1.6 | 2% |
| output_blocks.2.0.in_layers.2 | conv | 2 | 58.2 | 2.2 | 36.0 | 96.4 | 1.6 | 2% |
| input_blocks.7.0.in_layers.2 | conv | 2 | 53.4 | 1.2 | 37.7 | 92.3 | 0.5 | 1% |
| output_blocks.5.2.conv | conv | 2 | 37.8 | 0.0 | 38.2 | 76.0 | 2.9 | 4% |
| output_blocks.2.0.skip_connection | conv | 2 | 42.4 | 1.0 | 30.2 | 73.6 | 0.2 | 0% |
| output_blocks.2.1.proj_out | linear | 2 | 2.9 | 0.0 | 68.8 | 71.8 | 0.0 | 0% |
| input_blocks.7.0.skip_connection | conv | 2 | 41.7 | 1.3 | 27.5 | 70.6 | 0.1 | 0% |
| input_blocks.7.1.transformer_blocks.1 | block | 2 | 31.0 | 0.0 | 113.8 | 68.1 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 2 | 38.9 | 0.0 | 112.0 | 66.0 | - | - |
| input_blocks.8.1.transformer_blocks.1 | block | 2 | 32.9 | 0.0 | 110.2 | 65.9 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 2 | 31.6 | 0.0 | 110.4 | 65.6 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 2 | 31.8 | 0.0 | 111.3 | 65.3 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 2 | 31.2 | 0.0 | 111.4 | 65.2 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 2 | 38.9 | 0.0 | 109.4 | 65.0 | - | - |
| out.2 | conv | 2 | 18.1 | 0.0 | 46.3 | 64.4 | 0.1 | 0% |
| output_blocks.5.0.skip_connection | conv | 2 | 29.9 | 0.8 | 33.6 | 64.3 | 0.2 | 0% |
| input_blocks.8.1.transformer_blocks.0 | block | 2 | 31.9 | 0.0 | 106.8 | 64.2 | - | - |
| output_blocks.0.0.in_layers.2 | conv | 2 | 58.6 | 2.9 | 2.6 | 64.1 | 2.2 | 3% |
| input_blocks.8.1.transformer_blocks.2 | block | 2 | 32.4 | 0.0 | 109.0 | 63.5 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 2 | 32.5 | 0.0 | 108.9 | 63.2 | - | - |
| output_blocks.3.0.skip_connection | conv | 2 | 31.1 | 0.8 | 30.8 | 62.7 | 0.4 | 1% |
| input_blocks.7.1.transformer_blocks.9 | block | 2 | 32.3 | 0.0 | 107.9 | 62.4 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 2 | 32.7 | 0.0 | 107.5 | 62.4 | - | - |
| middle_block.1.transformer_blocks.3 | block | 2 | 33.8 | 0.0 | 106.7 | 62.3 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 2 | 32.2 | 0.0 | 107.0 | 62.3 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 2 | 32.6 | 0.0 | 106.6 | 62.1 | - | - |

## Reconciliation

device 695 ms; modules - ms (unrated - ms). Module times need eager steps: a window with replayed steps has no module table.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 185.64 | 23 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2147 | 65.56 | 31 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2653 | 62.40 | 24 |
| `gen_conv` | convolution | 131 | 54.71 | 418 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 48.86 | 16 |
| `(anonymous namespace)::launch_dequant_weight<1>(sycl::_V1::queue&, signed char const*, flo` | int8 | 599 | 40.12 | 67 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1252 | 31.87 | 25 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 30.06 | 215 |
| `(anonymous namespace)::launch_dequant_weight<2>(sycl::_V1::queue&, signed char const*, flo` | int8 | 60 | 25.23 | 420 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1272 | 22.78 | 18 |
| `conv_reorder` | convolution | 262 | 13.70 | 52 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 210 | 13.46 | 64 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 12.12 | 173 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 8.23 | 20 |
| `micro_sdpa` | attention | 140 | 8.00 | 57 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 6.60 | 47 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.44 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 5.70 | 41 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.32 | 76 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.11 | 73 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.29 | 20 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::FillFunctor<float>, at::d` | memory | 5401 | 4.11 | 1 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.26 | 2 |
| `at::native::xpu::ElementwiseGlobalRangeKernel<at::native::xpu::LegacyKernelScalarFunctor<c` | norm | 74 | 2.67 | 36 |
| `at::native::xpu::GNRowwiseMomentsVectorizedFunctor<c10::BFloat16, 32, 2>` | elementwise | 74 | 2.62 | 35 |
