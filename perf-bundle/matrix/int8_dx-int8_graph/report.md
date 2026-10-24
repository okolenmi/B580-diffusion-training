# Performance report (xpu)

Window: 3 steps (2 eager, 1 replayed). All times are per step.

## Step budget

- wall 9713 ms (best 8134); eager 8206 ms, replayed 12725 ms
- device busy 682 ms, utilisation 7%
- 38101 kernel launches, 723516 operators, 250 us host time per launch
- step peak memory 4073 MB
- phases: loss 0 ms

## Findings, largest first

1. **The step is launch bound** (9031 ms)
   - the device is busy 7% of the step (682 of 9713 ms) over 38101 launches and 723516 operators per step, 250 us of host time per launch
   - lever: graph replay is already running for some steps; compare replayed and eager step times above
2. **Slowest pieces by recoverable time: conv output_blocks.in_layers (up)** (671 ms)
   - 691.6 ms per step against an ideal 20.7 ms (3% efficient, compute-bound), 9 layers; all rated pieces together leave 11781 ms
   - lever: see the module table for the individual layers
3. **Slowest pieces by recoverable time: conv output_blocks.skip_connection (up)** (456 ms)
   - 458.3 ms per step against an ideal 2.5 ms (1% efficient, compute-bound), 9 layers; all rated pieces together leave 11781 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: linear ff.net (up)** (374 ms)
   - 382.7 ms per step against an ideal 9.1 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 11781 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: linear attn2.to_k (up)** (361 ms)
   - 362.8 ms per step against an ideal 1.5 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 11781 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: linear attn2.to_v (up)** (358 ms)
   - 359.9 ms per step against an ideal 1.5 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 11781 ms
   - lever: see the module table for the individual layers
7. **Dtype and copy kernels take a large share** (150 ms)
   - 22% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
8. **Many small memory-bound kernels** (144 ms)
   - 42% of device time (24255 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
9. **The int8 path's own kernels are a large share** (101 ms)
   - rotate, quantize and dequantize kernels are 15% of device time against 177 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 177.1 | 26% |
| copy | 5764 | 149.8 | 22% |
| elementwise | 9680 | 113.6 | 17% |
| int8 | 4487 | 101.1 | 15% |
| convolution | 533 | 71.4 | 10% |
| other | 623 | 35.3 | 5% |
| norm | 782 | 17.6 | 3% |
| attention | 140 | 8.1 | 1% |
| memory | 8011 | 6.5 | 1% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 95.1 TFLOPS, int8 184.9 TOPS, copy 366.2 GB/s, dispatch 11.0 us, queued launch 11.0 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| conv | output_blocks.in_layers | up | 9 | 691.6 | 20.7 | 3% | compute | 670.8 |
| conv | output_blocks.skip_connection | up | 9 | 458.3 | 2.5 | 1% | compute | 455.9 |
| linear int8 | ff.net | up | 36 | 382.7 | 9.1 | 2% | compute | 373.6 |
| linear int8 | attn2.to_k | up | 36 | 362.8 | 1.5 | 0% | memory | 361.3 |
| linear int8 | attn2.to_v | up | 36 | 359.9 | 1.5 | 0% | memory | 358.4 |
| attention | transformer_blocks.attn1 | up | 36 | 328.5 | 9.8 | 3% | compute | 318.7 |
| linear int8 | attn1.to_out | up | 30 | 318.0 | 2.1 | 1% | memory | 315.9 |
| linear int8 | attn2.to_out | up | 30 | 314.2 | 2.1 | 1% | memory | 312.1 |
| linear int8 | net.proj | up | 30 | 324.9 | 15.0 | 5% | compute | 310.0 |
| linear int8 | attn1.to_q | up | 30 | 311.5 | 2.1 | 1% | memory | 309.4 |
| linear int8 | attn1.to_k | up | 30 | 310.8 | 2.1 | 1% | memory | 308.7 |
| linear int8 | attn2.to_q | up | 30 | 309.6 | 2.1 | 1% | memory | 307.6 |
| linear int8 | attn1.to_v | up | 30 | 304.0 | 2.1 | 1% | memory | 301.9 |
| attention | transformer_blocks.attn2 | up | 36 | 311.6 | 9.8 | 3% | compute | 301.9 |
| conv | input_blocks.op | down | 2 | 285.3 | 0.4 | 0% | compute | 285.0 |
| attention | transformer_blocks.attn1 | down | 24 | 283.9 | 6.5 | 2% | compute | 277.4 |
| attention | transformer_blocks.attn2 | down | 24 | 277.6 | 6.5 | 2% | compute | 271.1 |
| linear int8 | ff.net | down | 24 | 268.5 | 6.1 | 2% | compute | 262.5 |
| linear int8 | attn2.to_k | down | 24 | 256.2 | 1.0 | 0% | memory | 255.2 |
| linear int8 | attn2.to_v | down | 24 | 255.2 | 1.0 | 0% | memory | 254.2 |
| linear int8 | attn1.to_out | down | 20 | 223.6 | 1.4 | 1% | memory | 222.3 |
| linear int8 | attn2.to_out | down | 20 | 221.4 | 1.4 | 1% | memory | 220.0 |
| conv | input_blocks.in_layers | down | 6 | 221.2 | 4.5 | 2% | compute | 216.7 |
| linear int8 | net.proj | down | 20 | 226.4 | 10.0 | 4% | compute | 216.4 |
| linear int8 | attn1.to_q | down | 20 | 217.0 | 1.4 | 1% | memory | 215.6 |
| linear int8 | attn1.to_k | down | 20 | 216.1 | 1.4 | 1% | memory | 214.7 |
| linear int8 | attn2.to_q | down | 20 | 214.9 | 1.4 | 1% | memory | 213.5 |
| linear int8 | attn1.to_v | down | 20 | 214.0 | 1.4 | 1% | memory | 212.6 |
| feedforward | transformer_blocks.ff | up | 36 | 187.3 | 33.1 | 18% | memory | 154.3 |
| feedforward | transformer_blocks.ff | down | 24 | 161.4 | 22.0 | 14% | memory | 139.4 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.6.0.op | conv | 2 | 46.4 | 0.0 | 176.0 | 222.3 | 0.2 | 0% |
| output_blocks.3.0.in_layers.2 | conv | 2 | 62.4 | 2.4 | 41.3 | 106.1 | 3.3 | 3% |
| output_blocks.4.0.in_layers.2 | conv | 2 | 60.7 | 1.5 | 40.6 | 102.8 | 2.2 | 2% |
| output_blocks.6.0.in_layers.2 | conv | 2 | 57.2 | 2.0 | 40.8 | 100.0 | 3.3 | 3% |
| output_blocks.5.0.in_layers.2 | conv | 2 | 60.6 | 1.2 | 38.1 | 99.9 | 1.6 | 2% |
| output_blocks.2.0.in_layers.2 | conv | 2 | 58.8 | 2.2 | 36.2 | 97.1 | 1.6 | 2% |
| input_blocks.7.0.in_layers.2 | conv | 2 | 56.0 | 1.2 | 37.9 | 95.1 | 0.5 | 1% |
| output_blocks.5.2.conv | conv | 2 | 38.2 | 0.0 | 38.0 | 76.2 | 2.9 | 4% |
| output_blocks.2.0.skip_connection | conv | 2 | 42.9 | 1.1 | 30.5 | 74.5 | 0.2 | 0% |
| input_blocks.5.1.transformer_blocks.0 | block | 2 | 26.5 | 0.0 | 128.2 | 72.2 | - | - |
| input_blocks.7.0.skip_connection | conv | 2 | 41.9 | 1.5 | 27.4 | 70.8 | 0.1 | 0% |
| input_blocks.7.1.transformer_blocks.1 | block | 2 | 33.6 | 0.0 | 131.8 | 69.5 | - | - |
| input_blocks.7.1.transformer_blocks.3 | block | 2 | 32.7 | 0.0 | 134.0 | 69.0 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 2 | 33.2 | 0.0 | 132.0 | 68.6 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 2 | 41.1 | 0.0 | 132.4 | 68.6 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 2 | 34.1 | 0.0 | 132.4 | 68.3 | - | - |
| input_blocks.7.1.transformer_blocks.6 | block | 2 | 33.6 | 0.0 | 128.9 | 67.0 | - | - |
| out.2 | conv | 2 | 18.2 | 0.0 | 48.5 | 66.7 | 0.1 | 0% |
| input_blocks.7.1.transformer_blocks.8 | block | 2 | 34.0 | 0.0 | 128.4 | 66.5 | - | - |
| input_blocks.7.1.transformer_blocks.5 | block | 2 | 34.6 | 0.0 | 129.2 | 66.1 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 2 | 34.1 | 0.0 | 127.9 | 66.1 | - | - |
| output_blocks.5.0.skip_connection | conv | 2 | 30.8 | 0.8 | 34.0 | 65.7 | 0.2 | 0% |
| input_blocks.7.1.transformer_blocks.9 | block | 2 | 33.9 | 0.0 | 126.0 | 65.2 | - | - |
| input_blocks.8.1.transformer_blocks.2 | block | 2 | 33.7 | 0.0 | 125.0 | 65.1 | - | - |
| input_blocks.8.1.transformer_blocks.0 | block | 2 | 33.8 | 0.0 | 126.0 | 64.8 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 2 | 34.4 | 0.0 | 124.6 | 64.7 | - | - |
| output_blocks.0.0.in_layers.2 | conv | 2 | 59.1 | 2.9 | 2.7 | 64.7 | 2.2 | 3% |
| input_blocks.8.1.transformer_blocks.1 | block | 2 | 35.0 | 0.0 | 125.2 | 64.6 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 2 | 34.0 | 0.0 | 124.3 | 64.2 | - | - |
| input_blocks.8.1.transformer_blocks.8 | block | 2 | 33.8 | 0.0 | 123.5 | 63.7 | - | - |

## Reconciliation

device 682 ms; modules - ms (unrated - ms). Module times need eager steps: a window with replayed steps has no module table.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 170.35 | 22 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2789 | 67.01 | 24 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2653 | 63.41 | 24 |
| `gen_conv` | convolution | 131 | 56.19 | 429 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 51.03 | 17 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1884 | 43.52 | 23 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 30.65 | 219 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 846 | 24.64 | 29 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1272 | 23.11 | 18 |
| `(anonymous namespace)::launch_rotate_bf16<1>(sycl::_V1::queue&, unsigned short const*, syc` | int8 | 582 | 15.52 | 27 |
| `conv_reorder` | convolution | 262 | 13.90 | 53 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 12.32 | 176 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 8.38 | 20 |
| `_quantize_row_kernel` | int8 | 642 | 8.18 | 13 |
| `micro_sdpa` | attention | 140 | 8.10 | 58 |
| `(anonymous namespace)::launch_rotate_bf16<2>(sycl::_V1::queue&, unsigned short const*, syc` | int8 | 60 | 7.47 | 125 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 6.74 | 48 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.48 | 2 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::FillFunctor<float>, at::d` | memory | 7969 | 5.81 | 1 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 5.72 | 41 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.53 | 79 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.02 | 72 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.38 | 21 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.24 | 2 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 3.08 | 93 |
