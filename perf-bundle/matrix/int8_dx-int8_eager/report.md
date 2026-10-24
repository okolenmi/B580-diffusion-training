# Performance report (xpu)

Window: 3 steps (3 eager, 0 replayed). All times are per step.

## Step budget

- wall 9919 ms (best 7909); eager 9919 ms, replayed - ms
- device busy 689 ms, utilisation 7%
- 38101 kernel launches, 723402 operators, 258 us host time per launch
- step peak memory 4072 MB
- phases: loss 0 ms

## Findings, largest first

1. **The step is launch bound** (9230 ms)
   - the device is busy 7% of the step (689 of 9919 ms) over 38101 launches and 723402 operators per step, 258 us of host time per launch
   - lever: graph replay (trainer.graph=true) removes the launch cost if the capture can use the same kernels
2. **Much of the module time has no cost model** (3049 ms)
   - 3049 of 11256 ms is in modules the cost model does not know
   - lever: extend `perf/cost.py` for those kinds
3. **Slowest pieces by recoverable time: conv output_blocks.in_layers (up)** (443 ms)
   - 457.3 ms per step against an ideal 13.8 ms (3% efficient, compute-bound), 9 layers; all rated pieces together leave 8026 ms
   - lever: see the module table for the individual layers
4. **Slowest pieces by recoverable time: conv output_blocks.skip_connection (up)** (305 ms)
   - 306.8 ms per step against an ideal 1.6 ms (1% efficient, compute-bound), 9 layers; all rated pieces together leave 8026 ms
   - lever: see the module table for the individual layers
5. **Slowest pieces by recoverable time: linear ff.net (up)** (257 ms)
   - 262.6 ms per step against an ideal 6.1 ms (2% efficient, compute-bound), 36 layers; all rated pieces together leave 8026 ms
   - lever: see the module table for the individual layers
6. **Slowest pieces by recoverable time: linear attn2.to_k (up)** (246 ms)
   - 247.3 ms per step against an ideal 1.0 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 8026 ms
   - lever: see the module table for the individual layers
7. **Slowest pieces by recoverable time: linear attn2.to_v (up)** (245 ms)
   - 246.4 ms per step against an ideal 1.0 ms (0% efficient, memory-bound), 36 layers; all rated pieces together leave 8026 ms
   - lever: see the module table for the individual layers
8. **Dtype and copy kernels take a large share** (151 ms)
   - 22% of device time is casts and copies
   - lever: find the modules issuing `.to(dtype)` round trips (autocast boundaries, adapter branches) and keep one dtype through them
9. **Many small memory-bound kernels** (146 ms)
   - 42% of device time (24255 calls per step) is elementwise, normalisation, copy, memory and reduction work
   - lever: fusion (fewer, larger kernels): norm + modulation, activation + multiply, cast + matmul input; half the time is the most a fusion usually recovers
10. **The int8 path's own kernels are a large share** (101 ms)
   - rotate, quantize and dequantize kernels are 15% of device time against 179 ms of matmuls
   - lever: int8_grad=int8 (no weight dequantize in backward), or a fused rotate+quantize in the kernels project; native weights if memory allows
11. **Module times do not add up to the device time** (0 ms)
   - modules account for 11256 ms of 689 ms device time per eager step
   - lever: the difference is work outside the hooked modules (loss, optimizer, data movement) or host waits; see the phase and category tables

## Device time by category

| category | calls | ms | share |
|---|---:|---:|---:|
| gemm | 8063 | 178.5 | 26% |
| copy | 5764 | 151.1 | 22% |
| elementwise | 9680 | 114.4 | 17% |
| int8 | 4487 | 100.6 | 15% |
| convolution | 533 | 73.4 | 11% |
| other | 623 | 36.3 | 5% |
| norm | 782 | 17.9 | 3% |
| attention | 140 | 8.5 | 1% |
| memory | 8011 | 6.4 | 1% |
| reduce | 18 | 1.5 | 0% |

## Measured ceilings (what 'ideal' means here)

gemm 95.1 TFLOPS, int8 184.9 TOPS, copy 366.2 GB/s, dispatch 11.0 us, queued launch 11.0 us. Backward costs are estimates (see perf/rating.py).

## Slowest piece groups by recoverable time

| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| conv | output_blocks.in_layers | up | 9 | 457.3 | 13.8 | 3% | compute | 443.5 |
| conv | output_blocks.skip_connection | up | 9 | 306.8 | 1.6 | 1% | compute | 305.2 |
| linear int8 | ff.net | up | 36 | 262.6 | 6.1 | 2% | compute | 256.6 |
| linear int8 | attn2.to_k | up | 36 | 247.3 | 1.0 | 0% | memory | 246.3 |
| linear int8 | attn2.to_v | up | 36 | 246.4 | 1.0 | 0% | memory | 245.4 |
| attention | transformer_blocks.attn1 | up | 36 | 225.0 | 6.5 | 3% | compute | 218.5 |
| linear int8 | attn1.to_out | up | 30 | 212.7 | 1.4 | 1% | memory | 211.3 |
| linear int8 | attn1.to_q | up | 30 | 212.3 | 1.4 | 1% | memory | 210.9 |
| linear int8 | attn1.to_k | up | 30 | 210.7 | 1.4 | 1% | memory | 209.3 |
| linear int8 | attn2.to_out | up | 30 | 210.7 | 1.4 | 1% | memory | 209.3 |
| attention | transformer_blocks.attn2 | up | 36 | 214.8 | 6.5 | 3% | compute | 208.2 |
| linear int8 | net.proj | up | 30 | 217.9 | 10.0 | 5% | compute | 207.9 |
| linear int8 | attn2.to_q | up | 30 | 208.1 | 1.4 | 1% | memory | 206.7 |
| linear int8 | attn1.to_v | up | 30 | 205.4 | 1.4 | 1% | memory | 204.1 |
| conv | input_blocks.op | down | 2 | 197.5 | 0.2 | 0% | compute | 197.3 |
| attention | transformer_blocks.attn1 | down | 24 | 186.1 | 4.3 | 2% | compute | 181.7 |
| attention | transformer_blocks.attn2 | down | 24 | 182.6 | 4.3 | 2% | compute | 178.3 |
| linear int8 | ff.net | down | 24 | 180.7 | 4.0 | 2% | compute | 176.7 |
| linear int8 | attn2.to_k | down | 24 | 172.9 | 0.7 | 0% | memory | 172.2 |
| linear int8 | attn2.to_v | down | 24 | 172.5 | 0.7 | 0% | memory | 171.9 |
| conv | input_blocks.in_layers | down | 6 | 155.6 | 3.0 | 2% | compute | 152.6 |
| linear int8 | attn2.to_out | down | 20 | 149.8 | 0.9 | 1% | memory | 148.8 |
| linear int8 | attn1.to_out | down | 20 | 149.3 | 0.9 | 1% | memory | 148.4 |
| linear int8 | net.proj | down | 20 | 154.6 | 6.7 | 4% | compute | 147.9 |
| linear int8 | attn1.to_q | down | 20 | 148.7 | 0.9 | 1% | memory | 147.8 |
| linear int8 | attn2.to_q | down | 20 | 146.8 | 0.9 | 1% | memory | 145.8 |
| linear int8 | attn1.to_k | down | 20 | 146.4 | 0.9 | 1% | memory | 145.5 |
| linear int8 | attn1.to_v | down | 20 | 144.1 | 0.9 | 1% | memory | 143.2 |
| feedforward | transformer_blocks.ff | up | 36 | 130.6 | 22.0 | 17% | memory | 108.6 |
| feedforward | transformer_blocks.ff | down | 24 | 106.2 | 14.7 | 14% | memory | 91.5 |

## Slowest modules (own time, nested modules excluded)

| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input_blocks.6.0.op | conv | 1 | 31.7 | 0.0 | 123.4 | 155.2 | 0.1 | 0% |
| output_blocks.3.0.in_layers.2 | conv | 1 | 41.1 | 1.3 | 27.2 | 69.6 | 2.2 | 3% |
| output_blocks.4.0.in_layers.2 | conv | 1 | 40.4 | 1.0 | 26.4 | 67.8 | 1.5 | 2% |
| output_blocks.5.0.in_layers.2 | conv | 1 | 39.9 | 1.1 | 26.3 | 67.3 | 1.1 | 2% |
| output_blocks.6.0.in_layers.2 | conv | 1 | 37.5 | 1.4 | 28.1 | 66.9 | 2.2 | 3% |
| input_blocks.7.0.in_layers.2 | conv | 1 | 37.4 | 0.8 | 26.2 | 64.4 | 0.4 | 1% |
| output_blocks.2.0.in_layers.2 | conv | 1 | 37.8 | 1.6 | 24.3 | 63.6 | 1.1 | 2% |
| output_blocks.5.2.conv | conv | 1 | 25.4 | 0.0 | 25.7 | 51.1 | 1.9 | 4% |
| output_blocks.2.0.skip_connection | conv | 1 | 28.7 | 0.6 | 20.8 | 50.0 | 0.1 | 0% |
| input_blocks.7.1.transformer_blocks.6 | block | 1 | 24.2 | 0.0 | 90.9 | 49.3 | - | - |
| input_blocks.7.0.skip_connection | conv | 1 | 29.1 | 0.8 | 18.5 | 48.4 | 0.0 | 0% |
| input_blocks.7.1.transformer_blocks.5 | block | 1 | 24.3 | 0.0 | 89.6 | 47.9 | - | - |
| input_blocks.7.1.transformer_blocks.7 | block | 1 | 23.7 | 0.0 | 87.6 | 46.3 | - | - |
| middle_block.1.transformer_blocks.8 | block | 1 | 25.2 | 0.0 | 88.2 | 45.8 | - | - |
| input_blocks.5.1.transformer_blocks.0 | block | 1 | 19.3 | 0.0 | 81.1 | 45.6 | - | - |
| input_blocks.8.1.transformer_blocks.3 | block | 1 | 24.7 | 0.0 | 86.0 | 45.2 | - | - |
| input_blocks.8.1.transformer_blocks.7 | block | 1 | 25.0 | 0.0 | 84.1 | 45.1 | - | - |
| input_blocks.7.1.transformer_blocks.0 | block | 1 | 29.0 | 0.0 | 85.2 | 44.9 | - | - |
| input_blocks.4.0.in_layers.2 | conv | 1 | 44.8 | 0.0 | 0.0 | 44.8 | 0.2 | 1% |
| middle_block.1.transformer_blocks.5 | block | 1 | 24.8 | 0.0 | 85.8 | 44.7 | - | - |
| input_blocks.8.1.transformer_blocks.5 | block | 1 | 24.3 | 0.0 | 86.9 | 44.6 | - | - |
| input_blocks.7.1.transformer_blocks.4 | block | 1 | 24.3 | 0.0 | 83.2 | 44.4 | - | - |
| middle_block.1.transformer_blocks.7 | block | 1 | 25.7 | 0.0 | 85.4 | 44.4 | - | - |
| input_blocks.8.1.transformer_blocks.4 | block | 1 | 24.2 | 0.0 | 84.7 | 44.4 | - | - |
| input_blocks.8.1.transformer_blocks.6 | block | 1 | 24.8 | 0.0 | 83.7 | 44.3 | - | - |
| output_blocks.5.0.skip_connection | conv | 1 | 20.4 | 0.6 | 23.0 | 44.0 | 0.1 | 0% |
| input_blocks.8.1.transformer_blocks.8 | block | 1 | 28.2 | 0.0 | 84.3 | 43.9 | - | - |
| middle_block.1.transformer_blocks.4 | block | 1 | 24.9 | 0.0 | 84.2 | 43.7 | - | - |
| input_blocks.7.1.transformer_blocks.2 | block | 1 | 23.5 | 0.0 | 86.2 | 43.5 | - | - |
| input_blocks.4.0.out_layers.3 | conv | 1 | 43.2 | 0.0 | 0.0 | 43.2 | 0.5 | 1% |

## Reconciliation

device 689 ms; modules' own time 11256 ms, of which 3049 ms is in modules with no cost model; wall 9919 ms.

## Slowest kernels

| kernel | category | calls | ms | us per call |
|---|---|---:|---:|---:|
| `gemm_kernel` | gemm | 7923 | 171.88 | 22 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<float>, at::det` | copy | 2789 | 68.07 | 24 |
| `at::native::xpu::UnrolledElementwiseKernel<at::native::xpu::CopyScalarFunc<c10::BFloat16>,` | copy | 2653 | 63.29 | 24 |
| `gen_conv` | convolution | 131 | 57.83 | 441 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 3045 | 51.77 | 17 |
| `(anonymous namespace)::launch_dequant_epilogue(sycl::_V1::queue&, int const*, sycl::_V1::e` | int8 | 1884 | 42.15 | 22 |
| `cute::MhaBackwardName<cute::FAKernel<cutlass::bfloat16_t, 64, 64, 32, 8, 4, 2, 2, false> >` | other | 140 | 31.31 | 224 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 846 | 23.85 | 28 |
| `(anonymous namespace)::launch_convrot_quant_warp<0>(sycl::_V1::queue&, unsigned short cons` | int8 | 1272 | 23.23 | 18 |
| `(anonymous namespace)::launch_rotate_bf16<1>(sycl::_V1::queue&, unsigned short const*, syc` | int8 | 582 | 15.84 | 27 |
| `conv_reorder` | convolution | 262 | 14.31 | 55 |
| `at::native::xpu::CatArrayBatchedCopy_contig<at::native::xpu::OpaqueType<2u>, unsigned int,` | copy | 70 | 12.52 | 179 |
| `at::native::xpu::VectorizedLayerNormKernelFunctor<c10::BFloat16, float, false>` | norm | 420 | 8.71 | 21 |
| `_quantize_row_kernel` | int8 | 642 | 8.52 | 13 |
| `micro_sdpa` | attention | 140 | 8.49 | 61 |
| `(anonymous namespace)::launch_rotate_bf16<2>(sycl::_V1::queue&, unsigned short const*, syc` | int8 | 60 | 7.50 | 125 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::AUnaryFunctor<float, floa` | elementwise | 3657 | 6.69 | 2 |
| `cute::MhaName<cutlass::fmha::kernel::XeFMHAFwdKernel<cutlass::fmha::kernel::FMHAProblemSha` | gemm | 140 | 6.65 | 48 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::GeluErfFunctor<c10::BFloa` | elementwise | 140 | 5.92 | 42 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::FillFunctor<float>, at::d` | memory | 7969 | 5.76 | 1 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.49 | 78 |
| `at::native::xpu::VectorizedElementwiseKernel<8, at::native::xpu::BinaryFunctor<c10::BFloat` | elementwise | 70 | 5.10 | 73 |
| `at::native::xpu::FusedNormKernelFunctor<c10::BFloat16, float, c10::BFloat16, unsigned int,` | norm | 210 | 4.55 | 22 |
| `at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::BinaryFunctor<float, floa` | elementwise | 1466 | 3.33 | 2 |
| `at::native::xpu::CatArrayBatchedCopy_alignedK_contig<at::native::xpu::OpaqueType<4u>, unsi` | copy | 33 | 2.99 | 91 |
