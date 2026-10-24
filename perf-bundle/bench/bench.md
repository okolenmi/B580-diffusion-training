# Micro-benchmarks (xpu, batch 2, 768 px)

Ceilings: gemm 105.5 TFLOPS, int8 199.5 TOPS, copy 398 GB/s. Efficiency is ideal time over measured time, ideal from the ceiling of whatever bounds the operation.

## gemm

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 1 cross k/v | 18432x2048 @ 2048x320 | 0.376 | 0.229 | 61% | compute |  |
| level 1 ff out | 18432x1280 @ 1280x320 | 0.244 | 0.150 | 62% | memory |  |
| level 2 projection | 4608x640 @ 640x640 | 0.055 | 0.036 | 65% | compute |  |
| level 2 cross k/v | 4608x2048 @ 2048x640 | 0.152 | 0.115 | 75% | compute |  |
| level 1 ff in | 18432x320 @ 320x2560 | 0.376 | 0.286 | 76% | compute |  |
| level 1 projection | 18432x320 @ 320x320 | 0.076 | 0.060 | 78% | memory |  |
| level 2 ff out | 4608x2560 @ 2560x640 | 0.183 | 0.143 | 78% | compute |  |
| level 3 projection | 1152x1280 @ 1280x1280 | 0.045 | 0.036 | 79% | compute |  |
| level 2 ff in | 4608x640 @ 640x5120 | 0.333 | 0.286 | 86% | compute |  |
| level 3 ff in | 1152x1280 @ 1280x10240 | 0.331 | 0.286 | 86% | compute |  |
| level 3 cross k/v | 1152x2048 @ 2048x1280 | 0.065 | 0.057 | 89% | compute |  |
| level 3 ff out | 1152x5120 @ 5120x1280 | 0.155 | 0.143 | 92% | compute |  |

## convolution

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 3 forward+dx | 2x1280x24x24 | 1.757 | 0.644 | 37% | compute |  |
| level 3 forward | 2x1280x24x24 | 0.850 | 0.322 | 38% | compute |  |
| level 1 forward+dx | 2x320x96x96 | 0.952 | 0.644 | 68% | compute |  |
| level 2 forward+dx | 2x640x48x48 | 0.886 | 0.644 | 73% | compute |  |
| level 1 forward | 2x320x96x96 | 0.434 | 0.322 | 74% | compute |  |
| level 2 forward | 2x640x48x48 | 0.419 | 0.322 | 77% | compute |  |

## norm

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 3 group norm forward+dx | 2x1280x24x24 | 0.387 | 0.037 | 10% | memory |  |
| level 3 layer norm forward+dx | 1152x1280 | 0.295 | 0.037 | 13% | memory |  |
| level 3 group norm forward | 2x1280x24x24 | 0.083 | 0.015 | 18% | memory |  |
| level 2 group norm forward+dx | 2x640x48x48 | 0.396 | 0.074 | 19% | memory |  |
| level 1 layer norm forward+dx | 18432x320 | 0.633 | 0.148 | 23% | memory |  |
| level 2 layer norm forward+dx | 4608x640 | 0.297 | 0.074 | 25% | memory |  |
| level 1 group norm forward+dx | 2x320x96x96 | 0.432 | 0.148 | 34% | memory |  |
| level 2 group norm forward | 2x640x48x48 | 0.085 | 0.030 | 35% | memory |  |
| level 1 group norm forward | 2x320x96x96 | 0.132 | 0.059 | 45% | memory |  |

## elementwise

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| chain of 8 (mul, add, silu) | 2949120 elements | 0.741 | 0.059 | 8% | memory | ideal is one fused pass (reads x, y, z, writes once) |
| cast to float32 | 2949120 elements | 0.043 | 0.044 | 103% | memory |  |
| silu | 2949120 elements | 0.022 | 0.030 | 136% | memory |  |
| add | 2949120 elements | 0.017 | 0.044 | 260% | memory |  |
| mul | 2949120 elements | 0.017 | 0.044 | 265% | memory |  |

## attention

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 3 self math forward | 2x20x576x64 (self, keys 576) | 1.068 | 0.032 | 3% | compute |  |
| level 3 self efficient forward | 2x20x576x64 (self, keys 576) | 1.067 | 0.032 | 3% | compute |  |
| level 3 self chunked_bf16 forward+backward | 2x20x576x64 (self, keys 576) | 3.670 | 0.113 | 3% | compute | captures; replay 3.64 ms |
| level 2 self chunked_bf16 forward+backward | 2x10x2304x64 (self, keys 2304) | 27.492 | 0.902 | 3% | compute | captures; replay 27.55 ms |
| level 3 self chunked_bf16 forward | 2x20x576x64 (self, keys 576) | 0.951 | 0.032 | 3% | compute |  |
| level 2 self chunked_bf16 forward | 2x10x2304x64 (self, keys 2304) | 7.440 | 0.258 | 3% | compute |  |
| level 2 self efficient forward | 2x10x2304x64 (self, keys 2304) | 7.008 | 0.258 | 4% | compute |  |
| level 2 self math forward | 2x10x2304x64 (self, keys 2304) | 7.004 | 0.258 | 4% | compute |  |
| level 3 cross chunked_bf16 forward+backward | 2x20x576x64 (cross, keys 77) | 1.273 | 0.050 | 4% | memory | captures; replay 0.72 ms |
| level 3 self math forward+backward | 2x20x576x64 (self, keys 576) | 2.435 | 0.113 | 5% | compute | captures; replay 2.42 ms |
| level 3 self efficient forward+backward | 2x20x576x64 (self, keys 576) | 2.432 | 0.113 | 5% | compute | captures; replay 2.47 ms |
| level 3 cross math forward+backward | 2x20x576x64 (cross, keys 77) | 1.005 | 0.050 | 5% | memory | captures; replay 0.53 ms |
| level 3 cross efficient forward+backward | 2x20x576x64 (cross, keys 77) | 0.988 | 0.050 | 5% | memory | captures; replay 0.53 ms |
| level 3 cross math forward | 2x20x576x64 (cross, keys 77) | 0.297 | 0.017 | 6% | memory |  |
| level 3 cross efficient forward | 2x20x576x64 (cross, keys 77) | 0.289 | 0.017 | 6% | memory |  |
| level 2 self math forward+backward | 2x10x2304x64 (self, keys 2304) | 15.478 | 0.902 | 6% | compute | captures; replay 15.67 ms |
| level 2 self efficient forward+backward | 2x10x2304x64 (self, keys 2304) | 15.475 | 0.902 | 6% | compute | captures; replay 15.62 ms |
| level 2 cross math forward | 2x10x2304x64 (cross, keys 77) | 0.519 | 0.031 | 6% | memory |  |
| level 2 cross efficient forward | 2x10x2304x64 (cross, keys 77) | 0.517 | 0.031 | 6% | memory |  |
| level 2 cross chunked_bf16 forward+backward | 2x10x2304x64 (cross, keys 77) | 1.538 | 0.092 | 6% | memory | captures; replay 1.52 ms |
| level 3 cross chunked_bf16 forward | 2x20x576x64 (cross, keys 77) | 0.270 | 0.017 | 6% | memory |  |
| level 3 cross math_bf16 forward+backward | 2x20x576x64 (cross, keys 77) | 0.800 | 0.050 | 6% | memory | captures; replay 0.29 ms |
| level 2 cross efficient forward+backward | 2x10x2304x64 (cross, keys 77) | 1.180 | 0.092 | 8% | memory | captures; replay 1.17 ms |
| level 2 cross math forward+backward | 2x10x2304x64 (cross, keys 77) | 1.179 | 0.092 | 8% | memory | captures; replay 1.17 ms |
| level 3 cross math_bf16 forward | 2x20x576x64 (cross, keys 77) | 0.202 | 0.017 | 8% | memory |  |
| level 3 self math_bf16 forward | 2x20x576x64 (self, keys 576) | 0.371 | 0.032 | 9% | compute |  |
| level 2 cross chunked_bf16 forward | 2x10x2304x64 (cross, keys 77) | 0.352 | 0.031 | 9% | memory |  |
| level 2 cross math_bf16 forward+backward | 2x10x2304x64 (cross, keys 77) | 0.877 | 0.092 | 10% | memory | captures; replay 0.73 ms |
| level 3 self math_bf16 forward+backward | 2x20x576x64 (self, keys 576) | 1.009 | 0.113 | 11% | compute | captures; replay 1.01 ms |
| level 2 self math_bf16 forward | 2x10x2304x64 (self, keys 2304) | 2.305 | 0.258 | 11% | compute |  |
| level 2 self math_bf16 forward+backward | 2x10x2304x64 (self, keys 2304) | 6.319 | 0.902 | 14% | compute | captures; replay 6.34 ms |
| level 2 cross math_bf16 forward | 2x10x2304x64 (cross, keys 77) | 0.202 | 0.031 | 15% | memory |  |
| level 3 cross default forward+backward | 2x20x576x64 (cross, keys 77) | 0.262 | 0.050 | 19% | memory | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 3 self default forward+backward | 2x20x576x64 (self, keys 576) | 0.518 | 0.113 | 22% | compute | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 3 self flash forward+backward | 2x20x576x64 (self, keys 576) | 0.513 | 0.113 | 22% | compute | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 3 cross flash forward+backward | 2x20x576x64 (cross, keys 77) | 0.178 | 0.050 | 28% | memory | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 2 cross flash forward+backward | 2x10x2304x64 (cross, keys 77) | 0.322 | 0.092 | 29% | memory | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 2 cross default forward+backward | 2x10x2304x64 (cross, keys 77) | 0.317 | 0.092 | 29% | memory | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 2 self flash forward+backward | 2x10x2304x64 (self, keys 2304) | 2.865 | 0.902 | 31% | compute | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 2 self default forward+backward | 2x10x2304x64 (self, keys 2304) | 2.855 | 0.902 | 32% | compute | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 3 self default forward | 2x20x576x64 (self, keys 576) | 0.058 | 0.032 | 56% | compute |  |
| level 3 self flash forward | 2x20x576x64 (self, keys 576) | 0.058 | 0.032 | 56% | compute |  |
| level 3 cross flash forward | 2x20x576x64 (cross, keys 77) | 0.026 | 0.017 | 65% | memory |  |
| level 3 cross default forward | 2x20x576x64 (cross, keys 77) | 0.026 | 0.017 | 65% | memory |  |
| level 2 cross flash forward | 2x10x2304x64 (cross, keys 77) | 0.040 | 0.031 | 76% | memory |  |
| level 2 cross default forward | 2x10x2304x64 (cross, keys 77) | 0.040 | 0.031 | 76% | memory |  |
| level 2 self default forward | 2x10x2304x64 (self, keys 2304) | 0.334 | 0.258 | 77% | compute |  |
| level 2 self flash forward | 2x10x2304x64 (self, keys 2304) | 0.334 | 0.258 | 77% | compute |  |

## linear layer

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 3 attention projection int8 (int8 dx) forward+dx | 1152x1280 @ 1280x1280 | 0.884 | 0.038 | 4% | memory | 0.32x of native speed |
| level 3 attention projection int8 (dense dx) forward+dx | 1152x1280 @ 1280x1280 | 0.571 | 0.038 | 7% | memory | 0.50x of native speed |
| level 3 cross k/v int8 (int8 dx) forward+dx | 1152x2048 @ 2048x1280 | 0.884 | 0.061 | 7% | compute | 0.27x of native speed |
| level 2 feed-forward out int8 (int8 dx) forward+dx | 4608x2560 @ 2560x640 | 1.703 | 0.156 | 9% | memory | 0.31x of native speed |
| level 3 cross k/v int8 (dense dx) forward+dx | 1152x2048 @ 2048x1280 | 0.584 | 0.061 | 10% | compute | 0.41x of native speed |
| level 2 cross k/v int8 (int8 dx) forward+dx | 4608x2048 @ 2048x640 | 1.235 | 0.131 | 11% | memory | 0.35x of native speed |
| level 3 feed-forward out weight dequantize | 1280x5120 | 0.383 | 0.049 | 13% | memory |  |
| level 3 feed-forward out int8 (int8 dx) forward+dx | 1152x5120 @ 5120x1280 | 0.897 | 0.151 | 17% | compute | 0.42x of native speed |
| level 3 cross k/v weight dequantize | 1280x2048 | 0.107 | 0.020 | 18% | memory |  |
| level 2 feed-forward out int8 (dense dx) forward+dx | 4608x2560 @ 2560x640 | 0.836 | 0.156 | 19% | memory | 0.64x of native speed |
| level 3 feed-forward out int8 (dense dx) forward+dx | 1152x5120 @ 5120x1280 | 0.775 | 0.151 | 20% | compute | 0.48x of native speed |
| level 2 cross k/v int8 (dense dx) forward+dx | 4608x2048 @ 2048x640 | 0.583 | 0.131 | 22% | memory | 0.74x of native speed |
| level 3 attention projection native forward+dx | 1152x1280 @ 1280x1280 | 0.284 | 0.072 | 25% | compute |  |
| level 2 cross k/v weight dequantize | 640x2048 | 0.037 | 0.010 | 27% | memory |  |
| level 3 feed-forward in int8 (dense dx) forward+dx | 1152x1280 @ 1280x10240 | 1.058 | 0.303 | 29% | compute | 0.59x of native speed |
| level 3 feed-forward in int8 (int8 dx) forward+dx | 1152x1280 @ 1280x10240 | 0.968 | 0.303 | 31% | compute | 0.64x of native speed |
| level 2 attention projection native forward+dx | 4608x640 @ 640x640 | 0.221 | 0.072 | 32% | compute |  |
| level 2 feed-forward out weight dequantize | 640x2560 | 0.036 | 0.012 | 34% | memory |  |
| level 3 attention projection weight dequantize | 1280x1280 | 0.035 | 0.012 | 35% | memory |  |
| level 3 feed-forward in weight dequantize | 10240x1280 | 0.250 | 0.099 | 39% | memory |  |
| level 3 cross k/v native forward+dx | 1152x2048 @ 2048x1280 | 0.237 | 0.115 | 48% | compute |  |
| level 2 cross k/v native forward+dx | 4608x2048 @ 2048x640 | 0.433 | 0.229 | 53% | compute |  |
| level 2 feed-forward out native forward+dx | 4608x2560 @ 2560x640 | 0.533 | 0.286 | 54% | compute |  |
| level 2 feed-forward in native forward+dx | 4608x640 @ 640x5120 | 0.880 | 0.573 | 65% | compute |  |
| level 3 feed-forward out native forward+dx | 1152x5120 @ 5120x1280 | 0.374 | 0.286 | 77% | compute |  |
| level 3 feed-forward in native forward+dx | 1152x1280 @ 1280x10240 | 0.623 | 0.573 | 92% | compute |  |
| level 2 attention projection int8 | 4608x640 @ 640x640 | - | - | - | - | stays native: 640 input features is not a multiple of 256 |
| level 2 feed-forward in int8 | 4608x640 @ 640x5120 | - | - | - | - | stays native: 640 input features is not a multiple of 256 |

## launch

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| 200 tiny kernels, eager | 1024 elements | 2.589 | - | - | - | 12.9 us per kernel |
| 200 tiny kernels, graph replay | 1024 elements | 0.327 | - | - | - | 1.6 us per kernel, 7.9x faster than eager |

