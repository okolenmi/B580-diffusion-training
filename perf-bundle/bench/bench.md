# Micro-benchmarks (xpu, batch 2, 768 px)

Ceilings: gemm 105.5 TFLOPS, int8 199.5 TOPS, copy 401 GB/s. Efficiency is ideal time over measured time, ideal from the ceiling of whatever bounds the operation.

## gemm

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 1 cross k/v | 18432x2048 @ 2048x320 | 0.375 | 0.229 | 61% | compute |  |
| level 1 ff out | 18432x1280 @ 1280x320 | 0.243 | 0.149 | 61% | memory |  |
| level 2 projection | 4608x640 @ 640x640 | 0.051 | 0.036 | 71% | compute |  |
| level 3 projection | 1152x1280 @ 1280x1280 | 0.048 | 0.036 | 74% | compute |  |
| level 2 cross k/v | 4608x2048 @ 2048x640 | 0.152 | 0.114 | 75% | compute |  |
| level 1 ff in | 18432x320 @ 320x2560 | 0.375 | 0.286 | 76% | compute |  |
| level 2 ff out | 4608x2560 @ 2560x640 | 0.182 | 0.143 | 79% | compute |  |
| level 1 projection | 18432x320 @ 320x320 | 0.072 | 0.059 | 82% | memory |  |
| level 2 ff in | 4608x640 @ 640x5120 | 0.333 | 0.286 | 86% | compute |  |
| level 3 ff in | 1152x1280 @ 1280x10240 | 0.326 | 0.286 | 88% | compute |  |
| level 3 cross k/v | 1152x2048 @ 2048x1280 | 0.065 | 0.057 | 88% | compute |  |
| level 3 ff out | 1152x5120 @ 5120x1280 | 0.154 | 0.143 | 93% | compute |  |

## convolution

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 3 forward | 2x1280x24x24 | 1.057 | 0.322 | 30% | compute |  |
| level 3 forward+dx | 2x1280x24x24 | 1.885 | 0.644 | 34% | compute |  |
| level 1 forward+dx | 2x320x96x96 | 1.159 | 0.644 | 56% | compute |  |
| level 2 forward | 2x640x48x48 | 0.537 | 0.322 | 60% | compute |  |
| level 2 forward+dx | 2x640x48x48 | 1.035 | 0.644 | 62% | compute |  |
| level 1 forward | 2x320x96x96 | 0.440 | 0.322 | 73% | compute |  |

## norm

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 3 group norm forward+dx | 2x1280x24x24 | 0.355 | 0.037 | 10% | memory |  |
| level 3 group norm forward | 2x1280x24x24 | 0.087 | 0.015 | 17% | memory |  |
| level 3 layer norm forward+dx | 1152x1280 | 0.191 | 0.037 | 19% | memory |  |
| level 2 group norm forward+dx | 2x640x48x48 | 0.376 | 0.074 | 20% | memory |  |
| level 1 layer norm forward+dx | 18432x320 | 0.597 | 0.147 | 25% | memory |  |
| level 2 layer norm forward+dx | 4608x640 | 0.264 | 0.074 | 28% | memory |  |
| level 1 group norm forward+dx | 2x320x96x96 | 0.433 | 0.147 | 34% | memory |  |
| level 2 group norm forward | 2x640x48x48 | 0.084 | 0.029 | 35% | memory |  |
| level 1 group norm forward | 2x320x96x96 | 0.129 | 0.059 | 46% | memory |  |

## elementwise

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| chain of 8 (mul, add, silu) | 2949120 elements | 0.724 | 0.059 | 8% | memory | ideal is one fused pass (reads x, y, z, writes once) |
| cast to float32 | 2949120 elements | 0.043 | 0.044 | 102% | memory |  |
| silu | 2949120 elements | 0.022 | 0.029 | 135% | memory |  |
| add | 2949120 elements | 0.014 | 0.044 | 305% | memory |  |
| mul | 2949120 elements | 0.014 | 0.044 | 306% | memory |  |

## attention

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 3 self math forward | 2x20x576x64 (self, keys 576) | 1.069 | 0.032 | 3% | compute |  |
| level 3 self efficient forward | 2x20x576x64 (self, keys 576) | 1.068 | 0.032 | 3% | compute |  |
| level 3 self chunked_bf16 forward+backward | 2x20x576x64 (self, keys 576) | 3.666 | 0.113 | 3% | compute | captures; replay 3.65 ms |
| level 2 self chunked_bf16 forward+backward | 2x10x2304x64 (self, keys 2304) | 27.452 | 0.902 | 3% | compute | captures; replay 27.51 ms |
| level 3 self chunked_bf16 forward | 2x20x576x64 (self, keys 576) | 0.956 | 0.032 | 3% | compute |  |
| level 2 self chunked_bf16 forward | 2x10x2304x64 (self, keys 2304) | 7.399 | 0.258 | 3% | compute |  |
| level 2 self efficient forward | 2x10x2304x64 (self, keys 2304) | 7.026 | 0.258 | 4% | compute |  |
| level 2 self math forward | 2x10x2304x64 (self, keys 2304) | 7.009 | 0.258 | 4% | compute |  |
| level 3 cross chunked_bf16 forward+backward | 2x20x576x64 (cross, keys 77) | 1.294 | 0.050 | 4% | memory | captures; replay 0.71 ms |
| level 3 self efficient forward+backward | 2x20x576x64 (self, keys 576) | 2.431 | 0.113 | 5% | compute | captures; replay 2.42 ms |
| level 3 self math forward+backward | 2x20x576x64 (self, keys 576) | 2.430 | 0.113 | 5% | compute | captures; replay 2.42 ms |
| level 3 cross efficient forward+backward | 2x20x576x64 (cross, keys 77) | 0.992 | 0.050 | 5% | memory | captures; replay 0.53 ms |
| level 3 cross math forward+backward | 2x20x576x64 (cross, keys 77) | 0.989 | 0.050 | 5% | memory | captures; replay 0.53 ms |
| level 3 cross efficient forward | 2x20x576x64 (cross, keys 77) | 0.289 | 0.017 | 6% | memory |  |
| level 3 cross math forward | 2x20x576x64 (cross, keys 77) | 0.288 | 0.017 | 6% | memory |  |
| level 2 cross math forward | 2x10x2304x64 (cross, keys 77) | 0.525 | 0.030 | 6% | memory |  |
| level 2 self efficient forward+backward | 2x10x2304x64 (self, keys 2304) | 15.517 | 0.902 | 6% | compute | captures; replay 15.64 ms |
| level 2 self math forward+backward | 2x10x2304x64 (self, keys 2304) | 15.491 | 0.902 | 6% | compute | captures; replay 15.62 ms |
| level 2 cross chunked_bf16 forward+backward | 2x10x2304x64 (cross, keys 77) | 1.552 | 0.091 | 6% | memory | captures; replay 1.52 ms |
| level 2 cross efficient forward | 2x10x2304x64 (cross, keys 77) | 0.516 | 0.030 | 6% | memory |  |
| level 3 cross math_bf16 forward+backward | 2x20x576x64 (cross, keys 77) | 0.819 | 0.050 | 6% | memory | captures; replay 0.29 ms |
| level 3 cross chunked_bf16 forward | 2x20x576x64 (cross, keys 77) | 0.268 | 0.017 | 6% | memory |  |
| level 2 cross math forward+backward | 2x10x2304x64 (cross, keys 77) | 1.224 | 0.091 | 7% | memory | captures; replay 1.17 ms |
| level 2 cross efficient forward+backward | 2x10x2304x64 (cross, keys 77) | 1.180 | 0.091 | 8% | memory | captures; replay 1.17 ms |
| level 3 cross math_bf16 forward | 2x20x576x64 (cross, keys 77) | 0.206 | 0.017 | 8% | memory |  |
| level 2 cross chunked_bf16 forward | 2x10x2304x64 (cross, keys 77) | 0.348 | 0.030 | 9% | memory |  |
| level 3 self math_bf16 forward | 2x20x576x64 (self, keys 576) | 0.368 | 0.032 | 9% | compute |  |
| level 2 cross math_bf16 forward+backward | 2x10x2304x64 (cross, keys 77) | 0.873 | 0.091 | 10% | memory | captures; replay 0.73 ms |
| level 2 self math_bf16 forward | 2x10x2304x64 (self, keys 2304) | 2.311 | 0.258 | 11% | compute |  |
| level 3 self math_bf16 forward+backward | 2x20x576x64 (self, keys 576) | 1.003 | 0.113 | 11% | compute | captures; replay 1.01 ms |
| level 2 self math_bf16 forward+backward | 2x10x2304x64 (self, keys 2304) | 6.342 | 0.902 | 14% | compute | captures; replay 6.32 ms |
| level 2 cross math_bf16 forward | 2x10x2304x64 (cross, keys 77) | 0.203 | 0.030 | 15% | memory |  |
| level 3 cross flash forward+backward | 2x20x576x64 (cross, keys 77) | 0.256 | 0.050 | 20% | memory | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 3 self default forward+backward | 2x20x576x64 (self, keys 576) | 0.515 | 0.113 | 22% | compute | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 3 cross default forward+backward | 2x20x576x64 (cross, keys 77) | 0.227 | 0.050 | 22% | memory | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 3 self flash forward+backward | 2x20x576x64 (self, keys 576) | 0.507 | 0.113 | 22% | compute | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 2 cross flash forward+backward | 2x10x2304x64 (cross, keys 77) | 0.321 | 0.091 | 28% | memory | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 2 cross default forward+backward | 2x10x2304x64 (cross, keys 77) | 0.320 | 0.091 | 28% | memory | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 2 self flash forward+backward | 2x10x2304x64 (self, keys 2304) | 2.872 | 0.902 | 31% | compute | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 2 self default forward+backward | 2x10x2304x64 (self, keys 2304) | 2.841 | 0.902 | 32% | compute | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 3 self flash forward | 2x20x576x64 (self, keys 576) | 0.058 | 0.032 | 56% | compute |  |
| level 3 self default forward | 2x20x576x64 (self, keys 576) | 0.057 | 0.032 | 56% | compute |  |
| level 3 cross flash forward | 2x20x576x64 (cross, keys 77) | 0.026 | 0.017 | 64% | memory |  |
| level 3 cross default forward | 2x20x576x64 (cross, keys 77) | 0.026 | 0.017 | 65% | memory |  |
| level 2 cross default forward | 2x10x2304x64 (cross, keys 77) | 0.040 | 0.030 | 76% | memory |  |
| level 2 cross flash forward | 2x10x2304x64 (cross, keys 77) | 0.040 | 0.030 | 76% | memory |  |
| level 2 self flash forward | 2x10x2304x64 (self, keys 2304) | 0.338 | 0.258 | 76% | compute |  |
| level 2 self default forward | 2x10x2304x64 (self, keys 2304) | 0.337 | 0.258 | 76% | compute |  |

## linear layer

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 3 attention projection int8 (int8 dx) forward+dx | 1152x1280 @ 1280x1280 | 0.873 | 0.038 | 4% | compute | 0.22x of native speed |
| level 3 attention projection int8 (dense dx) forward+dx | 1152x1280 @ 1280x1280 | 0.557 | 0.038 | 7% | compute | 0.35x of native speed |
| level 3 cross k/v int8 (int8 dx) forward+dx | 1152x2048 @ 2048x1280 | 0.870 | 0.061 | 7% | compute | 0.27x of native speed |
| level 2 feed-forward out int8 (int8 dx) forward+dx | 4608x2560 @ 2560x640 | 1.608 | 0.155 | 10% | memory | 0.33x of native speed |
| level 3 cross k/v int8 (dense dx) forward+dx | 1152x2048 @ 2048x1280 | 0.581 | 0.061 | 10% | compute | 0.41x of native speed |
| level 2 cross k/v int8 (int8 dx) forward+dx | 4608x2048 @ 2048x640 | 1.055 | 0.130 | 12% | memory | 0.41x of native speed |
| level 3 feed-forward out weight dequantize | 1280x5120 | 0.390 | 0.049 | 13% | memory |  |
| level 3 feed-forward out int8 (int8 dx) forward+dx | 1152x5120 @ 5120x1280 | 0.897 | 0.151 | 17% | compute | 0.42x of native speed |
| level 2 feed-forward out int8 (dense dx) forward+dx | 4608x2560 @ 2560x640 | 0.817 | 0.155 | 19% | memory | 0.65x of native speed |
| level 3 feed-forward out int8 (dense dx) forward+dx | 1152x5120 @ 5120x1280 | 0.782 | 0.151 | 19% | compute | 0.48x of native speed |
| level 2 cross k/v int8 (dense dx) forward+dx | 4608x2048 @ 2048x640 | 0.577 | 0.130 | 23% | memory | 0.75x of native speed |
| level 2 cross k/v weight dequantize | 640x2048 | 0.037 | 0.010 | 27% | memory |  |
| level 2 attention projection native forward+dx | 4608x640 @ 640x640 | 0.260 | 0.072 | 28% | compute |  |
| level 3 feed-forward in int8 (dense dx) forward+dx | 1152x1280 @ 1280x10240 | 1.043 | 0.303 | 29% | compute | 0.60x of native speed |
| level 3 feed-forward in int8 (int8 dx) forward+dx | 1152x1280 @ 1280x10240 | 0.967 | 0.303 | 31% | compute | 0.64x of native speed |
| level 3 attention projection weight dequantize | 1280x1280 | 0.037 | 0.012 | 33% | memory |  |
| level 2 feed-forward out weight dequantize | 640x2560 | 0.036 | 0.012 | 34% | memory |  |
| level 3 attention projection native forward+dx | 1152x1280 @ 1280x1280 | 0.196 | 0.072 | 36% | compute |  |
| level 3 cross k/v weight dequantize | 1280x2048 | 0.054 | 0.020 | 37% | memory |  |
| level 3 feed-forward in weight dequantize | 10240x1280 | 0.236 | 0.098 | 42% | memory |  |
| level 3 cross k/v native forward+dx | 1152x2048 @ 2048x1280 | 0.238 | 0.114 | 48% | compute |  |
| level 2 cross k/v native forward+dx | 4608x2048 @ 2048x640 | 0.431 | 0.229 | 53% | compute |  |
| level 2 feed-forward out native forward+dx | 4608x2560 @ 2560x640 | 0.531 | 0.286 | 54% | compute |  |
| level 2 feed-forward in native forward+dx | 4608x640 @ 640x5120 | 0.817 | 0.572 | 70% | compute |  |
| level 3 feed-forward out native forward+dx | 1152x5120 @ 5120x1280 | 0.374 | 0.286 | 77% | compute |  |
| level 3 feed-forward in native forward+dx | 1152x1280 @ 1280x10240 | 0.622 | 0.572 | 92% | compute |  |
| level 2 attention projection int8 | 4608x640 @ 640x640 | - | - | - | - | stays native: 640 input features is not a multiple of 256 |
| level 2 feed-forward in int8 | 4608x640 @ 640x5120 | - | - | - | - | stays native: 640 input features is not a multiple of 256 |

## launch

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| 200 tiny kernels, eager | 1024 elements | 2.675 | - | - | - | 13.4 us per kernel |
| 200 tiny kernels, graph replay | 1024 elements | 0.322 | - | - | - | 1.6 us per kernel, 8.3x faster than eager |

