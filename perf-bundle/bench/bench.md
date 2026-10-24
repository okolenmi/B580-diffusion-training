# Micro-benchmarks (xpu, batch 2, 768 px)

Ceilings: gemm 105.4 TFLOPS, int8 199.3 TOPS, copy 397 GB/s. Efficiency is ideal time over measured time, ideal from the ceiling of whatever bounds the operation.

## gemm

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 1 cross k/v | 18432x2048 @ 2048x320 | 0.376 | 0.229 | 61% | compute |  |
| level 1 ff out | 18432x1280 @ 1280x320 | 0.244 | 0.150 | 62% | memory |  |
| level 2 projection | 4608x640 @ 640x640 | 0.054 | 0.036 | 67% | compute |  |
| level 3 cross k/v | 1152x2048 @ 2048x1280 | 0.078 | 0.057 | 73% | compute |  |
| level 2 cross k/v | 4608x2048 @ 2048x640 | 0.150 | 0.115 | 76% | compute |  |
| level 1 projection | 18432x320 @ 320x320 | 0.078 | 0.060 | 77% | memory |  |
| level 1 ff in | 18432x320 @ 320x2560 | 0.373 | 0.286 | 77% | compute |  |
| level 2 ff out | 4608x2560 @ 2560x640 | 0.182 | 0.143 | 79% | compute |  |
| level 3 projection | 1152x1280 @ 1280x1280 | 0.045 | 0.036 | 79% | compute |  |
| level 2 ff in | 4608x640 @ 640x5120 | 0.337 | 0.286 | 85% | compute |  |
| level 3 ff in | 1152x1280 @ 1280x10240 | 0.328 | 0.286 | 87% | compute |  |
| level 3 ff out | 1152x5120 @ 5120x1280 | 0.154 | 0.143 | 93% | compute |  |

## convolution

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 3 forward+dx | 2x1280x24x24 | 1.769 | 0.645 | 36% | compute |  |
| level 1 forward+dx | 2x320x96x96 | 1.743 | 0.645 | 37% | compute |  |
| level 3 forward | 2x1280x24x24 | 0.852 | 0.322 | 38% | compute |  |
| level 1 forward | 2x320x96x96 | 0.513 | 0.322 | 63% | compute |  |
| level 2 forward+dx | 2x640x48x48 | 0.879 | 0.645 | 73% | compute |  |
| level 2 forward | 2x640x48x48 | 0.414 | 0.322 | 78% | compute |  |

## norm

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 3 group norm forward+dx | 2x1280x24x24 | 0.380 | 0.037 | 10% | memory |  |
| level 3 layer norm forward+dx | 1152x1280 | 0.284 | 0.037 | 13% | memory |  |
| level 3 group norm forward | 2x1280x24x24 | 0.081 | 0.015 | 18% | memory |  |
| level 2 group norm forward+dx | 2x640x48x48 | 0.363 | 0.074 | 20% | memory |  |
| level 1 layer norm forward+dx | 18432x320 | 0.594 | 0.148 | 25% | memory |  |
| level 2 layer norm forward+dx | 4608x640 | 0.290 | 0.074 | 26% | memory |  |
| level 1 group norm forward+dx | 2x320x96x96 | 0.429 | 0.148 | 35% | memory |  |
| level 2 group norm forward | 2x640x48x48 | 0.081 | 0.030 | 36% | memory |  |
| level 1 group norm forward | 2x320x96x96 | 0.126 | 0.059 | 47% | memory |  |

## elementwise

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| chain of 8 (mul, add, silu) | 2949120 elements | 0.751 | 0.059 | 8% | memory | ideal is one fused pass (reads x, y, z, writes once) |
| cast to float32 | 2949120 elements | 0.049 | 0.045 | 90% | memory |  |
| silu | 2949120 elements | 0.022 | 0.030 | 137% | memory |  |
| add | 2949120 elements | 0.013 | 0.045 | 338% | memory |  |
| mul | 2949120 elements | 0.012 | 0.045 | 366% | memory |  |

## attention

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 3 self math forward | 2x20x576x64 (self, keys 576) | 1.072 | 0.032 | 3% | compute |  |
| level 3 self efficient forward | 2x20x576x64 (self, keys 576) | 1.068 | 0.032 | 3% | compute |  |
| level 3 self chunked_bf16 forward+backward | 2x20x576x64 (self, keys 576) | 3.660 | 0.113 | 3% | compute | captures; replay 3.64 ms |
| level 2 self chunked_bf16 forward+backward | 2x10x2304x64 (self, keys 2304) | 27.416 | 0.902 | 3% | compute | captures; replay 27.52 ms |
| level 3 self chunked_bf16 forward | 2x20x576x64 (self, keys 576) | 0.949 | 0.032 | 3% | compute |  |
| level 2 self chunked_bf16 forward | 2x10x2304x64 (self, keys 2304) | 7.380 | 0.258 | 3% | compute |  |
| level 2 self efficient forward | 2x10x2304x64 (self, keys 2304) | 7.011 | 0.258 | 4% | compute |  |
| level 2 self math forward | 2x10x2304x64 (self, keys 2304) | 7.000 | 0.258 | 4% | compute |  |
| level 3 cross chunked_bf16 forward+backward | 2x20x576x64 (cross, keys 77) | 1.290 | 0.050 | 4% | memory | captures; replay 0.73 ms |
| level 3 self math forward+backward | 2x20x576x64 (self, keys 576) | 2.437 | 0.113 | 5% | compute | captures; replay 2.42 ms |
| level 3 self efficient forward+backward | 2x20x576x64 (self, keys 576) | 2.434 | 0.113 | 5% | compute | captures; replay 2.42 ms |
| level 3 cross math forward+backward | 2x20x576x64 (cross, keys 77) | 0.997 | 0.050 | 5% | memory | captures; replay 0.53 ms |
| level 3 cross efficient forward+backward | 2x20x576x64 (cross, keys 77) | 0.989 | 0.050 | 5% | memory | captures; replay 0.53 ms |
| level 2 self efficient forward+backward | 2x10x2304x64 (self, keys 2304) | 15.512 | 0.902 | 6% | compute | captures; replay 15.64 ms |
| level 2 self math forward+backward | 2x10x2304x64 (self, keys 2304) | 15.486 | 0.902 | 6% | compute | captures; replay 15.61 ms |
| level 2 cross efficient forward | 2x10x2304x64 (cross, keys 77) | 0.518 | 0.031 | 6% | memory |  |
| level 2 cross math forward | 2x10x2304x64 (cross, keys 77) | 0.516 | 0.031 | 6% | memory |  |
| level 3 cross math forward | 2x20x576x64 (cross, keys 77) | 0.282 | 0.017 | 6% | memory |  |
| level 3 cross efficient forward | 2x20x576x64 (cross, keys 77) | 0.281 | 0.017 | 6% | memory |  |
| level 2 cross chunked_bf16 forward+backward | 2x10x2304x64 (cross, keys 77) | 1.522 | 0.092 | 6% | memory | captures; replay 1.52 ms |
| level 3 cross math_bf16 forward+backward | 2x20x576x64 (cross, keys 77) | 0.803 | 0.050 | 6% | memory | captures; replay 0.28 ms |
| level 3 cross chunked_bf16 forward | 2x20x576x64 (cross, keys 77) | 0.259 | 0.017 | 6% | memory |  |
| level 2 cross math forward+backward | 2x10x2304x64 (cross, keys 77) | 1.184 | 0.092 | 8% | memory | captures; replay 1.17 ms |
| level 2 cross efficient forward+backward | 2x10x2304x64 (cross, keys 77) | 1.175 | 0.092 | 8% | memory | captures; replay 1.18 ms |
| level 3 cross math_bf16 forward | 2x20x576x64 (cross, keys 77) | 0.197 | 0.017 | 9% | memory |  |
| level 2 cross chunked_bf16 forward | 2x10x2304x64 (cross, keys 77) | 0.355 | 0.031 | 9% | memory |  |
| level 3 self math_bf16 forward | 2x20x576x64 (self, keys 576) | 0.370 | 0.032 | 9% | compute |  |
| level 2 cross math_bf16 forward+backward | 2x10x2304x64 (cross, keys 77) | 0.879 | 0.092 | 10% | memory | captures; replay 0.73 ms |
| level 3 self math_bf16 forward+backward | 2x20x576x64 (self, keys 576) | 1.003 | 0.113 | 11% | compute | captures; replay 1.01 ms |
| level 2 self math_bf16 forward | 2x10x2304x64 (self, keys 2304) | 2.290 | 0.258 | 11% | compute |  |
| level 2 self math_bf16 forward+backward | 2x10x2304x64 (self, keys 2304) | 6.298 | 0.902 | 14% | compute | captures; replay 6.34 ms |
| level 2 cross math_bf16 forward | 2x10x2304x64 (cross, keys 77) | 0.205 | 0.031 | 15% | memory |  |
| level 3 cross flash forward+backward | 2x20x576x64 (cross, keys 77) | 0.265 | 0.050 | 19% | memory | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 3 self default forward+backward | 2x20x576x64 (self, keys 576) | 0.516 | 0.113 | 22% | compute | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 3 self flash forward+backward | 2x20x576x64 (self, keys 576) | 0.512 | 0.113 | 22% | compute | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 3 cross default forward+backward | 2x20x576x64 (cross, keys 77) | 0.180 | 0.050 | 28% | memory | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 2 cross default forward+backward | 2x10x2304x64 (cross, keys 77) | 0.321 | 0.092 | 29% | memory | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 2 cross flash forward+backward | 2x10x2304x64 (cross, keys 77) | 0.318 | 0.092 | 29% | memory | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 2 self flash forward+backward | 2x10x2304x64 (self, keys 2304) | 2.886 | 0.902 | 31% | compute | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 2 self default forward+backward | 2x10x2304x64 (self, keys 2304) | 2.859 | 0.902 | 32% | compute | cannot be captured: RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Gra |
| level 3 self default forward | 2x20x576x64 (self, keys 576) | 0.058 | 0.032 | 56% | compute |  |
| level 3 self flash forward | 2x20x576x64 (self, keys 576) | 0.058 | 0.032 | 56% | compute |  |
| level 3 cross default forward | 2x20x576x64 (cross, keys 77) | 0.026 | 0.017 | 65% | memory |  |
| level 3 cross flash forward | 2x20x576x64 (cross, keys 77) | 0.026 | 0.017 | 65% | memory |  |
| level 2 cross default forward | 2x10x2304x64 (cross, keys 77) | 0.043 | 0.031 | 72% | memory |  |
| level 2 cross flash forward | 2x10x2304x64 (cross, keys 77) | 0.042 | 0.031 | 72% | memory |  |
| level 2 self flash forward | 2x10x2304x64 (self, keys 2304) | 0.337 | 0.258 | 77% | compute |  |
| level 2 self default forward | 2x10x2304x64 (self, keys 2304) | 0.337 | 0.258 | 77% | compute |  |

## linear layer

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| level 3 attention projection int8 (int8 dx) forward+dx | 1152x1280 @ 1280x1280 | 0.882 | 0.038 | 4% | memory | 0.31x of native speed |
| level 3 attention projection int8 (dense dx) forward+dx | 1152x1280 @ 1280x1280 | 0.554 | 0.038 | 7% | memory | 0.49x of native speed |
| level 3 cross k/v int8 (int8 dx) forward+dx | 1152x2048 @ 2048x1280 | 0.862 | 0.061 | 7% | compute | 0.24x of native speed |
| level 2 feed-forward out int8 (int8 dx) forward+dx | 4608x2560 @ 2560x640 | 1.580 | 0.157 | 10% | memory | 0.34x of native speed |
| level 2 cross k/v int8 (int8 dx) forward+dx | 4608x2048 @ 2048x640 | 1.248 | 0.131 | 11% | memory | 0.35x of native speed |
| level 3 cross k/v int8 (dense dx) forward+dx | 1152x2048 @ 2048x1280 | 0.565 | 0.061 | 11% | compute | 0.36x of native speed |
| level 3 feed-forward out weight dequantize | 1280x5120 | 0.399 | 0.049 | 12% | memory |  |
| level 3 feed-forward out int8 (int8 dx) forward+dx | 1152x5120 @ 5120x1280 | 0.908 | 0.152 | 17% | compute | 0.41x of native speed |
| level 3 cross k/v weight dequantize | 1280x2048 | 0.111 | 0.020 | 18% | memory |  |
| level 3 feed-forward out int8 (dense dx) forward+dx | 1152x5120 @ 5120x1280 | 0.793 | 0.152 | 19% | compute | 0.47x of native speed |
| level 2 attention projection native forward+dx | 4608x640 @ 640x640 | 0.358 | 0.072 | 20% | compute |  |
| level 2 cross k/v int8 (dense dx) forward+dx | 4608x2048 @ 2048x640 | 0.584 | 0.131 | 22% | memory | 0.74x of native speed |
| level 2 feed-forward out int8 (dense dx) forward+dx | 4608x2560 @ 2560x640 | 0.683 | 0.157 | 23% | memory | 0.78x of native speed |
| level 3 attention projection weight dequantize | 1280x1280 | 0.047 | 0.012 | 26% | memory |  |
| level 3 attention projection native forward+dx | 1152x1280 @ 1280x1280 | 0.271 | 0.072 | 26% | compute |  |
| level 3 feed-forward in int8 (dense dx) forward+dx | 1152x1280 @ 1280x10240 | 1.118 | 0.303 | 27% | compute | 0.56x of native speed |
| level 2 cross k/v weight dequantize | 640x2048 | 0.036 | 0.010 | 28% | memory |  |
| level 3 feed-forward in int8 (int8 dx) forward+dx | 1152x1280 @ 1280x10240 | 0.975 | 0.303 | 31% | compute | 0.64x of native speed |
| level 3 feed-forward in weight dequantize | 10240x1280 | 0.311 | 0.099 | 32% | memory |  |
| level 2 feed-forward out weight dequantize | 640x2560 | 0.036 | 0.012 | 35% | memory |  |
| level 2 cross k/v native forward+dx | 4608x2048 @ 2048x640 | 0.432 | 0.229 | 53% | compute |  |
| level 2 feed-forward out native forward+dx | 4608x2560 @ 2560x640 | 0.530 | 0.286 | 54% | compute |  |
| level 3 cross k/v native forward+dx | 1152x2048 @ 2048x1280 | 0.203 | 0.115 | 57% | compute |  |
| level 2 feed-forward in native forward+dx | 4608x640 @ 640x5120 | 0.872 | 0.573 | 66% | compute |  |
| level 3 feed-forward out native forward+dx | 1152x5120 @ 5120x1280 | 0.373 | 0.286 | 77% | compute |  |
| level 3 feed-forward in native forward+dx | 1152x1280 @ 1280x10240 | 0.621 | 0.573 | 92% | compute |  |
| level 2 attention projection int8 | 4608x640 @ 640x640 | - | - | - | - | stays native: 640 input features is not a multiple of 256 |
| level 2 feed-forward in int8 | 4608x640 @ 640x5120 | - | - | - | - | stays native: 640 input features is not a multiple of 256 |

## launch

| operation | shape | ms | ideal ms | efficiency | bound | note |
|---|---|---:|---:|---:|---|---|
| 200 tiny kernels, eager | 1024 elements | 2.520 | - | - | - | 12.6 us per kernel |
| 200 tiny kernels, graph replay | 1024 elements | 0.333 | - | - | - | 1.7 us per kernel, 7.6x faster than eager |

