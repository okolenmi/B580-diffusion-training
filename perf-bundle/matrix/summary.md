# Matrix summary

| scenario | status | wall ms | eager ms | replayed ms | device ms | utilisation | peak MB | top finding |
|---|---|---:|---:|---:|---:|---:|---:|---|
| native eager | ok | 1036 | 1036 | - | 669 | 65% | 6303 | Much of the module time has no cost model |
| native graph | ok | 1035 | 1029 | 5272 | 743 | 72% | 6528 | Dtype and copy kernels take a large share |
| int8 eager | ok | 1371 | 1371 | - | 794 | 58% | 4244 | Recomputing activations is a large part of the step |
| int8 graph | ok | 1325 | 1325 | 5422 | 853 | 64% | 4469 | Recomputing activations is a large part of the step |
| int8 dx-int8 eager | ok | 1607 | 1607 | - | 776 | 48% | 4251 | Much of the module time has no cost model |
| int8 dx-int8 graph | ok | 1560 | 1559 | 5760 | 853 | 55% | 4469 | The step is launch bound |
| native graph measured-checkpoints | ok | 1730 | 1730 | - | 654 | 38% | 7641 | Much of the module time has no cost model |
| int8 graph measured-checkpoints | ok | 2109 | 2109 | - | 777 | 37% | 6437 | Much of the module time has no cost model |
| native graph attention-math_bf16 | ok | 965 | 964 | 5071 | 708 | 73% | 6305 | Slowest pieces by recoverable time: attention transformer_blocks.attn1 (up) |
| native graph attention-chunked_bf16 | ok | 957 | 957 | 5074 | 784 | 82% | 6475 | Dtype and copy kernels take a large share |
| int8 graph attention-chunked_bf16 | ok | 1328 | 1326 | 5535 | 900 | 68% | 4417 | Recomputing activations is a large part of the step |
| native expandable-segments | ok | 1030 | 1030 | - | 667 | 65% | 6303 | Much of the module time has no cost model |
| native no-expandable-segments | ok | 965 | 965 | - | 658 | 68% | 6411 | Much of the module time has no cost model |

Every other scenario below is compared with the first, `native eager`.

## native graph against native eager

| step | native eager | native graph | change |
|---|---:|---:|---:|
| wall_ms | 1036 | 1035 | -1 |
| eager_ms | 1036 | 1029 | -7 |
| replayed_ms | - | 5272 | - |
| device_ms | 669 | 743 | +74 |
| launches | 25413 | 27628 | +2214 |
| utilisation | 65% | 72% | +7 points |
| peak_mb | 6303 | 6528 | +225 |

Device time by category (per step):

| category | native eager ms | native graph ms | change ms | calls native eager to native graph |
|---|---:|---:|---:|---|
| gemm | 241.8 | 281.7 | +39.9 | 8063 to 8390 |
| attention | 9.7 | 32.7 | +23.0 | 140 to 233 |
| copy | 168.5 | 191.0 | +22.4 | 4610 to 5829 |
| other | 46.0 | 33.9 | -12.1 | 610 to 516 |
| elementwise | 117.2 | 119.3 | +2.1 | 9051 to 9331 |
| norm | 20.5 | 19.6 | -0.9 | 782 to 782 |
| convolution | 61.7 | 61.0 | -0.8 | 512 to 466 |

Piece groups that changed most:

| group | native eager ms | native graph ms | change ms |
|---|---:|---:|---:|
| linear attn2.to_q up | 149.9 | 162.1 | +12.2 |
| norm transformer_blocks.norm1 down | 38.9 | 49.0 | +10.2 |
| linear attn1.to_q up | 160.7 | 151.5 | -9.2 |
| attention transformer_blocks.attn2 up | 192.1 | 183.5 | -8.6 |
| linear attn1.to_out up | 153.8 | 146.3 | -7.5 |
| attention transformer_blocks.attn1 up | 204.5 | 197.5 | -7.0 |
| attention transformer_blocks.attn2 down | 161.1 | 166.7 | +5.6 |
| linear attn2.to_out up | 154.0 | 148.4 | -5.6 |
| feedforward transformer_blocks.ff up | 115.7 | 110.7 | -5.0 |
| linear attn1.to_k up | 148.6 | 144.0 | -4.6 |
| linear attn2.to_v up | 149.4 | 144.9 | -4.5 |
| linear attn2.to_k up | 148.7 | 144.5 | -4.2 |

## int8 eager against native eager

| step | native eager | int8 eager | change |
|---|---:|---:|---:|
| wall_ms | 1036 | 1371 | +334 |
| eager_ms | 1036 | 1371 | +334 |
| device_ms | 669 | 794 | +125 |
| launches | 25413 | 32938 | +7525 |
| utilisation | 65% | 58% | -7 points |
| peak_mb | 6303 | 4244 | -2059 |

Device time by category (per step):

| category | native eager ms | int8 eager ms | change ms | calls native eager to int8 eager |
|---|---:|---:|---:|---|
| int8 | 0.0 | 148.3 | +148.3 | 0 to 3203 |
| gemm | 241.8 | 213.5 | -28.4 | 8063 to 8063 |
| memory | 2.2 | 4.9 | +2.7 | 1627 to 5443 |
| elementwise | 117.2 | 119.3 | +2.1 | 9051 to 9051 |
| convolution | 61.7 | 63.3 | +1.5 | 512 to 512 |
| norm | 20.5 | 20.1 | -0.4 | 782 to 782 |
| attention | 9.7 | 9.3 | -0.4 | 140 to 140 |
| copy | 168.5 | 168.2 | -0.4 | 4610 to 5116 |
| other | 46.0 | 46.3 | +0.2 | 610 to 610 |

Piece groups that changed most:

| group | native eager ms | int8 eager ms | change ms |
|---|---:|---:|---:|
| linear attn1.to_q up | 160.7 | 24.6 | -136.1 |
| linear net.proj up | 162.3 | 29.1 | -133.3 |
| linear attn1.to_out up | 153.8 | 23.9 | -129.9 |
| linear attn2.to_out up | 154.0 | 25.0 | -129.0 |
| linear attn2.to_q up | 149.9 | 24.3 | -125.5 |
| linear attn1.to_k up | 148.6 | 23.1 | -125.5 |
| linear attn1.to_v up | 141.3 | 16.6 | -124.7 |
| linear net.proj down | 116.5 | 22.0 | -94.5 |
| linear attn1.to_out down | 111.8 | 18.4 | -93.4 |
| linear attn1.to_q down | 109.9 | 18.7 | -91.2 |
| linear attn2.to_out down | 109.5 | 19.2 | -90.3 |
| linear attn1.to_k down | 107.8 | 18.9 | -88.9 |

## int8 graph against native eager

| step | native eager | int8 graph | change |
|---|---:|---:|---:|
| wall_ms | 1036 | 1325 | +289 |
| eager_ms | 1036 | 1325 | +289 |
| replayed_ms | - | 5422 | - |
| device_ms | 669 | 853 | +184 |
| launches | 25413 | 35109 | +9696 |
| utilisation | 65% | 64% | -0 points |
| peak_mb | 6303 | 4469 | -1834 |

Device time by category (per step):

| category | native eager ms | int8 graph ms | change ms | calls native eager to int8 graph |
|---|---:|---:|---:|---|
| int8 | 0.0 | 131.6 | +131.6 | 0 to 3203 |
| copy | 168.5 | 192.0 | +23.5 | 4610 to 6291 |
| attention | 9.7 | 32.6 | +22.9 | 140 to 233 |
| other | 46.0 | 34.3 | -11.7 | 610 to 516 |
| gemm | 241.8 | 252.8 | +11.0 | 8063 to 8390 |
| elementwise | 117.2 | 122.4 | +5.2 | 9051 to 9331 |
| memory | 2.2 | 4.9 | +2.7 | 1627 to 5879 |
| norm | 20.5 | 19.7 | -0.8 | 782 to 782 |
| convolution | 61.7 | 61.5 | -0.3 | 512 to 466 |
| reduce | 1.5 | 1.6 | +0.1 | 18 to 18 |

Piece groups that changed most:

| group | native eager ms | int8 graph ms | change ms |
|---|---:|---:|---:|
| linear attn1.to_q up | 160.7 | 24.9 | -135.8 |
| linear attn1.to_out up | 153.8 | 22.8 | -130.9 |
| linear net.proj up | 162.3 | 31.4 | -130.9 |
| linear attn2.to_out up | 154.0 | 24.7 | -129.2 |
| linear attn1.to_k up | 148.6 | 23.0 | -125.6 |
| linear attn2.to_q up | 149.9 | 24.3 | -125.5 |
| linear attn1.to_v up | 141.3 | 16.0 | -125.3 |
| linear net.proj down | 116.5 | 22.4 | -94.1 |
| linear attn1.to_out down | 111.8 | 17.8 | -94.1 |
| linear attn1.to_q down | 109.9 | 18.3 | -91.6 |
| linear attn2.to_out down | 109.5 | 18.9 | -90.6 |
| linear attn1.to_v down | 102.0 | 11.6 | -90.5 |

## int8 dx-int8 eager against native eager

| step | native eager | int8 dx-int8 eager | change |
|---|---:|---:|---:|
| wall_ms | 1036 | 1607 | +571 |
| eager_ms | 1036 | 1607 | +571 |
| device_ms | 669 | 776 | +107 |
| launches | 25413 | 38080 | +12667 |
| utilisation | 65% | 48% | -16 points |
| peak_mb | 6303 | 4251 | -2052 |

Device time by category (per step):

| category | native eager ms | int8 dx-int8 eager ms | change ms | calls native eager to int8 dx-int8 eager |
|---|---:|---:|---:|---|
| int8 | 0.0 | 126.0 | +126.0 | 0 to 4487 |
| gemm | 241.8 | 197.0 | -44.8 | 8063 to 8063 |
| elementwise | 117.2 | 132.4 | +15.2 | 9051 to 9693 |
| copy | 168.5 | 174.7 | +6.1 | 4610 to 5764 |
| memory | 2.2 | 7.2 | +5.0 | 1627 to 8011 |
| convolution | 61.7 | 62.4 | +0.6 | 512 to 512 |
| norm | 20.5 | 20.2 | -0.3 | 782 to 782 |
| attention | 9.7 | 9.4 | -0.3 | 140 to 140 |
| other | 46.0 | 45.8 | -0.3 | 610 to 610 |

Piece groups that changed most:

| group | native eager ms | int8 dx-int8 eager ms | change ms |
|---|---:|---:|---:|
| linear attn1.to_q up | 160.7 | 25.3 | -135.4 |
| linear net.proj up | 162.3 | 29.5 | -132.8 |
| linear attn1.to_out up | 153.8 | 24.3 | -129.5 |
| linear attn2.to_out up | 154.0 | 25.3 | -128.7 |
| linear attn1.to_k up | 148.6 | 24.4 | -124.2 |
| linear attn2.to_q up | 149.9 | 25.8 | -124.1 |
| linear attn1.to_v up | 141.3 | 17.4 | -123.9 |
| linear ff.net up | 151.9 | 251.9 | +99.9 |
| linear net.proj down | 116.5 | 22.8 | -93.7 |
| linear attn1.to_out down | 111.8 | 18.3 | -93.5 |
| linear attn2.to_k up | 148.7 | 240.7 | +92.0 |
| linear attn2.to_v up | 149.4 | 240.2 | +90.8 |

## int8 dx-int8 graph against native eager

| step | native eager | int8 dx-int8 graph | change |
|---|---:|---:|---:|
| wall_ms | 1036 | 1560 | +524 |
| eager_ms | 1036 | 1559 | +523 |
| replayed_ms | - | 5760 | - |
| device_ms | 669 | 853 | +183 |
| launches | 25413 | 40295 | +14881 |
| utilisation | 65% | 55% | -10 points |
| peak_mb | 6303 | 4469 | -1834 |

Device time by category (per step):

| category | native eager ms | int8 dx-int8 graph ms | change ms | calls native eager to int8 dx-int8 graph |
|---|---:|---:|---:|---|
| int8 | 0.0 | 125.2 | +125.2 | 0 to 4487 |
| copy | 168.5 | 196.6 | +28.1 | 4610 to 6983 |
| attention | 9.7 | 33.7 | +24.0 | 140 to 233 |
| elementwise | 117.2 | 135.1 | +17.9 | 9051 to 9973 |
| other | 46.0 | 35.1 | -11.0 | 610 to 516 |
| gemm | 241.8 | 237.3 | -4.5 | 8063 to 8390 |
| memory | 2.2 | 6.7 | +4.5 | 1627 to 8447 |
| convolution | 61.7 | 61.1 | -0.6 | 512 to 466 |
| norm | 20.5 | 20.4 | -0.1 | 782 to 782 |

Piece groups that changed most:

| group | native eager ms | int8 dx-int8 graph ms | change ms |
|---|---:|---:|---:|
| linear attn1.to_q up | 160.7 | 25.7 | -135.0 |
| linear net.proj up | 162.3 | 31.7 | -130.7 |
| linear attn1.to_out up | 153.8 | 24.5 | -129.3 |
| linear attn2.to_out up | 154.0 | 26.1 | -127.8 |
| linear attn1.to_v up | 141.3 | 17.2 | -124.1 |
| linear attn1.to_k up | 148.6 | 25.3 | -123.3 |
| linear attn2.to_q up | 149.9 | 26.7 | -123.2 |
| linear ff.net up | 151.9 | 256.4 | +104.4 |
| linear net.proj down | 116.5 | 23.0 | -93.5 |
| linear attn2.to_k up | 148.7 | 242.1 | +93.3 |
| linear attn1.to_out down | 111.8 | 18.7 | -93.2 |
| linear attn2.to_v up | 149.4 | 242.3 | +92.9 |

## native graph measured-checkpoints against native eager

| step | native eager | native graph measured-checkpoints | change |
|---|---:|---:|---:|
| wall_ms | 1036 | 1730 | +693 |
| eager_ms | 1036 | 1730 | +693 |
| device_ms | 669 | 654 | -15 |
| launches | 25413 | 42062 | +16648 |
| utilisation | 65% | 38% | -27 points |
| peak_mb | 6303 | 7641 | +1339 |

Device time by category (per step):

| category | native eager ms | native graph measured-checkpoints ms | change ms | calls native eager to native graph measured-checkpoints |
|---|---:|---:|---:|---|
| gemm | 241.8 | 263.3 | +21.5 | 8063 to 14114 |
| copy | 168.5 | 152.5 | -16.0 | 4610 to 7852 |
| elementwise | 117.2 | 101.9 | -15.3 | 9051 to 14705 |
| attention | 9.7 | 7.1 | -2.6 | 140 to 164 |
| convolution | 61.7 | 59.4 | -2.4 | 512 to 877 |
| reduce | 1.5 | 1.4 | -0.1 | 18 to 15 |
| norm | 20.5 | 20.6 | +0.1 | 782 to 1350 |

Piece groups that changed most:

| group | native eager ms | native graph measured-checkpoints ms | change ms |
|---|---:|---:|---:|
| attention transformer_blocks.attn2 up | 192.1 | 369.3 | +177.2 |
| linear attn2.to_q up | 149.9 | 293.5 | +143.7 |
| attention transformer_blocks.attn1 up | 204.5 | 341.6 | +137.1 |
| linear ff.net up | 151.9 | 282.6 | +130.7 |
| linear attn1.to_v up | 141.3 | 266.7 | +125.3 |
| linear net.proj up | 162.3 | 287.0 | +124.6 |
| linear attn1.to_k up | 148.6 | 272.7 | +124.1 |
| attention transformer_blocks.attn2 down | 161.1 | 284.6 | +123.5 |
| linear attn2.to_out up | 154.0 | 277.3 | +123.3 |
| linear attn1.to_out up | 153.8 | 276.3 | +122.5 |
| linear attn1.to_q up | 160.7 | 273.8 | +113.1 |
| feedforward transformer_blocks.ff up | 115.7 | 203.8 | +88.1 |

## int8 graph measured-checkpoints against native eager

| step | native eager | int8 graph measured-checkpoints | change |
|---|---:|---:|---:|
| wall_ms | 1036 | 2109 | +1073 |
| eager_ms | 1036 | 2109 | +1073 |
| device_ms | 669 | 777 | +108 |
| launches | 25413 | 49315 | +23902 |
| utilisation | 65% | 37% | -28 points |
| peak_mb | 6303 | 6437 | +134 |

Device time by category (per step):

| category | native eager ms | int8 graph measured-checkpoints ms | change ms | calls native eager to int8 graph measured-checkpoints |
|---|---:|---:|---:|---|
| int8 | 0.0 | 182.6 | +182.6 | 0 to 4412 |
| gemm | 241.8 | 208.2 | -33.6 | 8063 to 13034 |
| copy | 168.5 | 147.3 | -21.2 | 4610 to 7812 |
| elementwise | 117.2 | 98.0 | -19.2 | 9051 to 13841 |
| attention | 9.7 | 5.8 | -3.9 | 140 to 100 |
| memory | 2.2 | 5.6 | +3.4 | 1627 to 6738 |
| norm | 20.5 | 19.4 | -1.1 | 782 to 1262 |
| convolution | 61.7 | 62.7 | +0.9 | 512 to 910 |
| other | 46.0 | 45.7 | -0.3 | 610 to 1187 |

Piece groups that changed most:

| group | native eager ms | int8 graph measured-checkpoints ms | change ms |
|---|---:|---:|---:|
| attention transformer_blocks.attn2 up | 192.1 | 409.8 | +217.7 |
| linear ff.net up | 151.9 | 345.3 | +193.4 |
| linear ff.net down | 111.5 | 241.1 | +129.6 |
| attention transformer_blocks.attn2 down | 161.1 | 281.0 | +120.0 |
| linear attn1.to_q up | 160.7 | 51.3 | -109.4 |
| linear net.proj up | 162.3 | 53.4 | -108.9 |
| attention transformer_blocks.attn1 up | 204.5 | 312.1 | +107.6 |
| linear attn2.to_k up | 148.7 | 253.4 | +104.6 |
| linear attn1.to_out up | 153.8 | 50.6 | -103.2 |
| linear attn2.to_out up | 154.0 | 51.3 | -102.7 |
| linear attn2.to_q up | 149.9 | 50.9 | -98.9 |
| linear attn1.to_k up | 148.6 | 49.9 | -98.6 |

## native graph attention-math_bf16 against native eager

| step | native eager | native graph attention-math_bf16 | change |
|---|---:|---:|---:|
| wall_ms | 1036 | 965 | -71 |
| eager_ms | 1036 | 964 | -72 |
| replayed_ms | - | 5071 | - |
| device_ms | 669 | 708 | +39 |
| launches | 25413 | 26974 | +1561 |
| utilisation | 65% | 73% | +9 points |
| peak_mb | 6303 | 6305 | +2 |

Device time by category (per step):

| category | native eager ms | native graph attention-math_bf16 ms | change ms | calls native eager to native graph attention-math_bf16 |
|---|---:|---:|---:|---|
| gemm | 241.8 | 284.5 | +42.7 | 8063 to 8390 |
| attention | 9.7 | 24.1 | +14.4 | 140 to 233 |
| other | 46.0 | 33.9 | -12.1 | 610 to 516 |
| copy | 168.5 | 165.6 | -3.0 | 4610 to 5175 |
| norm | 20.5 | 18.9 | -1.6 | 782 to 782 |
| convolution | 61.7 | 61.0 | -0.7 | 512 to 466 |
| elementwise | 117.2 | 116.5 | -0.7 | 9051 to 9331 |
| memory | 2.2 | 2.3 | +0.1 | 1627 to 2063 |

Piece groups that changed most:

| group | native eager ms | native graph attention-math_bf16 ms | change ms |
|---|---:|---:|---:|
| attention transformer_blocks.attn2 down | 161.1 | 133.9 | -27.2 |
| attention transformer_blocks.attn1 down | 169.4 | 144.6 | -24.8 |
| attention transformer_blocks.attn2 up | 192.1 | 169.7 | -22.4 |
| attention transformer_blocks.attn1 up | 204.5 | 184.3 | -20.2 |
| feedforward transformer_blocks.ff down | 97.9 | 79.7 | -18.1 |
| linear attn1.to_q up | 160.7 | 142.9 | -17.8 |
| feedforward transformer_blocks.ff up | 115.7 | 100.8 | -15.0 |
| linear attn1.to_out up | 153.8 | 140.4 | -13.4 |
| linear attn1.to_out down | 111.8 | 99.3 | -12.5 |
| linear attn2.to_out up | 154.0 | 142.5 | -11.5 |
| linear ff.net down | 111.5 | 100.4 | -11.1 |
| linear attn2.to_out mid | 43.1 | 53.7 | +10.5 |

## native graph attention-chunked_bf16 against native eager

| step | native eager | native graph attention-chunked_bf16 | change |
|---|---:|---:|---:|
| wall_ms | 1036 | 957 | -79 |
| eager_ms | 1036 | 957 | -80 |
| replayed_ms | - | 5074 | - |
| device_ms | 669 | 784 | +115 |
| launches | 25413 | 28214 | +2801 |
| utilisation | 65% | 82% | +17 points |
| peak_mb | 6303 | 6475 | +172 |

Device time by category (per step):

| category | native eager ms | native graph attention-chunked_bf16 ms | change ms | calls native eager to native graph attention-chunked_bf16 |
|---|---:|---:|---:|---|
| copy | 168.5 | 218.7 | +50.2 | 4610 to 6155 |
| gemm | 241.8 | 283.1 | +41.3 | 8063 to 8466 |
| attention | 9.7 | 29.4 | +19.7 | 140 to 243 |
| elementwise | 117.2 | 134.2 | +17.0 | 9051 to 9455 |
| other | 46.0 | 33.7 | -12.3 | 610 to 516 |
| reduce | 1.5 | 3.4 | +1.9 | 18 to 68 |
| norm | 20.5 | 18.9 | -1.6 | 782 to 782 |
| convolution | 61.7 | 60.7 | -1.0 | 512 to 466 |

Piece groups that changed most:

| group | native eager ms | native graph attention-chunked_bf16 ms | change ms |
|---|---:|---:|---:|
| attention transformer_blocks.attn1 down | 169.4 | 143.7 | -25.7 |
| attention transformer_blocks.attn2 down | 161.1 | 135.7 | -25.4 |
| linear attn1.to_q up | 160.7 | 140.6 | -20.1 |
| feedforward transformer_blocks.ff down | 97.9 | 79.8 | -18.1 |
| linear attn1.to_out up | 153.8 | 138.4 | -15.4 |
| attention transformer_blocks.attn2 up | 192.1 | 177.0 | -15.1 |
| linear attn1.to_out down | 111.8 | 96.9 | -14.9 |
| attention transformer_blocks.attn1 up | 204.5 | 190.8 | -13.7 |
| linear ff.net down | 111.5 | 98.3 | -13.2 |
| linear attn1.to_k up | 148.6 | 135.6 | -13.0 |
| linear attn2.to_k up | 148.7 | 136.4 | -12.3 |
| linear attn2.to_v up | 149.4 | 137.2 | -12.2 |

## int8 graph attention-chunked_bf16 against native eager

| step | native eager | int8 graph attention-chunked_bf16 | change |
|---|---:|---:|---:|
| wall_ms | 1036 | 1328 | +291 |
| eager_ms | 1036 | 1326 | +290 |
| replayed_ms | - | 5535 | - |
| device_ms | 669 | 900 | +231 |
| launches | 25413 | 35739 | +10326 |
| utilisation | 65% | 68% | +3 points |
| peak_mb | 6303 | 4417 | -1886 |

Device time by category (per step):

| category | native eager ms | int8 graph attention-chunked_bf16 ms | change ms | calls native eager to int8 graph attention-chunked_bf16 |
|---|---:|---:|---:|---|
| int8 | 0.0 | 135.7 | +135.7 | 0 to 3203 |
| copy | 168.5 | 222.1 | +53.6 | 4610 to 6661 |
| elementwise | 117.2 | 137.3 | +20.1 | 9051 to 9455 |
| attention | 9.7 | 29.4 | +19.7 | 140 to 243 |
| other | 46.0 | 33.9 | -12.1 | 610 to 516 |
| gemm | 241.8 | 253.2 | +11.4 | 8063 to 8466 |
| memory | 2.2 | 4.8 | +2.6 | 1627 to 5879 |
| reduce | 1.5 | 3.4 | +1.9 | 18 to 68 |
| norm | 20.5 | 19.1 | -1.4 | 782 to 782 |
| convolution | 61.7 | 61.2 | -0.6 | 512 to 466 |

Piece groups that changed most:

| group | native eager ms | int8 graph attention-chunked_bf16 ms | change ms |
|---|---:|---:|---:|
| linear attn1.to_q up | 160.7 | 24.0 | -136.7 |
| linear attn1.to_out up | 153.8 | 22.1 | -131.7 |
| linear net.proj up | 162.3 | 31.2 | -131.2 |
| linear attn2.to_out up | 154.0 | 23.9 | -130.0 |
| linear attn1.to_k up | 148.6 | 21.9 | -126.7 |
| linear attn2.to_q up | 149.9 | 23.6 | -126.3 |
| linear attn1.to_v up | 141.3 | 15.8 | -125.5 |
| linear net.proj down | 116.5 | 20.9 | -95.6 |
| linear attn1.to_out down | 111.8 | 16.9 | -95.0 |
| linear attn1.to_q down | 109.9 | 17.5 | -92.4 |
| linear attn2.to_out down | 109.5 | 17.7 | -91.9 |
| linear attn1.to_v down | 102.0 | 11.1 | -90.9 |

## native expandable-segments against native eager

| step | native eager | native expandable-segments | change |
|---|---:|---:|---:|
| wall_ms | 1036 | 1030 | -7 |
| eager_ms | 1036 | 1030 | -7 |
| device_ms | 669 | 667 | -2 |
| launches | 25413 | 25413 | +0 |
| utilisation | 65% | 65% | +0 points |
| peak_mb | 6303 | 6303 | +0 |

Device time by category (per step):

| category | native eager ms | native expandable-segments ms | change ms | calls native eager to native expandable-segments |
|---|---:|---:|---:|---|
| other | 46.0 | 45.3 | -0.7 | 610 to 610 |
| norm | 20.5 | 19.8 | -0.7 | 782 to 782 |
| attention | 9.7 | 9.0 | -0.7 | 140 to 140 |
| gemm | 241.8 | 242.5 | +0.6 | 8063 to 8063 |
| elementwise | 117.2 | 116.8 | -0.4 | 9051 to 9051 |
| copy | 168.5 | 168.3 | -0.3 | 4610 to 4610 |
| convolution | 61.7 | 61.9 | +0.1 | 512 to 512 |
| memory | 2.2 | 2.1 | -0.1 | 1627 to 1627 |

Piece groups that changed most:

| group | native eager ms | native expandable-segments ms | change ms |
|---|---:|---:|---:|
| linear attn1.to_v up | 141.3 | 157.1 | +15.8 |
| linear attn1.to_q up | 160.7 | 153.2 | -7.6 |
| feedforward transformer_blocks.ff down | 97.9 | 91.2 | -6.7 |
| attention transformer_blocks.attn2 down | 161.1 | 155.2 | -5.9 |
| attention transformer_blocks.attn1 down | 169.4 | 164.0 | -5.4 |
| attention transformer_blocks.attn1 up | 204.5 | 199.2 | -5.3 |
| attention transformer_blocks.attn2 up | 192.1 | 187.2 | -4.8 |
| linear attn1.to_v down | 102.0 | 106.0 | +4.0 |
| linear ff.net mid | 44.2 | 48.1 | +3.9 |
| attention transformer_blocks.attn1 mid | 60.0 | 63.6 | +3.6 |
| linear net.proj up | 162.3 | 165.7 | +3.4 |
| linear ff.net down | 111.5 | 108.2 | -3.3 |

## native no-expandable-segments against native eager

| step | native eager | native no-expandable-segments | change |
|---|---:|---:|---:|
| wall_ms | 1036 | 965 | -71 |
| eager_ms | 1036 | 965 | -71 |
| device_ms | 669 | 658 | -11 |
| launches | 25413 | 25413 | +0 |
| utilisation | 65% | 68% | +4 points |
| peak_mb | 6303 | 6411 | +108 |

Device time by category (per step):

| category | native eager ms | native no-expandable-segments ms | change ms | calls native eager to native no-expandable-segments |
|---|---:|---:|---:|---|
| copy | 168.5 | 162.1 | -6.4 | 4610 to 4610 |
| other | 46.0 | 41.1 | -5.0 | 610 to 610 |
| gemm | 241.8 | 246.1 | +4.3 | 8063 to 8063 |
| elementwise | 117.2 | 115.1 | -2.1 | 9051 to 9051 |
| norm | 20.5 | 19.0 | -1.5 | 782 to 782 |
| attention | 9.7 | 8.9 | -0.8 | 140 to 140 |
| convolution | 61.7 | 62.2 | +0.5 | 512 to 512 |
| memory | 2.2 | 2.1 | -0.1 | 1627 to 1627 |

Piece groups that changed most:

| group | native eager ms | native no-expandable-segments ms | change ms |
|---|---:|---:|---:|
| attention transformer_blocks.attn1 down | 169.4 | 134.7 | -34.7 |
| attention transformer_blocks.attn2 down | 161.1 | 126.9 | -34.2 |
| attention transformer_blocks.attn1 up | 204.5 | 173.1 | -31.4 |
| attention transformer_blocks.attn2 up | 192.1 | 161.3 | -30.8 |
| linear attn1.to_q up | 160.7 | 136.8 | -23.9 |
| feedforward transformer_blocks.ff down | 97.9 | 76.0 | -21.9 |
| feedforward transformer_blocks.ff up | 115.7 | 96.1 | -19.6 |
| linear attn2.to_out up | 154.0 | 135.6 | -18.3 |
| linear attn1.to_out up | 153.8 | 135.7 | -18.0 |
| linear ff.net up | 151.9 | 135.2 | -16.8 |
| linear attn1.to_out down | 111.8 | 95.4 | -16.5 |
| linear attn2.to_k up | 148.7 | 133.4 | -15.4 |

