# Matrix summary

| scenario | status | wall ms | eager ms | replayed ms | device ms | utilisation | peak MB | top finding |
|---|---|---:|---:|---:|---:|---:|---:|---|
| native eager | ok | 8442 | 8442 | - | 614 | 7% | 6125 | The step is launch bound |
| native graph | ok | 9636 | 8170 | 12567 | 620 | 6% | 6126 | The step is launch bound |
| int8 eager | ok | 11086 | 11086 | - | 805 | 7% | 4066 | The step is launch bound |
| int8 graph | ok | 10967 | 9493 | 13917 | 744 | 7% | 4067 | The step is launch bound |
| int8 dx-int8 eager | ok | 9919 | 9919 | - | 689 | 7% | 4072 | The step is launch bound |
| int8 dx-int8 graph | ok | 9713 | 8206 | 12725 | 682 | 7% | 4073 | The step is launch bound |
| native graph measured-checkpoints | ok | 11973 | 11973 | - | 617 | 5% | 7182 | The step is launch bound |
| int8 graph measured-checkpoints | ok | 12958 | 12958 | - | 727 | 6% | 5728 | The step is launch bound |
| native graph attention-math_bf16 | ok | 7381 | 6003 | 10136 | 598 | 8% | 6126 | The step is launch bound |
| native graph attention-chunked_bf16 | ok | 7382 | 6002 | 10142 | 595 | 8% | 6126 | The step is launch bound |
| int8 graph attention-chunked_bf16 | ok | 8688 | 7262 | 11541 | 695 | 8% | 4067 | The step is launch bound |
| native expandable-segments | ok | 7627 | 7627 | - | 642 | 8% | 6125 | The step is launch bound |
| native no-expandable-segments | ok | 7399 | 7399 | - | 603 | 8% | 6244 | The step is launch bound |

Every other scenario below is compared with the first, `native eager`.

## native graph against native eager

| step | native eager | native graph | change |
|---|---:|---:|---:|
| wall_ms | 8442 | 9636 | +1194 |
| eager_ms | 8442 | 8170 | -272 |
| replayed_ms | - | 12567 | - |
| device_ms | 614 | 620 | +6 |
| launches | 25434 | 25434 | +0 |
| utilisation | 7% | 6% | -1 points |
| peak_mb | 6125 | 6126 | +1 |

Device time by category (per step):

| category | native eager ms | native graph ms | change ms | calls native eager to native graph |
|---|---:|---:|---:|---|
| copy | 146.7 | 150.9 | +4.2 | 4610 to 4610 |
| elementwise | 102.0 | 104.8 | +2.8 | 9038 to 9038 |
| convolution | 74.1 | 73.0 | -1.1 | 533 to 533 |
| other | 37.0 | 36.5 | -0.6 | 623 to 623 |
| norm | 18.1 | 18.5 | +0.4 | 782 to 782 |
| gemm | 223.9 | 224.2 | +0.4 | 8063 to 8063 |
| reduce | 1.6 | 1.5 | -0.1 | 18 to 18 |
| attention | 8.4 | 8.3 | -0.1 | 140 to 140 |

Piece groups that changed most:

| group | native eager ms | native graph ms | change ms |
|---|---:|---:|---:|
| conv output_blocks.in_layers up | 459.0 | 709.6 | +250.6 |
| attention transformer_blocks.attn1 up | 226.4 | 437.6 | +211.2 |
| attention transformer_blocks.attn2 up | 215.7 | 423.6 | +207.9 |
| attention transformer_blocks.attn1 down | 198.7 | 384.0 | +185.3 |
| attention transformer_blocks.attn2 down | 191.1 | 376.1 | +185.0 |
| conv output_blocks.skip_connection up | 308.0 | 476.6 | +168.6 |
| linear attn2.to_q up | 163.6 | 293.4 | +129.8 |
| feedforward transformer_blocks.ff up | 130.4 | 257.4 | +127.1 |
| linear ff.net up | 168.7 | 285.9 | +117.3 |
| linear net.proj up | 175.2 | 291.6 | +116.5 |
| linear attn2.to_v up | 161.7 | 276.2 | +114.5 |
| linear attn1.to_k up | 162.9 | 275.5 | +112.6 |

## int8 eager against native eager

| step | native eager | int8 eager | change |
|---|---:|---:|---:|
| wall_ms | 8442 | 11086 | +2644 |
| eager_ms | 8442 | 11086 | +2644 |
| device_ms | 614 | 805 | +191 |
| launches | 25434 | 32959 | +7525 |
| utilisation | 7% | 7% | -0 points |
| peak_mb | 6125 | 4066 | -2059 |

Device time by category (per step):

| category | native eager ms | int8 eager ms | change ms | calls native eager to int8 eager |
|---|---:|---:|---:|---|
| int8 | 0.0 | 169.3 | +169.3 | 0 to 3203 |
| copy | 146.7 | 183.8 | +37.1 | 4610 to 5116 |
| gemm | 223.9 | 194.5 | -29.3 | 8063 to 8063 |
| elementwise | 102.0 | 109.0 | +7.0 | 9038 to 9038 |
| memory | 2.0 | 5.4 | +3.3 | 1627 to 5443 |
| norm | 18.1 | 19.3 | +1.2 | 782 to 782 |
| other | 37.0 | 38.2 | +1.2 | 623 to 623 |
| convolution | 74.1 | 75.0 | +1.0 | 533 to 533 |
| attention | 8.4 | 9.0 | +0.6 | 140 to 140 |

Piece groups that changed most:

| group | native eager ms | int8 eager ms | change ms |
|---|---:|---:|---:|
| linear attn2.to_out up | 180.9 | 31.0 | -149.9 |
| linear net.proj up | 175.2 | 33.9 | -141.3 |
| linear attn1.to_q up | 166.5 | 29.9 | -136.5 |
| linear attn1.to_out up | 166.3 | 30.2 | -136.1 |
| linear attn2.to_q up | 163.6 | 30.2 | -133.4 |
| linear attn1.to_k up | 162.9 | 29.6 | -133.3 |
| linear attn1.to_v up | 157.3 | 25.6 | -131.8 |
| linear net.proj down | 129.6 | 25.5 | -104.1 |
| linear attn2.to_q down | 121.6 | 23.7 | -97.9 |
| linear attn1.to_v down | 119.1 | 21.3 | -97.8 |
| linear attn2.to_out down | 120.7 | 23.8 | -97.0 |
| linear attn1.to_q down | 121.6 | 26.7 | -94.9 |

## int8 graph against native eager

| step | native eager | int8 graph | change |
|---|---:|---:|---:|
| wall_ms | 8442 | 10967 | +2525 |
| eager_ms | 8442 | 9493 | +1051 |
| replayed_ms | - | 13917 | - |
| device_ms | 614 | 744 | +130 |
| launches | 25434 | 32959 | +7525 |
| utilisation | 7% | 7% | -0 points |
| peak_mb | 6125 | 4067 | -2058 |

Device time by category (per step):

| category | native eager ms | int8 graph ms | change ms | calls native eager to int8 graph |
|---|---:|---:|---:|---|
| int8 | 0.0 | 138.6 | +138.6 | 0 to 3203 |
| gemm | 223.9 | 194.1 | -29.8 | 8063 to 8063 |
| copy | 146.7 | 157.5 | +10.8 | 4610 to 5116 |
| elementwise | 102.0 | 106.5 | +4.4 | 9038 to 9038 |
| memory | 2.0 | 5.1 | +3.1 | 1627 to 5443 |
| convolution | 74.1 | 76.8 | +2.7 | 533 to 533 |
| other | 37.0 | 36.0 | -1.0 | 623 to 623 |
| attention | 8.4 | 9.1 | +0.7 | 140 to 140 |
| norm | 18.1 | 18.7 | +0.6 | 782 to 782 |
| reduce | 1.6 | 1.5 | -0.1 | 18 to 18 |

Piece groups that changed most:

| group | native eager ms | int8 graph ms | change ms |
|---|---:|---:|---:|
| conv output_blocks.in_layers up | 459.0 | 722.5 | +263.5 |
| attention transformer_blocks.attn1 up | 226.4 | 458.0 | +231.6 |
| linear ff.net up | 168.7 | 394.5 | +225.9 |
| linear attn2.to_k up | 160.9 | 384.4 | +223.5 |
| attention transformer_blocks.attn2 up | 215.7 | 438.9 | +223.2 |
| linear attn2.to_v up | 161.7 | 372.4 | +210.7 |
| attention transformer_blocks.attn2 down | 191.1 | 380.0 | +188.9 |
| attention transformer_blocks.attn1 down | 198.7 | 386.9 | +188.1 |
| conv output_blocks.skip_connection up | 308.0 | 484.2 | +176.2 |
| linear ff.net down | 121.9 | 275.9 | +154.0 |
| linear attn2.to_k down | 120.9 | 263.2 | +142.3 |
| linear attn2.to_v down | 121.4 | 262.9 | +141.4 |

## int8 dx-int8 eager against native eager

| step | native eager | int8 dx-int8 eager | change |
|---|---:|---:|---:|
| wall_ms | 8442 | 9919 | +1477 |
| eager_ms | 8442 | 9919 | +1477 |
| device_ms | 614 | 689 | +75 |
| launches | 25434 | 38101 | +12667 |
| utilisation | 7% | 7% | -0 points |
| peak_mb | 6125 | 4072 | -2053 |

Device time by category (per step):

| category | native eager ms | int8 dx-int8 eager ms | change ms | calls native eager to int8 dx-int8 eager |
|---|---:|---:|---:|---|
| int8 | 0.0 | 100.6 | +100.6 | 0 to 4487 |
| gemm | 223.9 | 178.5 | -45.3 | 8063 to 8063 |
| elementwise | 102.0 | 114.4 | +12.3 | 9038 to 9680 |
| memory | 2.0 | 6.4 | +4.4 | 1627 to 8011 |
| copy | 146.7 | 151.1 | +4.4 | 4610 to 5764 |
| other | 37.0 | 36.3 | -0.7 | 623 to 623 |
| convolution | 74.1 | 73.4 | -0.7 | 533 to 533 |
| norm | 18.1 | 17.9 | -0.1 | 782 to 782 |
| attention | 8.4 | 8.5 | +0.1 | 140 to 140 |
| reduce | 1.6 | 1.5 | -0.1 | 18 to 18 |

Piece groups that changed most:

| group | native eager ms | int8 dx-int8 eager ms | change ms |
|---|---:|---:|---:|
| linear attn2.to_out up | 180.9 | 26.8 | -154.2 |
| linear net.proj up | 175.2 | 30.4 | -144.8 |
| linear attn1.to_out up | 166.3 | 27.3 | -139.0 |
| linear attn1.to_q up | 166.5 | 28.3 | -138.1 |
| linear attn2.to_q up | 163.6 | 26.6 | -136.9 |
| linear attn1.to_v up | 157.3 | 21.9 | -135.4 |
| linear attn1.to_k up | 162.9 | 27.6 | -135.3 |
| linear net.proj down | 129.6 | 21.6 | -108.0 |
| linear attn1.to_q down | 121.6 | 18.6 | -103.0 |
| linear attn2.to_out down | 120.7 | 19.3 | -101.4 |
| linear attn1.to_k down | 117.7 | 18.2 | -99.5 |
| linear attn1.to_v down | 119.1 | 22.3 | -96.8 |

## int8 dx-int8 graph against native eager

| step | native eager | int8 dx-int8 graph | change |
|---|---:|---:|---:|
| wall_ms | 8442 | 9713 | +1271 |
| eager_ms | 8442 | 8206 | -236 |
| replayed_ms | - | 12725 | - |
| device_ms | 614 | 682 | +68 |
| launches | 25434 | 38101 | +12667 |
| utilisation | 7% | 7% | -0 points |
| peak_mb | 6125 | 4073 | -2052 |

Device time by category (per step):

| category | native eager ms | int8 dx-int8 graph ms | change ms | calls native eager to int8 dx-int8 graph |
|---|---:|---:|---:|---|
| int8 | 0.0 | 101.1 | +101.1 | 0 to 4487 |
| gemm | 223.9 | 177.1 | -46.8 | 8063 to 8063 |
| elementwise | 102.0 | 113.6 | +11.5 | 9038 to 9680 |
| memory | 2.0 | 6.5 | +4.5 | 1627 to 8011 |
| copy | 146.7 | 149.8 | +3.1 | 4610 to 5764 |
| convolution | 74.1 | 71.4 | -2.7 | 533 to 533 |
| other | 37.0 | 35.3 | -1.7 | 623 to 623 |
| norm | 18.1 | 17.6 | -0.5 | 782 to 782 |
| attention | 8.4 | 8.1 | -0.3 | 140 to 140 |
| reduce | 1.6 | 1.5 | -0.1 | 18 to 18 |

Piece groups that changed most:

| group | native eager ms | int8 dx-int8 graph ms | change ms |
|---|---:|---:|---:|
| conv output_blocks.in_layers up | 459.0 | 691.6 | +232.6 |
| linear ff.net up | 168.7 | 382.7 | +214.0 |
| linear attn2.to_k up | 160.9 | 362.8 | +201.9 |
| linear attn2.to_v up | 161.7 | 359.9 | +198.2 |
| conv output_blocks.skip_connection up | 308.0 | 458.3 | +150.3 |
| linear ff.net down | 121.9 | 268.5 | +146.6 |
| linear attn2.to_out up | 180.9 | 37.7 | -143.2 |
| linear attn2.to_k down | 120.9 | 256.2 | +135.3 |
| linear attn2.to_v down | 121.4 | 255.2 | +133.7 |
| linear net.proj up | 175.2 | 43.9 | -131.2 |
| linear attn1.to_out up | 166.3 | 37.7 | -128.6 |
| linear attn1.to_v up | 157.3 | 29.2 | -128.1 |

## native graph measured-checkpoints against native eager

| step | native eager | native graph measured-checkpoints | change |
|---|---:|---:|---:|
| wall_ms | 8442 | 11973 | +3531 |
| eager_ms | 8442 | 11973 | +3531 |
| device_ms | 614 | 617 | +3 |
| launches | 25434 | 42013 | +16579 |
| utilisation | 7% | 5% | -2 points |
| peak_mb | 6125 | 7182 | +1057 |

Device time by category (per step):

| category | native eager ms | native graph measured-checkpoints ms | change ms | calls native eager to native graph measured-checkpoints |
|---|---:|---:|---:|---|
| gemm | 223.9 | 254.0 | +30.2 | 8063 to 14054 |
| copy | 146.7 | 133.8 | -12.9 | 4610 to 7808 |
| elementwise | 102.0 | 90.9 | -11.2 | 9038 to 14654 |
| convolution | 74.1 | 71.6 | -2.5 | 533 to 946 |
| attention | 8.4 | 6.6 | -1.8 | 140 to 160 |
| norm | 18.1 | 18.7 | +0.6 | 782 to 1352 |
| other | 37.0 | 37.5 | +0.4 | 623 to 1214 |
| reduce | 1.6 | 1.4 | -0.2 | 18 to 19 |
| memory | 2.0 | 2.1 | +0.1 | 1627 to 1806 |

Piece groups that changed most:

| group | native eager ms | native graph measured-checkpoints ms | change ms |
|---|---:|---:|---:|
| attention transformer_blocks.attn2 up | 215.7 | 380.6 | +164.9 |
| attention transformer_blocks.attn1 up | 226.4 | 351.9 | +125.5 |
| linear output_blocks.proj_out up | 20.7 | 145.1 | +124.4 |
| linear attn1.to_out up | 166.3 | 284.1 | +117.8 |
| linear attn1.to_q up | 166.5 | 283.8 | +117.4 |
| linear attn1.to_v up | 157.3 | 273.4 | +116.1 |
| linear attn1.to_k up | 162.9 | 278.8 | +115.9 |
| linear attn2.to_q up | 163.6 | 278.6 | +115.0 |
| linear ff.net up | 168.7 | 283.2 | +114.5 |
| linear net.proj up | 175.2 | 287.2 | +112.0 |
| linear attn2.to_out up | 180.9 | 279.4 | +98.5 |
| attention transformer_blocks.attn2 down | 191.1 | 289.2 | +98.1 |

## int8 graph measured-checkpoints against native eager

| step | native eager | int8 graph measured-checkpoints | change |
|---|---:|---:|---:|
| wall_ms | 8442 | 12958 | +4516 |
| eager_ms | 8442 | 12958 | +4516 |
| device_ms | 614 | 727 | +113 |
| launches | 25434 | 49641 | +24207 |
| utilisation | 7% | 6% | -2 points |
| peak_mb | 6125 | 5728 | -397 |

Device time by category (per step):

| category | native eager ms | int8 graph measured-checkpoints ms | change ms | calls native eager to int8 graph measured-checkpoints |
|---|---:|---:|---:|---|
| int8 | 0.0 | 166.7 | +166.7 | 0 to 4456 |
| gemm | 223.9 | 202.1 | -21.8 | 8063 to 13102 |
| copy | 146.7 | 131.0 | -15.7 | 4610 to 7864 |
| elementwise | 102.0 | 88.8 | -13.3 | 9038 to 13870 |
| convolution | 74.1 | 70.1 | -4.0 | 533 to 946 |
| memory | 2.0 | 5.7 | +3.7 | 1627 to 6798 |
| attention | 8.4 | 5.3 | -3.1 | 140 to 104 |
| other | 37.0 | 38.0 | +1.0 | 623 to 1214 |
| norm | 18.1 | 17.6 | -0.4 | 782 to 1268 |
| reduce | 1.6 | 1.4 | -0.2 | 18 to 19 |

Piece groups that changed most:

| group | native eager ms | int8 graph measured-checkpoints ms | change ms |
|---|---:|---:|---:|
| attention transformer_blocks.attn2 up | 215.7 | 446.8 | +231.2 |
| linear ff.net up | 168.7 | 357.2 | +188.6 |
| attention transformer_blocks.attn2 down | 191.1 | 320.2 | +129.1 |
| linear ff.net down | 121.9 | 250.3 | +128.4 |
| linear attn2.to_out up | 180.9 | 52.9 | -128.0 |
| linear net.proj up | 175.2 | 56.1 | -119.0 |
| linear attn1.to_out up | 166.3 | 52.6 | -113.7 |
| linear attn1.to_q up | 166.5 | 52.9 | -113.6 |
| linear attn2.to_v up | 161.7 | 273.9 | +112.2 |
| attention transformer_blocks.attn1 up | 226.4 | 338.3 | +111.9 |
| linear attn2.to_q up | 163.6 | 52.6 | -111.0 |
| linear attn1.to_k up | 162.9 | 52.2 | -110.7 |

## native graph attention-math_bf16 against native eager

| step | native eager | native graph attention-math_bf16 | change |
|---|---:|---:|---:|
| wall_ms | 8442 | 7381 | -1061 |
| eager_ms | 8442 | 6003 | -2439 |
| replayed_ms | - | 10136 | - |
| device_ms | 614 | 598 | -16 |
| launches | 25434 | 25434 | +0 |
| utilisation | 7% | 8% | +1 points |
| peak_mb | 6125 | 6126 | +1 |

Device time by category (per step):

| category | native eager ms | native graph attention-math_bf16 ms | change ms | calls native eager to native graph attention-math_bf16 |
|---|---:|---:|---:|---|
| elementwise | 102.0 | 98.2 | -3.8 | 9038 to 9038 |
| copy | 146.7 | 143.2 | -3.5 | 4610 to 4610 |
| convolution | 74.1 | 70.7 | -3.3 | 533 to 533 |
| other | 37.0 | 34.7 | -2.3 | 623 to 623 |
| gemm | 223.9 | 222.6 | -1.2 | 8063 to 8063 |
| norm | 18.1 | 17.1 | -0.9 | 782 to 782 |
| attention | 8.4 | 7.9 | -0.5 | 140 to 140 |
| memory | 2.0 | 2.1 | +0.1 | 1627 to 1627 |

Piece groups that changed most:

| group | native eager ms | native graph attention-math_bf16 ms | change ms |
|---|---:|---:|---:|
| conv output_blocks.in_layers up | 459.0 | 671.5 | +212.5 |
| conv output_blocks.skip_connection up | 308.0 | 449.8 | +141.8 |
| conv input_blocks.op down | 196.3 | 280.5 | +84.2 |
| attention transformer_blocks.attn1 up | 226.4 | 295.9 | +69.5 |
| linear ff.net up | 168.7 | 236.6 | +68.0 |
| conv input_blocks.in_layers down | 149.6 | 213.6 | +64.0 |
| attention transformer_blocks.attn2 up | 215.7 | 278.9 | +63.3 |
| linear net.proj up | 175.2 | 234.6 | +59.4 |
| attention transformer_blocks.attn2 down | 191.1 | 242.5 | +51.4 |
| attention transformer_blocks.attn1 down | 198.7 | 249.8 | +51.1 |
| linear attn1.to_out up | 166.3 | 215.6 | +49.3 |
| linear attn2.to_q up | 163.6 | 212.7 | +49.1 |

## native graph attention-chunked_bf16 against native eager

| step | native eager | native graph attention-chunked_bf16 | change |
|---|---:|---:|---:|
| wall_ms | 8442 | 7382 | -1060 |
| eager_ms | 8442 | 6002 | -2440 |
| replayed_ms | - | 10142 | - |
| device_ms | 614 | 595 | -19 |
| launches | 25434 | 25434 | +0 |
| utilisation | 7% | 8% | +1 points |
| peak_mb | 6125 | 6126 | +1 |

Device time by category (per step):

| category | native eager ms | native graph attention-chunked_bf16 ms | change ms | calls native eager to native graph attention-chunked_bf16 |
|---|---:|---:|---:|---|
| copy | 146.7 | 141.4 | -5.3 | 4610 to 4610 |
| convolution | 74.1 | 69.8 | -4.3 | 533 to 533 |
| elementwise | 102.0 | 98.1 | -4.0 | 9038 to 9038 |
| other | 37.0 | 34.8 | -2.2 | 623 to 623 |
| gemm | 223.9 | 222.8 | -1.1 | 8063 to 8063 |
| norm | 18.1 | 17.0 | -1.0 | 782 to 782 |
| attention | 8.4 | 7.8 | -0.5 | 140 to 140 |
| reduce | 1.6 | 1.5 | -0.1 | 18 to 18 |

Piece groups that changed most:

| group | native eager ms | native graph attention-chunked_bf16 ms | change ms |
|---|---:|---:|---:|
| conv output_blocks.in_layers up | 459.0 | 667.5 | +208.5 |
| conv output_blocks.skip_connection up | 308.0 | 447.4 | +139.4 |
| conv input_blocks.op down | 196.3 | 282.0 | +85.7 |
| attention transformer_blocks.attn1 up | 226.4 | 300.6 | +74.2 |
| conv input_blocks.in_layers down | 149.6 | 214.8 | +65.2 |
| linear attn1.to_q up | 166.5 | 230.3 | +63.9 |
| attention transformer_blocks.attn2 up | 215.7 | 278.3 | +62.7 |
| linear net.proj up | 175.2 | 231.9 | +56.7 |
| linear attn1.to_out up | 166.3 | 221.0 | +54.7 |
| linear ff.net up | 168.7 | 220.8 | +52.1 |
| linear attn2.to_q up | 163.6 | 214.8 | +51.2 |
| attention transformer_blocks.attn1 down | 198.7 | 249.2 | +50.5 |

## int8 graph attention-chunked_bf16 against native eager

| step | native eager | int8 graph attention-chunked_bf16 | change |
|---|---:|---:|---:|
| wall_ms | 8442 | 8688 | +246 |
| eager_ms | 8442 | 7262 | -1180 |
| replayed_ms | - | 11541 | - |
| device_ms | 614 | 695 | +81 |
| launches | 25434 | 32959 | +7525 |
| utilisation | 7% | 8% | +1 points |
| peak_mb | 6125 | 4067 | -2058 |

Device time by category (per step):

| category | native eager ms | int8 graph attention-chunked_bf16 ms | change ms | calls native eager to int8 graph attention-chunked_bf16 |
|---|---:|---:|---:|---|
| int8 | 0.0 | 120.5 | +120.5 | 0 to 3203 |
| gemm | 223.9 | 192.2 | -31.6 | 8063 to 8063 |
| convolution | 74.1 | 69.7 | -4.4 | 533 to 533 |
| memory | 2.0 | 4.8 | +2.8 | 1627 to 5443 |
| other | 37.0 | 34.7 | -2.3 | 623 to 623 |
| elementwise | 102.0 | 100.1 | -2.0 | 9038 to 9038 |
| norm | 18.1 | 17.2 | -0.9 | 782 to 782 |
| attention | 8.4 | 8.0 | -0.4 | 140 to 140 |
| copy | 146.7 | 146.5 | -0.2 | 4610 to 5116 |
| reduce | 1.6 | 1.5 | -0.1 | 18 to 18 |

Piece groups that changed most:

| group | native eager ms | int8 graph attention-chunked_bf16 ms | change ms |
|---|---:|---:|---:|
| conv output_blocks.in_layers up | 459.0 | 686.3 | +227.3 |
| linear attn2.to_v up | 161.7 | 307.0 | +145.4 |
| linear ff.net up | 168.7 | 313.6 | +144.9 |
| linear attn2.to_out up | 180.9 | 36.6 | -144.3 |
| conv output_blocks.skip_connection up | 308.0 | 452.2 | +144.2 |
| linear attn2.to_k up | 160.9 | 296.4 | +135.5 |
| linear net.proj up | 175.2 | 43.4 | -131.8 |
| linear attn1.to_q up | 166.5 | 36.6 | -129.9 |
| linear attn1.to_out up | 166.3 | 36.6 | -129.7 |
| linear attn1.to_v up | 157.3 | 28.2 | -129.1 |
| linear attn1.to_k up | 162.9 | 36.0 | -126.9 |
| linear attn2.to_q up | 163.6 | 36.8 | -126.7 |

## native expandable-segments against native eager

| step | native eager | native expandable-segments | change |
|---|---:|---:|---:|
| wall_ms | 8442 | 7627 | -815 |
| eager_ms | 8442 | 7627 | -815 |
| device_ms | 614 | 642 | +29 |
| launches | 25434 | 25434 | +0 |
| utilisation | 7% | 8% | +1 points |
| peak_mb | 6125 | 6125 | +0 |

Device time by category (per step):

| category | native eager ms | native expandable-segments ms | change ms | calls native eager to native expandable-segments |
|---|---:|---:|---:|---|
| copy | 146.7 | 160.8 | +14.1 | 4610 to 4610 |
| norm | 18.1 | 27.2 | +9.1 | 782 to 782 |
| elementwise | 102.0 | 105.9 | +3.8 | 9038 to 9038 |
| convolution | 74.1 | 76.8 | +2.7 | 533 to 533 |
| other | 37.0 | 35.6 | -1.4 | 623 to 623 |
| attention | 8.4 | 9.4 | +1.0 | 140 to 140 |
| gemm | 223.9 | 223.1 | -0.8 | 8063 to 8063 |
| reduce | 1.6 | 1.5 | -0.1 | 18 to 18 |
| memory | 2.0 | 2.1 | +0.1 | 1627 to 1627 |

Piece groups that changed most:

| group | native eager ms | native expandable-segments ms | change ms |
|---|---:|---:|---:|
| attention transformer_blocks.attn1 down | 198.7 | 168.8 | -29.9 |
| attention transformer_blocks.attn2 up | 215.7 | 186.7 | -29.0 |
| linear attn2.to_out up | 180.9 | 152.1 | -28.8 |
| attention transformer_blocks.attn2 down | 191.1 | 164.2 | -26.8 |
| attention transformer_blocks.attn1 up | 226.4 | 200.5 | -25.9 |
| linear attn1.to_v down | 119.1 | 101.7 | -17.4 |
| linear attn1.to_q down | 121.6 | 104.7 | -16.9 |
| linear attn2.to_v down | 121.4 | 104.6 | -16.8 |
| feedforward transformer_blocks.ff down | 113.2 | 96.6 | -16.6 |
| linear net.proj up | 175.2 | 159.0 | -16.1 |
| linear attn1.to_out up | 166.3 | 150.2 | -16.1 |
| linear attn1.to_q up | 166.5 | 151.1 | -15.4 |

## native no-expandable-segments against native eager

| step | native eager | native no-expandable-segments | change |
|---|---:|---:|---:|
| wall_ms | 8442 | 7399 | -1043 |
| eager_ms | 8442 | 7399 | -1043 |
| device_ms | 614 | 603 | -11 |
| launches | 25434 | 25434 | +0 |
| utilisation | 7% | 8% | +1 points |
| peak_mb | 6125 | 6244 | +119 |

Device time by category (per step):

| category | native eager ms | native no-expandable-segments ms | change ms | calls native eager to native no-expandable-segments |
|---|---:|---:|---:|---|
| other | 37.0 | 32.5 | -4.6 | 623 to 623 |
| convolution | 74.1 | 71.2 | -2.9 | 533 to 533 |
| elementwise | 102.0 | 100.0 | -2.0 | 9038 to 9038 |
| copy | 146.7 | 144.9 | -1.8 | 4610 to 4610 |
| gemm | 223.9 | 225.1 | +1.3 | 8063 to 8063 |
| norm | 18.1 | 17.4 | -0.7 | 782 to 782 |
| reduce | 1.6 | 1.4 | -0.2 | 18 to 18 |
| attention | 8.4 | 8.2 | -0.1 | 140 to 140 |

Piece groups that changed most:

| group | native eager ms | native no-expandable-segments ms | change ms |
|---|---:|---:|---:|
| linear attn2.to_out up | 180.9 | 144.0 | -36.9 |
| attention transformer_blocks.attn1 down | 198.7 | 162.8 | -36.0 |
| attention transformer_blocks.attn2 down | 191.1 | 156.6 | -34.5 |
| attention transformer_blocks.attn1 up | 226.4 | 192.0 | -34.5 |
| attention transformer_blocks.attn2 up | 215.7 | 183.9 | -31.8 |
| linear ff.net up | 168.7 | 145.3 | -23.4 |
| feedforward transformer_blocks.ff down | 113.2 | 90.0 | -23.3 |
| linear attn1.to_out up | 166.3 | 145.2 | -21.1 |
| linear net.proj up | 175.2 | 154.1 | -21.1 |
| linear attn1.to_v up | 157.3 | 136.7 | -20.7 |
| linear attn1.to_k up | 162.9 | 142.5 | -20.4 |
| feedforward transformer_blocks.ff up | 130.4 | 110.6 | -19.8 |

