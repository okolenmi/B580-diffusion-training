# Matrix summary

| scenario | status | wall ms | eager ms | replayed ms | device ms | utilisation | peak MB | top finding |
|---|---|---:|---:|---:|---:|---:|---:|---|
| native eager | ok | 949 | 949 | - | 656 | 69% | 6303 | Much of the module time has no cost model |
| native graph | FAILED: exit code 1 | | | | | | | log: perf-bundle/matrix/native_graph/run.log |
| int8 eager | ok | 1360 | 1360 | - | 778 | 57% | 4244 | Recomputing activations is a large part of the step |
| int8 graph | FAILED: exit code 1 | | | | | | | log: perf-bundle/matrix/int8_graph/run.log |
| int8 dx-int8 eager | ok | 1500 | 1500 | - | 758 | 51% | 4251 | Much of the module time has no cost model |
| int8 dx-int8 graph | FAILED: exit code 1 | | | | | | | log: perf-bundle/matrix/int8_dx-int8_graph/run.log |
| native graph measured-checkpoints | ok | 1672 | 1672 | - | 641 | 38% | 7604 | Much of the module time has no cost model |
| int8 graph measured-checkpoints | ok | 2055 | 2055 | - | 750 | 37% | 6381 | Much of the module time has no cost model |
| native graph attention-math_bf16 | FAILED: exit code 1 | | | | | | | log: perf-bundle/matrix/native_graph_attention-math_bf16/run.log |
| native graph attention-chunked_bf16 | FAILED: exit code 1 | | | | | | | log: perf-bundle/matrix/native_graph_attention-chunked_bf16/run.log |
| int8 graph attention-chunked_bf16 | FAILED: exit code 1 | | | | | | | log: perf-bundle/matrix/int8_graph_attention-chunked_bf16/run.log |
| native expandable-segments | ok | 1024 | 1024 | - | 664 | 65% | 6303 | Much of the module time has no cost model |
| native no-expandable-segments | ok | 1021 | 1021 | - | 666 | 65% | 6411 | Much of the module time has no cost model |

Every other scenario below is compared with the first, `native eager`.

## int8 eager against native eager

| step | native eager | int8 eager | change |
|---|---:|---:|---:|
| wall_ms | 949 | 1360 | +411 |
| eager_ms | 949 | 1360 | +411 |
| device_ms | 656 | 778 | +122 |
| launches | 25413 | 32938 | +7525 |
| utilisation | 69% | 57% | -12 points |
| peak_mb | 6303 | 4244 | -2059 |

Device time by category (per step):

| category | native eager ms | int8 eager ms | change ms | calls native eager to int8 eager |
|---|---:|---:|---:|---|
| int8 | 0.0 | 142.7 | +142.7 | 0 to 3203 |
| gemm | 243.8 | 214.1 | -29.7 | 8063 to 8063 |
| copy | 160.3 | 164.0 | +3.6 | 4610 to 5116 |
| elementwise | 114.5 | 117.6 | +3.0 | 9051 to 9051 |
| memory | 2.1 | 4.6 | +2.4 | 1627 to 5443 |
| other | 44.3 | 44.6 | +0.3 | 610 to 610 |
| convolution | 61.4 | 61.2 | -0.1 | 512 to 512 |
| norm | 18.9 | 19.0 | +0.1 | 782 to 782 |

Piece groups that changed most:

| group | native eager ms | int8 eager ms | change ms |
|---|---:|---:|---:|
| linear net.proj up | 150.6 | 28.2 | -122.5 |
| linear attn1.to_q up | 137.0 | 23.6 | -113.4 |
| linear attn1.to_out up | 136.0 | 22.6 | -113.3 |
| linear attn2.to_q up | 135.9 | 23.4 | -112.6 |
| linear attn2.to_out up | 136.4 | 23.9 | -112.5 |
| linear attn1.to_v up | 127.8 | 16.0 | -111.8 |
| linear attn1.to_k up | 133.6 | 22.4 | -111.2 |
| linear net.proj down | 106.1 | 20.6 | -85.5 |
| linear attn1.to_q down | 97.9 | 17.6 | -80.3 |
| linear attn2.to_out down | 97.6 | 17.7 | -80.0 |
| linear attn2.to_q down | 96.8 | 17.3 | -79.4 |
| linear attn1.to_out down | 96.2 | 17.0 | -79.2 |

## int8 dx-int8 eager against native eager

| step | native eager | int8 dx-int8 eager | change |
|---|---:|---:|---:|
| wall_ms | 949 | 1500 | +550 |
| eager_ms | 949 | 1500 | +550 |
| device_ms | 656 | 758 | +103 |
| launches | 25413 | 38080 | +12667 |
| utilisation | 69% | 51% | -18 points |
| peak_mb | 6303 | 4251 | -2052 |

Device time by category (per step):

| category | native eager ms | int8 dx-int8 eager ms | change ms | calls native eager to int8 dx-int8 eager |
|---|---:|---:|---:|---|
| int8 | 0.0 | 120.6 | +120.6 | 0 to 4487 |
| gemm | 243.8 | 200.0 | -43.7 | 8063 to 8063 |
| elementwise | 114.5 | 130.4 | +15.9 | 9051 to 9693 |
| copy | 160.3 | 165.2 | +4.8 | 4610 to 5764 |
| memory | 2.1 | 6.4 | +4.3 | 1627 to 8011 |
| norm | 18.9 | 19.3 | +0.4 | 782 to 782 |
| other | 44.3 | 44.5 | +0.2 | 610 to 610 |
| convolution | 61.4 | 61.4 | +0.1 | 512 to 512 |

Piece groups that changed most:

| group | native eager ms | int8 dx-int8 eager ms | change ms |
|---|---:|---:|---:|
| linear net.proj up | 150.6 | 27.8 | -122.8 |
| linear attn1.to_q up | 137.0 | 23.4 | -113.6 |
| linear attn1.to_out up | 136.0 | 22.4 | -113.5 |
| linear attn2.to_out up | 136.4 | 23.3 | -113.1 |
| linear attn2.to_q up | 135.9 | 24.0 | -111.9 |
| linear attn1.to_v up | 127.8 | 16.1 | -111.8 |
| linear attn1.to_k up | 133.6 | 22.0 | -111.5 |
| linear ff.net up | 135.6 | 229.4 | +93.8 |
| linear attn2.to_k up | 133.7 | 218.7 | +85.0 |
| linear attn1.to_q down | 97.9 | 17.0 | -80.8 |
| linear attn2.to_out down | 97.6 | 17.1 | -80.5 |
| linear attn1.to_out down | 96.2 | 16.2 | -80.0 |

## native graph measured-checkpoints against native eager

| step | native eager | native graph measured-checkpoints | change |
|---|---:|---:|---:|
| wall_ms | 949 | 1672 | +723 |
| eager_ms | 949 | 1672 | +723 |
| device_ms | 656 | 641 | -14 |
| launches | 25413 | 42155 | +16742 |
| utilisation | 69% | 38% | -31 points |
| peak_mb | 6303 | 7604 | +1301 |

Device time by category (per step):

| category | native eager ms | native graph measured-checkpoints ms | change ms | calls native eager to native graph measured-checkpoints |
|---|---:|---:|---:|---|
| gemm | 243.8 | 264.4 | +20.7 | 8063 to 14122 |
| copy | 160.3 | 143.1 | -17.3 | 4610 to 7852 |
| elementwise | 114.5 | 98.3 | -16.2 | 9051 to 14737 |
| attention | 8.8 | 6.6 | -2.2 | 140 to 164 |
| other | 44.3 | 44.8 | +0.5 | 610 to 1187 |
| norm | 18.9 | 19.2 | +0.3 | 782 to 1358 |
| convolution | 61.4 | 61.3 | -0.1 | 512 to 910 |

Piece groups that changed most:

| group | native eager ms | native graph measured-checkpoints ms | change ms |
|---|---:|---:|---:|
| attention transformer_blocks.attn2 up | 165.9 | 335.9 | +170.0 |
| linear attn1.to_k up | 133.6 | 267.5 | +134.0 |
| attention transformer_blocks.attn1 up | 174.3 | 298.5 | +124.2 |
| attention transformer_blocks.attn2 down | 127.5 | 246.6 | +119.0 |
| linear ff.net up | 135.6 | 253.6 | +118.0 |
| linear attn1.to_out up | 136.0 | 251.6 | +115.6 |
| linear attn2.to_out up | 136.4 | 251.6 | +115.2 |
| linear attn1.to_q up | 137.0 | 250.7 | +113.7 |
| linear attn1.to_v up | 127.8 | 240.8 | +112.9 |
| linear attn2.to_q up | 135.9 | 248.7 | +112.8 |
| linear net.proj up | 150.6 | 261.6 | +111.0 |
| linear attn1.to_out down | 96.2 | 183.3 | +87.0 |

## int8 graph measured-checkpoints against native eager

| step | native eager | int8 graph measured-checkpoints | change |
|---|---:|---:|---:|
| wall_ms | 949 | 2055 | +1105 |
| eager_ms | 949 | 2055 | +1105 |
| device_ms | 656 | 750 | +95 |
| launches | 25413 | 49605 | +24192 |
| utilisation | 69% | 37% | -33 points |
| peak_mb | 6303 | 6381 | +78 |

Device time by category (per step):

| category | native eager ms | int8 graph measured-checkpoints ms | change ms | calls native eager to int8 graph measured-checkpoints |
|---|---:|---:|---:|---|
| int8 | 0.0 | 167.0 | +167.0 | 0 to 4456 |
| gemm | 243.8 | 211.1 | -32.7 | 8063 to 13102 |
| copy | 160.3 | 138.4 | -22.0 | 4610 to 7864 |
| elementwise | 114.5 | 96.7 | -17.8 | 9051 to 13897 |
| attention | 8.8 | 5.4 | -3.4 | 140 to 104 |
| memory | 2.1 | 5.0 | +2.9 | 1627 to 6798 |
| other | 44.3 | 45.3 | +1.0 | 610 to 1187 |
| norm | 18.9 | 18.2 | -0.7 | 782 to 1268 |
| convolution | 61.4 | 61.7 | +0.4 | 512 to 910 |

Piece groups that changed most:

| group | native eager ms | int8 graph measured-checkpoints ms | change ms |
|---|---:|---:|---:|
| linear ff.net up | 135.6 | 335.9 | +200.3 |
| attention transformer_blocks.attn2 up | 165.9 | 350.6 | +184.7 |
| linear ff.net down | 97.2 | 241.0 | +143.8 |
| attention transformer_blocks.attn2 down | 127.5 | 246.4 | +118.9 |
| linear attn2.to_v down | 95.0 | 196.6 | +101.6 |
| linear net.proj up | 150.6 | 50.7 | -99.9 |
| linear attn2.to_k down | 95.5 | 189.6 | +94.2 |
| attention transformer_blocks.attn1 up | 174.3 | 268.1 | +93.8 |
| linear attn2.to_out up | 136.4 | 47.9 | -88.5 |
| linear attn1.to_out up | 136.0 | 47.7 | -88.3 |
| linear attn2.to_q up | 135.9 | 48.0 | -87.9 |
| linear attn1.to_v up | 127.8 | 40.7 | -87.2 |

## native expandable-segments against native eager

| step | native eager | native expandable-segments | change |
|---|---:|---:|---:|
| wall_ms | 949 | 1024 | +75 |
| eager_ms | 949 | 1024 | +75 |
| device_ms | 656 | 664 | +8 |
| launches | 25413 | 25413 | +0 |
| utilisation | 69% | 65% | -4 points |
| peak_mb | 6303 | 6303 | +0 |

Device time by category (per step):

| category | native eager ms | native expandable-segments ms | change ms | calls native eager to native expandable-segments |
|---|---:|---:|---:|---|
| copy | 160.3 | 166.3 | +6.0 | 4610 to 4610 |
| gemm | 243.8 | 241.9 | -1.8 | 8063 to 8063 |
| elementwise | 114.5 | 116.3 | +1.8 | 9051 to 9051 |
| norm | 18.9 | 19.7 | +0.8 | 782 to 782 |
| other | 44.3 | 45.0 | +0.7 | 610 to 610 |
| convolution | 61.4 | 61.7 | +0.3 | 512 to 512 |
| attention | 8.8 | 9.1 | +0.3 | 140 to 140 |

Piece groups that changed most:

| group | native eager ms | native expandable-segments ms | change ms |
|---|---:|---:|---:|
| attention transformer_blocks.attn1 down | 135.7 | 158.3 | +22.6 |
| attention transformer_blocks.attn2 down | 127.5 | 149.7 | +22.1 |
| linear attn1.to_v up | 127.8 | 145.8 | +18.0 |
| attention transformer_blocks.attn1 up | 174.3 | 190.8 | +16.5 |
| feedforward transformer_blocks.ff down | 76.5 | 89.9 | +13.4 |
| attention transformer_blocks.attn2 up | 165.9 | 178.9 | +13.0 |
| linear attn2.to_k down | 95.5 | 107.4 | +12.0 |
| feedforward transformer_blocks.ff up | 97.1 | 108.6 | +11.5 |
| linear attn2.to_k up | 133.7 | 145.2 | +11.5 |
| linear ff.net up | 135.6 | 146.8 | +11.1 |
| linear attn1.to_q up | 137.0 | 147.6 | +10.6 |
| linear attn1.to_out up | 136.0 | 146.3 | +10.4 |

## native no-expandable-segments against native eager

| step | native eager | native no-expandable-segments | change |
|---|---:|---:|---:|
| wall_ms | 949 | 1021 | +71 |
| eager_ms | 949 | 1021 | +71 |
| device_ms | 656 | 666 | +11 |
| launches | 25413 | 25413 | +0 |
| utilisation | 69% | 65% | -4 points |
| peak_mb | 6303 | 6411 | +108 |

Device time by category (per step):

| category | native eager ms | native no-expandable-segments ms | change ms | calls native eager to native no-expandable-segments |
|---|---:|---:|---:|---|
| copy | 160.3 | 167.5 | +7.1 | 4610 to 4610 |
| elementwise | 114.5 | 116.8 | +2.3 | 9051 to 9051 |
| other | 44.3 | 42.4 | -1.9 | 610 to 610 |
| gemm | 243.8 | 244.8 | +1.0 | 8063 to 8063 |
| norm | 18.9 | 19.9 | +1.0 | 782 to 782 |
| convolution | 61.4 | 62.1 | +0.8 | 512 to 512 |
| attention | 8.8 | 9.1 | +0.3 | 140 to 140 |
| memory | 2.1 | 2.1 | -0.1 | 1627 to 1627 |

Piece groups that changed most:

| group | native eager ms | native no-expandable-segments ms | change ms |
|---|---:|---:|---:|
| attention transformer_blocks.attn1 down | 135.7 | 159.5 | +23.8 |
| attention transformer_blocks.attn2 down | 127.5 | 149.0 | +21.5 |
| linear net.proj up | 150.6 | 169.8 | +19.2 |
| attention transformer_blocks.attn1 up | 174.3 | 191.4 | +17.1 |
| linear attn1.to_q up | 137.0 | 152.9 | +15.9 |
| attention transformer_blocks.attn2 up | 165.9 | 179.5 | +13.6 |
| linear attn2.to_out down | 97.6 | 111.0 | +13.3 |
| linear attn1.to_k up | 133.6 | 146.8 | +13.2 |
| feedforward transformer_blocks.ff down | 76.5 | 88.9 | +12.4 |
| linear attn1.to_k down | 95.3 | 107.5 | +12.1 |
| feedforward transformer_blocks.ff up | 97.1 | 107.9 | +10.9 |
| linear attn2.to_out up | 136.4 | 147.2 | +10.8 |

